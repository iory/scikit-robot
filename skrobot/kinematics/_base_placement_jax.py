"""JAX version of the shared-base Levenberg-Marquardt loop.

The NumPy loop in :mod:`skrobot.kinematics.base_placement` drops finished
starts from the batch every iteration, which keeps the work small but
spends most of its time in Python and in small ``einsum`` calls. Here one
iteration -- linearize, Schur step, trial residual, accept or reject --
is a single jitted function over a fixed-size batch. The batch shrinks to
the next power of sixteen when fifteen sixteenths of it have finished,
so a solve compiles for two or three sizes only; each costs seconds,
which is why the sizes are spaced so far apart.

Everything runs in float64, inside ``jax.enable_x64`` so the global
setting is left alone. In float32 the accept and stall tests round
differently, and the solve takes a different path than the NumPy one.

Targets that follow the base are evaluated on the host, as in the NumPy
loop, together with their finite-difference derivative; within one
iteration the trial step moves them along that derivative.
"""

import numpy as np


def _rodrigues(jnp, twist):
    """``(..., 3, 3)`` rotation for ``(..., 3)`` axis-angle vectors."""
    theta = jnp.linalg.norm(twist, axis=-1, keepdims=True)
    axis = twist / jnp.maximum(theta, 1e-12)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = jnp.zeros_like(x)
    k = jnp.stack([
        jnp.stack([zero, -z, y], axis=-1),
        jnp.stack([z, zero, -x], axis=-1),
        jnp.stack([-y, x, zero], axis=-1),
    ], axis=-2)
    s = jnp.sin(theta)[..., None]
    c = jnp.cos(theta)[..., None]
    return jnp.eye(3) + s * k + (1.0 - c) * (k @ k)


def _rotation_log(jnp, actual, target):
    """Body-frame log of ``actual^T @ target``; see the NumPy version."""
    r = jnp.swapaxes(actual, -1, -2) @ target
    vee = 0.5 * jnp.stack([
        r[..., 2, 1] - r[..., 1, 2],
        r[..., 0, 2] - r[..., 2, 0],
        r[..., 1, 0] - r[..., 0, 1],
    ], axis=-1)
    cos_theta = jnp.clip((jnp.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0,
                         -1.0, 1.0)
    sin_theta = jnp.linalg.norm(vee, axis=-1)
    theta = jnp.arctan2(sin_theta, cos_theta)
    safe = sin_theta > 1e-6
    scale = jnp.where(safe, theta / jnp.where(safe, sin_theta, 1.0), 1.0)
    log = vee * scale[..., None]
    # Half turn: the axis is the dominant column of the symmetric part.
    sym = (r + jnp.eye(3)) / 2.0
    col = jnp.argmax(jnp.diagonal(sym, axis1=-2, axis2=-1), axis=-1)
    axis = jnp.take_along_axis(sym, col[..., None, None], axis=-1)[..., 0]
    axis = axis / jnp.maximum(
        jnp.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    sign = jnp.sign(jnp.sum(axis * vee, axis=-1))
    sign = jnp.where(sign == 0, 1.0, sign)
    near_pi = jnp.logical_and(jnp.logical_not(safe), cos_theta < 0)
    return jnp.where(near_pi[..., None],
                     theta[..., None] * axis * sign[..., None], log)


class JaxLevenbergMarquardt(object):
    """Jitted LM iterations for one :class:`_SharedBaseProblem`."""

    def __init__(self, problem, thre, rthre, max_pos_step, max_rot_step,
                 bounds):
        import jax
        import jax.numpy as jnp

        from skrobot.kinematics.differentiable import compute_geometric_jacobian_jax

        self.jax = jax
        self.jnp = jnp
        self.problem = problem
        fk = problem.fk_params
        n_joints = fk['n_joints']
        non_mimic = problem.non_mimic
        n_opt = problem.n_opt
        n_targets = problem.n_targets

        jac_one = jax.vmap(lambda a: compute_geometric_jacobian_jax(a, fk))
        row_mask = jnp.asarray(problem.row_mask)
        tolerance = (None if problem.rotation_tolerance is None
                     else jnp.asarray(problem.rotation_tolerance))
        base_frame = jnp.asarray(problem.base_frame)
        lower = jnp.asarray(problem.lower)
        upper = jnp.asarray(problem.upper)
        unbounded = jnp.asarray(problem.unbounded)
        bounded = jnp.asarray(problem.bounded)
        span = jnp.asarray(problem.span)
        margin_weight = problem.margin_weight
        has_moving = bool(problem.target_fns)
        moving_idx = jnp.asarray([i for i, _ in problem.target_fns],
                                 dtype=jnp.int32)
        # The fixed targets stay on the device; only the few that follow
        # the base cross over each iteration.
        static_pos = jnp.asarray(problem.target_pos)
        static_rot = jnp.asarray(problem.target_rot)
        lo = None if bounds is None else jnp.asarray(bounds[0])
        hi = None if bounds is None else jnp.asarray(bounds[1])

        def base_rotations(yaw):
            c, s = jnp.cos(yaw), jnp.sin(yaw)
            z, o = jnp.zeros_like(yaw), jnp.ones_like(yaw)
            return jnp.stack([jnp.stack([c, -s, z], -1),
                              jnp.stack([s, c, z], -1),
                              jnp.stack([z, z, o], -1)], -2)

        def chain(q):
            n_cand = q.shape[0]
            angles = jnp.zeros((n_cand * n_targets, n_joints), q.dtype)
            angles = angles.at[:, non_mimic].set(q.reshape(-1, n_opt))
            jac, pos, rot = jac_one(angles)
            return (jac.reshape(n_cand, n_targets, 6, n_opt),
                    pos.reshape(n_cand, n_targets, 3),
                    rot.reshape(n_cand, n_targets, 3, 3))

        def ee_world(bases, q):
            jac, pos, rot = chain(q)
            base_rot = base_rotations(bases[:, 2])
            base_pos = bases[:, jnp.array([0, 1, 3])]
            world_pos = jnp.einsum('kij,knj->kni', base_rot, pos) \
                + base_pos[:, None, :]
            world_rot = base_rot[:, None] @ rot
            sel = base_frame[None, :, None]
            meas_pos = jnp.where(sel, pos, world_pos)
            meas_rot = jnp.where(sel[..., None], rot, world_rot)
            return (jac, meas_pos, meas_rot, world_pos, world_rot, base_rot,
                    base_pos)

        def residual_from(meas_pos, meas_rot, tpos, trot):
            pos_err = jnp.einsum('knji,knj->kni', meas_rot, tpos - meas_pos)
            rot_err = _rotation_log(jnp, meas_rot, trot)
            if tolerance is not None:
                rot_err = jnp.where(jnp.abs(rot_err) <= tolerance, 0.0,
                                    rot_err)
            return jnp.concatenate([pos_err, rot_err], -1) * row_mask

        def linearize(bases, q, tpos, trot, dt):
            (jac, meas_pos, meas_rot, world_pos, world_rot, base_rot,
             base_pos) = ee_world(bases, q)
            err = residual_from(meas_pos, meas_rot, tpos, trot)
            arm_jac = jnp.concatenate([
                base_rot[:, None] @ jac[:, :, :3],
                base_rot[:, None] @ jac[:, :, 3:]], axis=2)
            rel = world_pos - base_pos[:, None, :]
            one = jnp.ones(rel.shape[:2])
            zero = jnp.zeros(rel.shape[:2])
            base_jac = jnp.stack([
                jnp.stack([one, zero, -rel[..., 1]], -1),
                jnp.stack([zero, one, rel[..., 0]], -1),
                jnp.stack([zero, zero, zero], -1),
                jnp.stack([zero, zero, zero], -1),
                jnp.stack([zero, zero, zero], -1),
                jnp.stack([zero, zero, one], -1),
            ], -2)
            base_jac = jnp.where(base_frame[None, :, None, None], 0.0,
                                 base_jac)
            if has_moving:
                base_jac = base_jac - dt
            rot_t = jnp.swapaxes(world_rot, -1, -2)

            def to_hand(j6):
                return jnp.concatenate(
                    [rot_t @ j6[:, :, :3], rot_t @ j6[:, :, 3:]], axis=2)

            mask = row_mask[..., None]
            if tolerance is not None:
                live = jnp.concatenate(
                    [jnp.ones(err.shape[:-1] + (3,), bool),
                     err[..., 3:] != 0.0], -1)
                mask = mask * live[..., None]
            return err, to_hand(base_jac) * mask, to_hand(arm_jac) * mask

        def margin_terms(q, floor):
            to_lower = (q - lower) / span
            to_upper = (upper - q) / span
            margin = jnp.minimum(to_lower, to_upper)
            slope = jnp.where(to_lower < to_upper, 1.0, -1.0) / span
            target = floor[:, None, None]
            active = bounded & (margin < target)
            m_err = jnp.where(active, margin_weight * (target - margin), 0.0)
            m_jac = jnp.where(active, margin_weight * slope, 0.0)
            return m_err, m_jac

        def clip_q(q):
            return jnp.where(unbounded, q, jnp.clip(q, lower, upper))

        def clip_rows(err):
            pn = jnp.linalg.norm(err[..., :3], axis=-1, keepdims=True)
            rn = jnp.linalg.norm(err[..., 3:], axis=-1, keepdims=True)
            ps = jnp.minimum(1.0, max_pos_step / jnp.maximum(pn, 1e-12))
            rs = jnp.minimum(1.0, max_rot_step / jnp.maximum(rn, 1e-12))
            return jnp.concatenate([err[..., :3] * ps, err[..., 3:] * rs], -1)

        def schur(err, bj, aj, damping, base_weight, m_err, m_jac):
            bj = bj * base_weight
            h_aa = jnp.einsum('knri,knrj->knij', aj, aj) \
                + damping[:, None, None, None] * jnp.eye(n_opt)
            h_aa = h_aa + jax.vmap(jax.vmap(jnp.diag))(m_jac ** 2)
            h_ba = jnp.einsum('knri,knrj->knij', bj, aj)
            g_a = jnp.einsum('knri,knr->kni', aj, err) + m_jac * m_err
            g_b = jnp.einsum('knri,knr->ki', bj, err)
            h_bb = jnp.einsum('knri,knrj->kij', bj, bj) \
                + damping[:, None, None] * jnp.eye(3)
            rhs = jnp.concatenate(
                [jnp.swapaxes(h_ba, -1, -2), g_a[..., None]], -1)
            sol = jnp.linalg.solve(h_aa, rhs)
            inv_h_ab, inv_g_a = sol[..., :3], sol[..., 3]
            s_mat = h_bb - jnp.einsum('knij,knjl->kil', h_ba, inv_h_ab)
            red = g_b - jnp.einsum('knij,knj->ki', h_ba, inv_g_a)
            d_base = jnp.linalg.solve(s_mat, red[..., None])[..., 0]
            d_arm = inv_g_a - jnp.einsum('knij,kj->kni', inv_h_ab, d_base)
            return d_base * base_weight, d_arm

        def cost_of(err, weight, m_err):
            return (jnp.sum((err * weight[..., None]) ** 2, axis=(1, 2))
                    + jnp.sum(m_err ** 2, axis=(1, 2)))

        def iterate(state, base_weight, mpos, mrot, mdt):
            bases, q, damping, stalls, done, weight, floor = state
            n_cand = bases.shape[0]
            tpos = jnp.broadcast_to(static_pos, (n_cand,) + static_pos.shape)
            trot = jnp.broadcast_to(static_rot, (n_cand,) + static_rot.shape)
            dt = None
            if has_moving:
                tpos = tpos.at[:, moving_idx].set(mpos)
                trot = trot.at[:, moving_idx].set(mrot)
                dt = jnp.zeros((n_cand, n_targets, 6, 3)).at[
                    :, moving_idx].set(mdt)
            err, bj, aj = linearize(bases, q, tpos, trot, dt)
            m_err, m_jac = margin_terms(q, floor)
            cost = cost_of(err, weight, m_err)
            pn = jnp.linalg.norm(err[..., :3], axis=-1)
            rn = jnp.linalg.norm(err[..., 3:], axis=-1)
            reached = (pn < thre) & (rn < rthre)
            done = done | (jnp.all(reached | (weight == 0), axis=1)
                           & ~jnp.any(m_err > 0, axis=(1, 2)))
            done = done | (damping >= 1e4) | (stalls >= 5)
            w = weight[..., None]
            d_base, d_arm = schur(clip_rows(err) * w, bj * w[..., None],
                                  aj * w[..., None], damping, base_weight,
                                  m_err, m_jac)
            new_bases = bases.at[:, :3].add(d_base)
            if lo is not None:
                new_bases = new_bases.at[:, :2].set(
                    jnp.clip(new_bases[:, :2], lo, hi))
            new_q = clip_q(q + d_arm)
            step = new_bases[:, :3] - bases[:, :3]
            if has_moving:
                # First-order move of the base-following targets.
                ntpos = tpos + jnp.einsum('knia,ka->kni', dt[..., :3, :],
                                          step)
                ntrot = _rodrigues(jnp, jnp.einsum(
                    'knia,ka->kni', dt[..., 3:, :], step)) @ trot
            else:
                ntpos, ntrot = tpos, trot
            _, mp, mr, _, _, _, _ = ee_world(new_bases, new_q)
            new_err = residual_from(mp, mr, ntpos, ntrot)
            new_m_err, _ = margin_terms(new_q, floor)
            new_cost = cost_of(new_err, weight, new_m_err)
            accept = (new_cost < cost) & ~done
            damping = jnp.where(accept, jnp.maximum(damping * 0.5, 1e-6),
                                jnp.minimum(damping * 4.0, 1e4))
            small = cost - new_cost < 1e-3 * jnp.maximum(cost, 1e-12)
            stalls = jnp.where(accept, jnp.where(small, stalls + 1, 0),
                               stalls)
            bases = jnp.where(accept[:, None], new_bases, bases)
            q = jnp.where(accept[:, None, None], new_q, q)
            return (bases, q, damping, stalls, done, weight, floor)

        self._iterate = jax.jit(iterate)
        self._has_moving = has_moving

    def _moving(self, bases):
        """Host-side poses ``(K, D, 3)``, ``(K, D, 3, 3)`` of the targets
        that follow the base, and their base derivative ``(K, D, 6, 3)``."""
        from skrobot.kinematics.differentiable import rotation_error_so3_log_batch_numpy
        problem = self.problem
        n_cand = bases.shape[0]
        n_moving = len(problem.target_fns)
        if not n_moving:
            return (np.zeros((n_cand, 0, 3)), np.zeros((n_cand, 0, 3, 3)),
                    np.zeros((n_cand, 0, 6, 3)))
        pos, rot = problem._moving_targets(bases)
        deriv = np.zeros((n_cand, n_moving, 6, 3))
        eps = 1e-5
        for axis in range(3):
            step = np.zeros(4)
            step[axis] = eps
            pos_p, rot_p = problem._moving_targets(bases + step)
            pos_m, rot_m = problem._moving_targets(bases - step)
            twist = rotation_error_so3_log_batch_numpy(
                rot_m.reshape(-1, 3, 3), rot_p.reshape(-1, 3, 3))
            twist = np.einsum('bij,bj->bi', rot_m.reshape(-1, 3, 3),
                              twist).reshape(pos_p.shape)
            deriv[..., :3, axis] = (pos_p - pos_m) / (2 * eps)
            deriv[..., 3:, axis] = twist / (2 * eps)
        return pos, rot, deriv

    def run(self, bases, q, active, stop, base_weight, margin_target):
        """Iterate in place on ``bases`` / ``q``; return the final error."""
        with self.jax.enable_x64(True):
            self._run(bases, q, active, stop, base_weight, margin_target)
        return self.problem.residual(bases, q)

    def _run(self, bases, q, active, stop, base_weight, margin_target):
        n_cand = bases.shape[0]
        floor = (np.zeros(n_cand) if margin_target is None
                 else np.asarray(margin_target, dtype=np.float64))
        weight = active.astype(np.float64)
        damping = np.full(n_cand, 1e-2)
        stalls = np.zeros(n_cand, dtype=np.int64)
        done = np.zeros(n_cand, dtype=bool)
        base_weight = self.jax.device_put(
            np.asarray(base_weight, dtype=np.float64))

        jax = self.jax
        rows = np.arange(n_cand)
        size = _bucket(n_cand)
        state = None
        moving = None
        for _ in range(stop):
            if state is None:
                padded = _pad(rows, size)
                # One transfer for the lot; jnp.asarray per array costs a
                # dispatch each.
                state = jax.device_put((
                    bases[padded], q[padded], damping[padded],
                    stalls[padded], done[padded] | (padded < 0),
                    weight[padded], floor[padded]))
                host_bases = bases[padded].copy()
                host_moving = self._moving(host_bases)
                moving = jax.device_put(host_moving)
            elif self._has_moving:
                # Only rows whose base moved need their targets again; a
                # rejected step leaves the base where it was.
                new_bases = np.asarray(state[0])
                changed = np.any(new_bases != host_bases, axis=1)
                if np.any(changed):
                    fresh = self._moving(new_bases[changed])
                    for held, new in zip(host_moving, fresh):
                        held[changed] = new
                    host_bases = new_bases.copy()
                    moving = jax.device_put(host_moving)
            state = self._iterate(state, base_weight, *moving)
            live = ~np.asarray(state[4])[:len(rows)]
            if not np.any(live):
                break
            if size > 16 and np.count_nonzero(live) <= size // 16:
                # Shrink the batch: write the rows back and repack the
                # ones still running.
                self._write_back(state, rows, bases, q, damping, stalls,
                                 done)
                rows = rows[live]
                size = _bucket(len(rows))
                state = None
        if state is not None:
            self._write_back(state, rows, bases, q, damping, stalls, done)

    @staticmethod
    def _write_back(state, rows, bases, q, damping, stalls, done):
        n = len(rows)
        bases[rows] = np.asarray(state[0])[:n]
        q[rows] = np.asarray(state[1])[:n]
        damping[rows] = np.asarray(state[2])[:n]
        stalls[rows] = np.asarray(state[3])[:n]
        done[rows] = np.asarray(state[4])[:n]


def _bucket(n):
    """Smallest power of sixteen, at least 16, holding ``n`` rows."""
    size = 16
    while size < n:
        size *= 16
    return size


def _pad(rows, size):
    """``rows`` padded to ``size`` with -1, which marks a dummy row.

    Index -1 picks the last row of the batch, a real row whose copy is
    flagged done and never written back.
    """
    return np.concatenate([rows, np.full(size - len(rows), -1)])

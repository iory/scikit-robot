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


def _enable_x64(jax):
    """Context manager turning on float64 for the code inside it.

    It is ``jax.enable_x64`` in current JAX and only
    ``jax.experimental.enable_x64`` up to 0.7.
    """
    if hasattr(jax, 'enable_x64'):
        return jax.enable_x64(True)
    from jax.experimental import enable_x64
    return enable_x64(True)


class JaxLevenbergMarquardt(object):
    """Jitted LM iterations for one :class:`_SharedBaseProblem`."""

    def __init__(self, problem, thre, rthre, max_pos_step, max_rot_step,
                 bounds):
        import jax
        import jax.numpy as jnp

        from skrobot.kinematics.base_placement import _clip_joints
        from skrobot.kinematics.base_placement import _clip_rows
        from skrobot.kinematics.base_placement import _cost
        from skrobot.kinematics.base_placement import _margin_terms
        from skrobot.kinematics.base_placement import _place_on_base
        from skrobot.kinematics.base_placement import _pose_residual
        from skrobot.kinematics.base_placement import _residual_jacobians
        from skrobot.kinematics.base_placement import _schur_step
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
        # Kept in NumPy: only its pattern is read, at trace time.
        base_frame = problem.base_frame
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

        def chain(q):
            n_cand = q.shape[0]
            angles = jnp.zeros((n_cand * n_targets, n_joints), q.dtype)
            angles = angles.at[:, non_mimic].set(q.reshape(-1, n_opt))
            jac, pos, rot = jac_one(angles)
            return (jac.reshape(n_cand, n_targets, 6, n_opt),
                    pos.reshape(n_cand, n_targets, 3),
                    rot.reshape(n_cand, n_targets, 3, 3))

        def residual(meas_pos, meas_rot, target_pos, target_rot):
            return _pose_residual(meas_pos, meas_rot, target_pos, target_rot, row_mask,
                                  tolerance, jnp)

        def linearize(bases, q, target_pos, target_rot, dt):
            jac, pos, rot = chain(q)
            (meas_pos, meas_rot, world_pos, world_rot, base_rot,
             base_pos) = _place_on_base(pos, rot, bases, base_frame, jnp)
            err = residual(meas_pos, meas_rot, target_pos, target_rot)
            base_jac, arm_jac = _residual_jacobians(
                err, jac, world_pos, world_rot, base_rot, base_pos,
                base_frame, row_mask, tolerance, dt, jnp)
            return err, base_jac, arm_jac

        def margin_terms(q, floor):
            return _margin_terms(q, floor, lower, upper, span, bounded,
                                 margin_weight, jnp)

        def iterate(state, base_weight, mpos, mrot, mdt):
            bases, q, damping, stalls, done, weight, floor = state
            n_cand = bases.shape[0]
            target_pos = jnp.broadcast_to(static_pos, (n_cand,) + static_pos.shape)
            target_rot = jnp.broadcast_to(static_rot, (n_cand,) + static_rot.shape)
            dt = None
            if has_moving:
                target_pos = target_pos.at[:, moving_idx].set(mpos)
                target_rot = target_rot.at[:, moving_idx].set(mrot)
                dt = jnp.zeros((n_cand, n_targets, 6, 3)).at[
                    :, moving_idx].set(mdt)
            err, bj, aj = linearize(bases, q, target_pos, target_rot, dt)
            m_err, m_jac = margin_terms(q, floor)
            cost = _cost(err * weight[..., None], m_err, jnp)
            pos_norm = jnp.linalg.norm(err[..., :3], axis=-1)
            rot_norm = jnp.linalg.norm(err[..., 3:], axis=-1)
            reached = (pos_norm < thre) & (rot_norm < rthre)
            done = done | (jnp.all(reached | (weight == 0), axis=1)
                           & ~jnp.any(m_err > 0, axis=(1, 2)))
            done = done | (damping >= 1e4) | (stalls >= 5)
            w = weight[..., None]
            d_base, d_arm = _schur_step(
                _clip_rows(err, max_pos_step, max_rot_step, jnp) * w,
                bj * w[..., None], aj * w[..., None], damping, base_weight,
                m_err, m_jac, jnp)
            new_bases = bases.at[:, :3].add(d_base)
            if lo is not None:
                new_bases = new_bases.at[:, :2].set(
                    jnp.clip(new_bases[:, :2], lo, hi))
            new_q = _clip_joints(q + d_arm, lower, upper, unbounded, jnp)
            step = new_bases[:, :3] - bases[:, :3]
            if has_moving:
                # First-order move of the base-following targets.
                new_target_pos = target_pos + jnp.einsum(
                    'knia,ka->kni', dt[..., :3, :], step)
                new_target_rot = _rodrigues(jnp, jnp.einsum(
                    'knia,ka->kni', dt[..., 3:, :], step)) @ target_rot
            else:
                new_target_pos, new_target_rot = target_pos, target_rot
            _, pos, rot = chain(new_q)
            mp, mr, _, _, _, _ = _place_on_base(pos, rot, new_bases,
                                                base_frame, jnp)
            new_err = residual(mp, mr, new_target_pos, new_target_rot)
            new_m_err, _ = margin_terms(new_q, floor)
            new_cost = _cost(new_err * weight[..., None], new_m_err, jnp)
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
        problem = self.problem
        n_cand = bases.shape[0]
        if not problem.target_fns:
            return (np.zeros((n_cand, 0, 3)), np.zeros((n_cand, 0, 3, 3)),
                    np.zeros((n_cand, 0, 6, 3)))
        pos, rot = problem._moving_targets(bases)
        return pos, rot, problem.moving_target_jacobian(bases)

    def run(self, bases, q, active, stop, base_weight, margin_target):
        """Iterate in place on ``bases`` / ``q``; return the final error."""
        with _enable_x64(self.jax):
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

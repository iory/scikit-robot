"""Find one mobile-base pose from which a robot reaches several targets.

A mobile manipulator often has to reach a set of end-effector poses one
after another without driving in between -- pick from a shelf, place on a
table, press a button. Batch IK with ``use_base`` answers each target with
its own base pose; this module answers all of them with a single, shared
one.

The problem has one planar base ``(x, y, yaw)`` and one arm configuration
per target. Each target's residual depends only on the base and its own
arm configuration, so the Gauss-Newton system is arrow shaped and the base
step is recovered through a Schur complement, at a cost linear in the
number of targets.
"""

import warnings

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import matrix2ypr
from skrobot.coordinates.math import normalize_angle
from skrobot.coordinates.math import normalize_mask
from skrobot.coordinates.math import rotation_matrix
from skrobot.kinematics.differentiable import _apply_rotation_tolerance
from skrobot.kinematics.differentiable import _normalize_axis_tolerance
from skrobot.kinematics.differentiable import compute_geometric_jacobian_batched_numpy
from skrobot.kinematics.differentiable import extract_fk_parameters
from skrobot.kinematics.differentiable import forward_kinematics_ee_batched_numpy
from skrobot.kinematics.differentiable import rotation_error_so3_log_batch_numpy


class BasePlacementResult(object):
    """Result of :func:`find_base_placement`.

    Attributes
    ----------
    base_pose : skrobot.coordinates.Coordinates
        Pose of the robot's root link. Planar: at ``z = base_height``
        (0 by default, the convention of ``use_base='planar'``) and turned
        about ``z`` only, by the yaw from the world +x axis to the root
        link's +x axis -- the robot's forward axis when its URDF follows
        the ROS convention. ``base_pose.x_axis`` is where it points.
    angle_vectors : numpy.ndarray
        ``(n_targets, len(robot_model.joint_list))``: one full
        ``robot_model.angle_vector()`` per target, in the order of
        ``robot_model.joint_list`` -- not only the solved joints. Joints
        outside the solved chain (a gripper's fingers, for instance) keep
        the angles the model had when the solve started.
    success : numpy.ndarray
        ``(n_targets,)`` bool, whether each target was reached within the
        thresholds. A placement that misses targets is still returned;
        check this before using it.
    position_errors : numpy.ndarray
        ``(n_targets,)`` position error in meters, over the masked
        end-effector axes.
    rotation_errors : numpy.ndarray
        ``(n_targets,)`` rotation error in radians, over the masked
        end-effector axes.
    joint_margins : numpy.ndarray
        ``(n_targets,)`` how far each target's arm configuration stays
        from the joint limits: the smallest ``min(q - lower, upper - q) /
        (upper - lower)`` over the bounded joints, so 0.5 is mid-range
        and 0 is at a limit.
    score : float
        What ranked this placement among those that reached as many
        targets: ``score(result)`` if given, else
        :attr:`min_joint_margin`.
    candidate_index : int
        Which initial base pose the returned placement started from.
    base_valid : bool
        ``is_valid_base(base_pose)``, or True when none was given.
    target_coords : list of Coordinates
        The targets as solved for this base: a target given as a function
        is evaluated at ``base_pose``.
    at_bound : bool
        Whether the base ended on an edge of ``base_bounds``. The solve
        would have gone further; a wider region may hold a better base.
    bound_distance : float
        Distance in meters from the base to the nearest edge of
        ``base_bounds`` (inf without bounds). A base a few millimeters
        inside is as suspect as one on the edge: the edge may have held
        it during the solve and let go at the end.
    candidates : list of BasePlacementResult
        Every candidate's result, best first. Empty on the entries of this
        list themselves.
    """

    def __init__(self, base_pose, angle_vectors, success, position_errors,
                 rotation_errors, joint_margins, candidate_index,
                 base_valid=True, at_bound=False, bound_distance=np.inf,
                 target_coords=None):
        self.base_pose = base_pose
        self.bound_distance = bound_distance
        self.target_coords = target_coords
        self.base_valid = base_valid
        self.at_bound = at_bound
        self.angle_vectors = angle_vectors
        self.success = success
        self.position_errors = position_errors
        self.rotation_errors = rotation_errors
        self.joint_margins = joint_margins
        self.candidate_index = candidate_index
        self.score = None
        self.candidates = []

    @property
    def min_joint_margin(self):
        """float: Smallest joint-limit margin over all targets."""
        return float(np.min(self.joint_margins))

    @property
    def n_reached(self):
        """int: Number of targets reached."""
        return int(np.count_nonzero(self.success))

    @property
    def all_reached(self):
        """bool: Whether every target was reached."""
        return bool(np.all(self.success))

    def __repr__(self):
        return ('<BasePlacementResult reached {}/{} base_pos={} '
                'base_yaw={:.3f} min_joint_margin={:.3f}>'.format(
                    self.n_reached, len(self.success),
                    np.round(self.base_pose.worldpos(), 4),
                    _yaw_of(self.base_pose.worldrot()),
                    self.min_joint_margin))


def _yaw_of(rot):
    return float(matrix2ypr(rot)[0])


def _planar_coords(x, y, yaw, z=0.0):
    return Coordinates(pos=np.array([x, y, z]), rot=rotation_matrix(yaw, 'z'))


def _base_matrices(base):
    """``(K, 4, 4)`` root-link poses from ``(K, 4)`` ``(x, y, yaw, z)``."""
    mats = np.tile(np.eye(4), (base.shape[0], 1, 1))
    mats[:, :3, :3] = _base_rotations(base[:, 2])
    mats[:, :3, 3] = base[:, [0, 1, 3]]
    return mats


def vectorized_target(fn):
    """Mark a base-following target function as taking every base at once.

    A plain target function is called once per candidate base with a
    :class:`~skrobot.coordinates.Coordinates`, which is simple but costs a
    Python call and a ``Coordinates`` per candidate and solver iteration.
    A function marked with this decorator is called once per iteration
    with a ``(K, 4, 4)`` array of root-link poses, and must return the
    ``(K, 4, 4)`` world poses of its target.

    Examples
    --------
    >>> @vectorized_target
    ... def stand(base_matrices):
    ...     out = np.tile(stand_pose, (len(base_matrices), 1, 1))
    ...     out[:, 0, 3] = base_matrices[:, 0, 3] + 0.30
    ...     return out
    """
    fn.vectorized = True
    return fn


def _base_rotations(yaw, xp=np):
    """``(..., 3, 3)`` rotations about z for an array of yaw angles.

    ``rotation_matrix`` builds one; this builds a batch, with ``xp`` the
    array module (``numpy`` or ``jax.numpy``).
    """
    c, s = xp.cos(yaw), xp.sin(yaw)
    zeros = xp.zeros_like(yaw)
    ones = xp.ones_like(yaw)
    return xp.stack([
        xp.stack([c, -s, zeros], axis=-1),
        xp.stack([s, c, zeros], axis=-1),
        xp.stack([zeros, zeros, ones], axis=-1),
    ], axis=-2)


def _is_single_mask(mask):
    if mask is None or isinstance(mask, (bool, np.bool_, str)):
        return True
    arr = np.asarray(mask, dtype=object)
    # Three scalars are one mask, e.g. [1, 1, 0]; anything else is a
    # list with one mask per target.
    return arr.shape == (3,) and all(
        isinstance(m, (bool, np.bool_, int, float, np.number)) for m in arr)


def _resolve_masks(mask, n_targets):
    """``(n_targets, 3)`` mask from one mask or one per target."""
    if _is_single_mask(mask):
        return np.tile(normalize_mask(mask).astype(np.float64),
                       (n_targets, 1))
    if isinstance(mask, np.ndarray) and mask.ndim == 2:
        mask = list(mask)
    if len(mask) != n_targets:
        raise ValueError(
            'Got {} masks for {} targets'.format(len(mask), n_targets))
    return np.array([normalize_mask(m) for m in mask], dtype=np.float64)


def _resolve_tolerances(tolerance, n_targets):
    """``(n_targets, 3)`` per-axis tolerance, 0 where there is none."""
    rows = np.atleast_2d(np.asarray(tolerance, dtype=object))
    if rows.shape == (1, 3):
        rows = np.repeat(rows, n_targets, axis=0)
    if rows.shape != (n_targets, 3):
        raise ValueError('rotation_tolerance must be 3 values or one '
                         'triple per target, got shape {}'.format(rows.shape))
    tol = [_normalize_axis_tolerance(row) for row in rows]
    return np.array([np.zeros(3) if t is None else t for t in tol])


def _targets_to_pos_rot(target_coords):
    if isinstance(target_coords, Coordinates):
        target_coords = [target_coords]
    if len(target_coords) == 0:
        raise ValueError('target_coords is empty')
    pos = np.array([c.worldpos() for c in target_coords], dtype=np.float64)
    rot = np.array([c.worldrot() for c in target_coords], dtype=np.float64)
    return pos, rot


def _chain_fk_params(robot_model, link_list, move_target):
    """FK parameters of the arm chain, expressed in the root-link frame."""
    fk_params = extract_fk_parameters(robot_model, link_list, move_target)
    root_link = robot_model._find_fullbody_root_link()
    root_inv = root_link.worldcoords().inverse_transformation()
    chain_base = root_inv.transform(Coordinates(
        pos=fk_params['base_position'], rot=fk_params['base_rotation']))
    fk_params['base_position'] = chain_base.worldpos().copy()
    fk_params['base_rotation'] = chain_base.worldrot().copy()
    return fk_params


class _SharedBaseProblem(object):
    """Residuals and Jacobians of the shared-base problem, vectorized.

    Every array is batched over ``(n_candidates, n_targets)``: each
    candidate is an independent problem with its own base and arm
    configurations, solved in lockstep.
    """

    def __init__(self, fk_params, target_pos, target_rot, position_mask,
                 rotation_mask, margin_target=None,
                 margin_weight=0.1, rotation_tolerance=None, base_frame=None,
                 target_fns=None):
        self.fk_params = fk_params
        n_joints = fk_params['n_joints']
        mimic_parent = fk_params['mimic_parent_indices']
        self.non_mimic = np.array(
            [i for i in range(n_joints) if mimic_parent[i] < 0],
            dtype=np.int64)
        self.n_opt = len(self.non_mimic)
        self.lower = fk_params['joint_limits_lower'][self.non_mimic]
        self.upper = fk_params['joint_limits_upper'][self.non_mimic]
        self.unbounded = fk_params['joint_limits_unbounded'][self.non_mimic]
        self.target_pos = target_pos
        self.target_rot = target_rot
        # [(index, f)]: targets that follow the base, f(base_pose) giving
        # their world pose.
        self.target_fns = sorted((target_fns or {}).items())
        self.n_targets = len(target_pos)
        # (N, 6): which world axes each target constrains.
        self.row_mask = np.concatenate([
            np.broadcast_to(position_mask, (self.n_targets, 3)),
            np.broadcast_to(rotation_mask, (self.n_targets, 3)),
        ], axis=1)
        # (N,) True for targets given in the root-link frame: they move
        # with the base, so only the arm can close their error.
        self.base_frame = (np.zeros(self.n_targets, dtype=bool)
                           if base_frame is None else base_frame)
        self.rotation_tolerance = rotation_tolerance  # (N, 3) or None
        self.margin_target = margin_target
        self.margin_weight = margin_weight
        self.bounded = ~self.unbounded & (self.upper > self.lower)
        # Set by solve_shared_base_ik when the JAX loop is used.
        self.jax_lm = None
        self.span = np.where(self.bounded, self.upper - self.lower, 1.0)

    def _full_angles(self, q):
        """Expand optimized angles to the chain's full joint vector."""
        flat = q.reshape(-1, self.n_opt)
        angles = np.zeros((flat.shape[0], self.fk_params['n_joints']))
        angles[:, self.non_mimic] = flat
        return angles

    def margin_terms(self, q, margin_target):
        """Hinge residual keeping joints ``margin_target`` off the limits.

        ``margin_target`` is None, a scalar, or ``(K,)`` -- one per
        candidate; see :func:`_margin_terms`.
        """
        if margin_target is None:
            zeros = np.zeros_like(q)
            return zeros, zeros
        return _margin_terms(q, margin_target, self.lower, self.upper,
                             self.span, self.bounded, self.margin_weight)

    def clip(self, q):
        return _clip_joints(q, self.lower, self.upper, self.unbounded)

    def _ee_world(self, base, q, with_jacobian):
        n_cand = base.shape[0]
        angles = self._full_angles(q)
        if with_jacobian:
            jac, pos, rot = compute_geometric_jacobian_batched_numpy(
                angles, self.fk_params)
            jac = jac.reshape(n_cand, self.n_targets, 6, self.n_opt)
        else:
            pos, rot = forward_kinematics_ee_batched_numpy(
                angles, self.fk_params)
            jac = None
        pos = pos.reshape(n_cand, self.n_targets, 3)
        rot = rot.reshape(n_cand, self.n_targets, 3, 3)
        return (jac,) + _place_on_base(pos, rot, base, self.base_frame)

    def _moving_targets(self, base):
        """``(K, D, 3)`` and ``(K, D, 3, 3)`` poses of the base-following
        targets for each candidate base."""
        n_cand = base.shape[0]
        pos = np.empty((n_cand, len(self.target_fns), 3))
        rot = np.empty((n_cand, len(self.target_fns), 3, 3))
        matrices = None
        for d, (_, fn) in enumerate(self.target_fns):
            if getattr(fn, 'vectorized', False):
                if matrices is None:
                    matrices = _base_matrices(base)
                out = np.asarray(fn(matrices), dtype=np.float64)
                pos[:, d] = out[:, :3, 3]
                rot[:, d] = out[:, :3, :3]
            else:
                for k in range(n_cand):
                    c = fn(_planar_coords(*base[k]))
                    pos[k, d] = c.worldpos()
                    rot[k, d] = c.worldrot()
        return pos, rot

    def moving_target_jacobian(self, base):
        """``(K, D, 6, 3)`` world twist of each base-following target per
        unit base ``(x, y, yaw)``, by central differences of its function."""
        eps = 1e-5
        jac = np.empty((base.shape[0], len(self.target_fns), 6, 3))
        for axis in range(3):
            step = np.zeros(4)
            step[axis] = eps
            pos_p, rot_p = self._moving_targets(base + step)
            pos_m, rot_m = self._moving_targets(base - step)
            twist = rot_m @ rotation_error_so3_log_batch_numpy(
                rot_m, rot_p)[..., None]
            jac[..., :3, axis] = (pos_p - pos_m) / (2 * eps)
            jac[..., 3:, axis] = twist[..., 0] / (2 * eps)
        return jac

    def targets(self, base):
        """``(K, N, 3)`` and ``(K, N, 3, 3)`` target poses for each base."""
        n_cand = base.shape[0]
        pos = np.broadcast_to(self.target_pos, (n_cand, self.n_targets, 3))
        rot = np.broadcast_to(self.target_rot,
                              (n_cand, self.n_targets, 3, 3))
        if self.target_fns:
            idx = [i for i, _ in self.target_fns]
            pos, rot = pos.copy(), rot.copy()
            pos[:, idx], rot[:, idx] = self._moving_targets(base)
        return pos, rot

    def residual(self, base, q):
        """``(K, N, 6)`` masked end-effector-frame error."""
        _, meas_pos, meas_rot, _, _, _, _ = self._ee_world(base, q, False)
        return _pose_residual(meas_pos, meas_rot, *self.targets(base),
                              row_mask=self.row_mask,
                              tolerance=self.rotation_tolerance)

    def linearize(self, base, q):
        """Residual, base Jacobian ``(K, N, 6, 3)``, arm Jacobian."""
        (jac, meas_pos, meas_rot, world_pos, world_rot, base_rot,
         base_pos) = self._ee_world(base, q, True)
        err = _pose_residual(meas_pos, meas_rot, *self.targets(base),
                             row_mask=self.row_mask,
                             tolerance=self.rotation_tolerance)
        moving_jac = None
        if self.target_fns:
            moving_jac = np.zeros(err.shape[:2] + (6, 3))
            moving_jac[:, [i for i, _ in self.target_fns]] = \
                self.moving_target_jacobian(base)
        base_jac, arm_jac = _residual_jacobians(
            err, jac, world_pos, world_rot, base_rot, base_pos,
            self.base_frame, self.row_mask, self.rotation_tolerance,
            moving_jac)
        return err, base_jac, arm_jac


def _margin_terms(q, margin_target, lower, upper, span, bounded, weight,
                  xp=np):
    """Hinge residual keeping joints ``margin_target`` off their limits.

    ``margin_target`` is a scalar or ``(K,)``, one per candidate. Returns
    ``(err, jac)``, both ``(K, N, n_opt)``: joint ``j`` of target ``n``
    contributes the row ``jac[..., j] * dq_j = err[..., j]`` while its
    normalized margin is under the target, and zero once it is clear.
    Each row touches one joint of one target, so it only adds to the
    diagonal of that target's arm block.
    """
    target = xp.asarray(margin_target, dtype=xp.float64)
    if target.ndim == 1:
        target = target[:, None, None]
    to_lower = (q - lower) / span
    to_upper = (upper - q) / span
    margin = xp.minimum(to_lower, to_upper)
    slope = xp.where(to_lower < to_upper, 1.0, -1.0) / span
    active = bounded & (margin < target)
    err = xp.where(active, weight * (target - margin), 0.0)
    jac = xp.where(active, weight * slope, 0.0)
    return err, jac


def _clip_joints(q, lower, upper, unbounded, xp=np):
    return xp.where(unbounded, q, xp.clip(q, lower, upper))


def _place_on_base(pos, rot, base, base_frame, xp=np):
    """End-effector poses from the chain's root-frame FK and the bases.

    Returns ``(meas_pos, meas_rot, world_pos, world_rot, base_rot,
    base_pos)``: the measured poses are the world ones except for targets
    given in the root-link frame, which are measured by the chain's own
    FK -- already relative to the root link wherever base_height puts it.
    """
    base_rot = _base_rotations(base[:, 2], xp)  # (K, 3, 3)
    base_pos = base[:, xp.asarray([0, 1, 3])]  # (K, 3)
    world_pos = (base_rot[:, None] @ pos[..., None])[..., 0] \
        + base_pos[:, None, :]
    world_rot = base_rot[:, None] @ rot
    meas_pos, meas_rot = world_pos, world_rot
    if np.any(base_frame):
        sel = base_frame[None, :, None]
        meas_pos = xp.where(sel, pos, world_pos)
        meas_rot = xp.where(sel[..., None], rot, world_rot)
    return meas_pos, meas_rot, world_pos, world_rot, base_rot, base_pos


def _pose_residual(meas_pos, meas_rot, target_pos, target_rot, row_mask,
                   tolerance=None, xp=np):
    """``(K, N, 6)`` masked error in the end-effector frame.

    The frame where the masks apply -- the same frame as
    difference_position / difference_rotation, so a mask means what it
    means to inverse_kinematics.
    """
    pos_err = (xp.swapaxes(meas_rot, -1, -2)
               @ (target_pos - meas_pos)[..., None])[..., 0]
    rot_err = rotation_error_so3_log_batch_numpy(meas_rot, target_rot, xp)
    if tolerance is not None:
        # As in batch_inverse_kinematics: a component within its
        # tolerance counts as reached and stops pulling.
        rot_err = _apply_rotation_tolerance(rot_err, tolerance, xp)
    return xp.concatenate([pos_err, rot_err], axis=-1) * row_mask


def _residual_jacobians(err, jac, world_pos, world_rot, base_rot, base_pos,
                        base_frame, row_mask, tolerance=None,
                        moving_jac=None, xp=np):
    """Base ``(K, N, 6, 3)`` and arm ``(K, N, 6, n_opt)`` Jacobians of
    :func:`_pose_residual`, from the chain's root-frame Jacobian ``jac``.

    ``moving_jac`` is ``(K, N, 6, 3)``, the world twist per unit base of
    targets that follow the base (zero for the others), or None.
    """
    # The chain Jacobian is in the root frame; rotate both halves.
    arm_jac = xp.concatenate([
        base_rot[:, None] @ jac[:, :, :3],
        base_rot[:, None] @ jac[:, :, 3:],
    ], axis=2)

    rel = world_pos - base_pos[:, None, :]
    one = xp.ones(rel.shape[:2])
    zero = xp.zeros(rel.shape[:2])
    base_jac = xp.stack([
        xp.stack([one, zero, -rel[..., 1]], axis=-1),
        xp.stack([zero, one, rel[..., 0]], axis=-1),
        xp.stack([zero, zero, zero], axis=-1),
        xp.stack([zero, zero, zero], axis=-1),
        xp.stack([zero, zero, zero], axis=-1),
        xp.stack([zero, zero, one], axis=-1),
    ], axis=-2)
    if np.any(base_frame):
        base_jac = xp.where(base_frame[None, :, None, None], 0.0, base_jac)
    if moving_jac is not None:
        # A target that follows the base moves the error both ways.
        base_jac = base_jac - moving_jac

    # World-frame twists into the end-effector frame of the residual; for
    # root-frame targets too, since R_world^T R_base J == R_root^T J.
    # Under a partial position mask this drops how turning the hand
    # turns the frame the error is measured in, as inverse_kinematics
    # does; the term vanishes as the masked error does.
    rot_t = xp.swapaxes(world_rot, -1, -2)

    def to_hand(jac6):
        return xp.concatenate([
            rot_t @ jac6[:, :, :3],
            rot_t @ jac6[:, :, 3:],
        ], axis=2)

    mask = row_mask[..., None]
    if tolerance is not None:
        # Rows inside the dead zone have no gradient.
        live = xp.concatenate(
            [xp.ones(err.shape[:-1] + (3,), dtype=bool),
             err[..., 3:] != 0.0], axis=-1)
        mask = mask * live[..., None]
    return to_hand(base_jac) * mask, to_hand(arm_jac) * mask


def _cost(err, margin_err=None, xp=np):
    cost = xp.sum(err ** 2, axis=(1, 2))
    if margin_err is not None:
        cost = cost + xp.sum(margin_err ** 2, axis=(1, 2))
    return cost


def _clip_rows(err, max_pos_step, max_rot_step, xp=np):
    """Cap each target's error so one far target cannot drag the base."""
    pos_norm = xp.linalg.norm(err[..., :3], axis=-1, keepdims=True)
    rot_norm = xp.linalg.norm(err[..., 3:], axis=-1, keepdims=True)
    pos_scale = xp.minimum(1.0, max_pos_step / xp.maximum(pos_norm, 1e-12))
    rot_scale = xp.minimum(1.0, max_rot_step / xp.maximum(rot_norm, 1e-12))
    return xp.concatenate(
        [err[..., :3] * pos_scale, err[..., 3:] * rot_scale], axis=-1)


def _schur_step(err, base_jac, arm_jac, damping, base_weight,
                margin_err=None, margin_jac=None, xp=np):
    """Solve the arrow-shaped damped normal equations.

    The unknowns are one base step shared by every target and one arm step
    per target. Eliminating the arm steps leaves a 3x3 system for the base.
    """
    n_opt = arm_jac.shape[-1]
    base_jac = base_jac * base_weight
    lam = damping[:, None, None, None]

    arm_t = xp.swapaxes(arm_jac, -1, -2)
    base_t = xp.swapaxes(base_jac, -1, -2)
    h_aa = arm_t @ arm_jac \
        + lam * xp.eye(n_opt)
    h_ba = base_t @ arm_jac  # (K, N, 3, n)
    g_a = (arm_t @ err[..., None])[..., 0]
    if margin_jac is not None:
        h_aa = h_aa + (margin_jac ** 2)[..., None] * xp.eye(n_opt)
        g_a = g_a + margin_jac * margin_err
    g_b = (base_t @ err[..., None])[..., 0].sum(axis=1)
    h_bb = (base_t @ base_jac).sum(axis=1) \
        + damping[:, None, None] * xp.eye(3)

    # H_aa^-1 [H_ab, g_a] in one solve per target.
    rhs = xp.concatenate(
        [xp.swapaxes(h_ba, -1, -2), g_a[..., None]], axis=-1)
    sol = xp.linalg.solve(h_aa, rhs)
    inv_h_ab = sol[..., :3]
    inv_g_a = sol[..., 3]

    schur = h_bb - (h_ba @ inv_h_ab).sum(axis=1)
    reduced = g_b - (h_ba @ inv_g_a[..., None])[..., 0].sum(axis=1)
    d_base = xp.linalg.solve(schur, reduced[..., None])[..., 0]
    d_arm = inv_g_a - (inv_h_ab @ d_base[:, None, :, None])[..., 0]
    return d_base * base_weight, d_arm


def _errors(err):
    return (np.linalg.norm(err[..., :3], axis=-1),
            np.linalg.norm(err[..., 3:], axis=-1))


def _levenberg_marquardt(problem, bases, q, active, stop, thre, rthre,
                         base_weight, bounds, max_pos_step, max_rot_step,
                         margin_target=None):
    """Iterate on the active targets of every candidate, in place.

    A candidate stops once its active targets are reached and its joints
    are clear of the margin, or once it stalls: steps keep failing until
    the damping saturates, or five accepted steps in a row each lower the
    cost by less than 0.1%. Stopped candidates drop out of the batch, so
    a few slow ones do not cost the price of all of them.
    ``margin_target`` is None or ``(K,)``, one per candidate.
    """
    if problem.jax_lm is not None:
        return problem.jax_lm.run(bases, q, active, stop, base_weight,
                                  margin_target)
    n_cand = bases.shape[0]

    def margin(qs, rows):
        return problem.margin_terms(
            qs, None if margin_target is None else margin_target[rows])

    weight = active[..., None].astype(np.float64)
    damping = np.full(n_cand, 1e-2)
    stalls = np.zeros(n_cand, dtype=np.int64)
    done = np.zeros(n_cand, dtype=bool)
    err, base_jac, arm_jac = problem.linearize(bases, q)
    m_err, m_jac = margin(q, slice(None))
    cost = _cost(err * weight, m_err)
    for _ in range(stop):
        pos_norm, rot_norm = _errors(err)
        reached = (pos_norm < thre) & (rot_norm < rthre)
        done |= np.all(reached | ~active, axis=1) \
            & ~np.any(m_err > 0, axis=(1, 2))
        done |= (damping >= 1e4) | (stalls >= 5)
        run = np.where(~done)[0]
        if len(run) == 0:
            break
        w = weight[run]
        d_base, d_arm = _schur_step(
            _clip_rows(err[run], max_pos_step, max_rot_step) * w,
            base_jac[run] * w[..., None], arm_jac[run] * w[..., None],
            damping[run], base_weight, m_err[run], m_jac[run])
        new_bases = bases[run].copy()
        new_bases[:, :3] += d_base
        if bounds is not None:
            new_bases[:, :2] = np.clip(new_bases[:, :2], bounds[0], bounds[1])
        new_q = problem.clip(q[run] + d_arm)
        new_m_err, _ = margin(new_q, run)
        new_cost = _cost(problem.residual(new_bases, new_q) * w, new_m_err)
        accept = new_cost < cost[run]
        damping[run] = np.where(accept, np.maximum(damping[run] * 0.5, 1e-6),
                                np.minimum(damping[run] * 4.0, 1e4))
        small = cost[run] - new_cost < 1e-3 * np.maximum(cost[run], 1e-12)
        stalls[run] = np.where(
            accept, np.where(small, stalls[run] + 1, 0), stalls[run])
        moved = run[accept]
        if len(moved):
            bases[moved] = new_bases[accept]
            q[moved] = new_q[accept]
            e, bj, aj = problem.linearize(bases[moved], q[moved])
            me, mj = margin(q[moved], moved)
            err[moved], base_jac[moved], arm_jac[moved] = e, bj, aj
            m_err[moved], m_jac[moved] = me, mj
            cost[moved] = _cost(e * weight[moved], me)
    return err


def _climb_margin(problem, bases, q, err, stop, thre, rthre, base_weight,
                  bounds, max_pos_step, max_rot_step, floor, step,
                  max_steps=None, verbose=False):
    """Raise each candidate's smallest joint margin while it stays valid.

    At most ``max_steps`` rounds when given. Works in place on ``bases``
    and ``q``; returns the matching error.
    """
    pos_err, rot_err = _errors(err)
    reached = (pos_err < thre) & (rot_err < rthre)
    current = np.min(_joint_margins(problem.fk_params, q), axis=1)
    # Placements rank by targets reached before margin, so only the starts
    # that reach the most targets can gain from a climb.
    n_reached = np.count_nonzero(reached, axis=1)
    running = (n_reached > 0) & (n_reached == np.max(n_reached))
    n_round = 0
    while np.any(running):
        if max_steps is not None and n_round >= max_steps:
            break
        n_round += 1
        rows = np.where(running)[0]
        trial_bases = bases[rows].copy()
        trial_q = q[rows].copy()
        target = np.maximum(current[rows] + step, floor)
        trial_err = _levenberg_marquardt(
            problem, trial_bases, trial_q, reached[rows], stop, thre, rthre,
            base_weight, bounds, max_pos_step, max_rot_step, target)
        pe, re = _errors(trial_err)
        kept = np.all(((pe < thre) & (re < rthre)) | ~reached[rows], axis=1)
        margin = np.min(_joint_margins(problem.fk_params, trial_q), axis=1)
        better = kept & (margin > current[rows] + 1e-4)
        won = rows[better]
        bases[won] = trial_bases[better]
        q[won] = trial_q[better]
        err[won] = trial_err[better]
        current[won] = margin[better]
        running[rows[~better]] = False
        if verbose:
            print('  margin climb round {}: {} of {} candidates still '
                  'climbing, best smallest margin {:.3f}'.format(
                      n_round, int(np.count_nonzero(running)), len(running),
                      float(np.max(current))))
    return err


def solve_shared_base_ik(
        fk_params, target_pos, target_rot, initial_bases, initial_q,
        position_mask=True, rotation_mask=True, stop=200, thre=1e-3,
        rthre=np.deg2rad(1.0), base_weight=1.0, base_bounds=None,
        drop_unreachable=True, base_height=0.0, margin_target=None,
        margin_weight=0.1, rotation_tolerance=None, base_frame=None,
        target_fns=None, maximize_margin=False, margin_step=0.01,
        max_margin_steps=None, verbose=False, backend='numpy',
        max_pos_step=0.1, max_rot_step=0.5):
    """Solve shared-base IK from several initial bases at once.

    Levenberg-Marquardt on the arrow-shaped system: each candidate keeps
    its own damping, which shrinks after a step that lowers the cost and
    grows after one that does not (the step is then discarded).

    Least squares lets a target that cannot be reached pull the base away
    from the ones that can. With ``drop_unreachable``, a candidate that
    stalls with targets unreached stops weighing the worst quarter of
    those it misses (at least one) and solves again, until the targets
    it still weighs are all reached. A
    last pass then holds the base and moves each dropped target's arm as
    close as it gets, so the reported errors are those of the final base.

    With ``maximize_margin``, each start that reaches the most targets
    then raises its margin target a step above the smallest margin it
    has, solves again -- base included -- and keeps the result if every
    target it reached is still reached and the smallest margin grew; it
    stops at the first step that fails. Starts that reach fewer targets
    rank below them whatever their margin, and are not climbed. The
    climb reaches towards the largest smallest margin the targets allow
    near each start, which the hinge alone does not: once every joint
    clears ``margin_target`` it stops pushing.

    Parameters
    ----------
    fk_params : dict
        Chain FK parameters in the root-link frame.
    target_pos : numpy.ndarray
        ``(N, 3)`` target positions in the world frame.
    target_rot : numpy.ndarray
        ``(N, 3, 3)`` target rotations in the world frame.
    initial_bases : numpy.ndarray
        ``(K, 3)`` initial ``(x, y, yaw)``.
    initial_q : numpy.ndarray
        ``(K, N, n_opt)`` or ``(n_opt,)`` initial arm angles.
    position_mask, rotation_mask : bool, str, sequence or list, optional
        Which end-effector axes to constrain, as in
        :meth:`~skrobot.model.RobotModel.inverse_kinematics`. ``True``
        (default) constrains all. A list with one mask per target sets
        them target by target.
    stop : int
        Maximum number of iterations per solve.
    thre : float
        Position threshold in meters for a target to count as reached.
    rthre : float
        Rotation threshold in radians.
    base_weight : float or sequence of 3
        Scale of the base step against the arm's; smaller keeps the base
        nearer its initial pose.
    base_bounds : tuple, optional
        ``((x_min, x_max), (y_min, y_max))`` box the base is kept in.
    drop_unreachable : bool
        Drop stalled targets one at a time, as described above.
    base_height : float or numpy.ndarray
        Fixed height of the root link above the world origin, meters --
        one for all starts or ``(K,)``, one per start.
    margin_target : float, optional
        Normalized joint-limit margin (0 to 0.5) to keep; joints closer
        to a limit are pushed back by a hinge term of weight
        ``margin_weight``. None (default) leaves the margin alone.
    margin_weight : float
        Weight of that term against errors in meters and radians.
    rotation_tolerance : sequence of 3, or list of them, optional
        Per-axis rotation error, in the end-effector frame, that counts
        as reached -- one triple for all targets or one per target.
    base_frame : numpy.ndarray, optional
        ``(N,)`` bool, True where ``target_pos`` / ``target_rot`` are in
        the root-link frame rather than the world frame.
    target_fns : dict, optional
        ``{index: f}`` for targets that follow the base: ``f(base_pose)``
        returns the target's world pose for a root-link pose, and replaces
        ``target_pos[index]`` / ``target_rot[index]``.
    maximize_margin : bool
        Climb the smallest joint-limit margin, as described above.
    margin_step : float
        How far each climbing step raises the margin target.
    max_margin_steps : int, optional
        Cap on the climbing rounds; None climbs until no candidate gains.
    verbose : bool
        Print the progress of the climb.
    backend : {'numpy', 'jax'}
        Run the Levenberg-Marquardt iterations in NumPy, or jitted in JAX
        (needs ``jax``). JAX pays a few seconds of compilation per solve
        and computes every start of a batch each iteration, so it only
        pays off on a GPU: on a 74-target, 200-start problem it took 28 s
        on a GPU against 38 s for NumPy, and 281 s on a CPU. Both run in
        float64 and take the same steps up to rounding.
    max_pos_step, max_rot_step : float
        Per-target cap on the error fed to one step, in meters and
        radians.

    Returns
    -------
    bases : numpy.ndarray
        ``(K, 3)`` solved ``(x, y, yaw)``.
    q : numpy.ndarray
        ``(K, N, n_opt)`` solved arm angles.
    pos_err : numpy.ndarray
        ``(K, N)`` position error norms.
    rot_err : numpy.ndarray
        ``(K, N)`` rotation error norms.
    """
    n_targets = len(target_pos)
    problem = _SharedBaseProblem(
        fk_params, target_pos, target_rot,
        _resolve_masks(position_mask, n_targets),
        _resolve_masks(rotation_mask, n_targets),
        margin_target=margin_target, margin_weight=margin_weight,
        rotation_tolerance=(
            None if rotation_tolerance is None
            else _resolve_tolerances(rotation_tolerance, n_targets)),
        base_frame=(None if base_frame is None
                    else np.asarray(base_frame, dtype=bool)),
        target_fns=target_fns)
    bases = np.array(initial_bases, dtype=np.float64).reshape(-1, 3)
    n_cand = bases.shape[0]
    # The height rides along as a fourth column the solver never steps,
    # so starts at different heights share one batch.
    bases = np.concatenate([bases, np.broadcast_to(
        np.asarray(base_height, dtype=np.float64), (n_cand,))[:, None]],
        axis=1)
    q = np.broadcast_to(
        np.asarray(initial_q, dtype=np.float64),
        (n_cand, problem.n_targets, problem.n_opt)).copy()
    q = problem.clip(q)
    base_weight = np.broadcast_to(
        np.asarray(base_weight, dtype=np.float64), (3,))
    bounds = None
    if base_bounds is not None:
        bounds = (np.array([base_bounds[0][0], base_bounds[1][0]]),
                  np.array([base_bounds[0][1], base_bounds[1][1]]))
        bases[:, :2] = np.clip(bases[:, :2], bounds[0], bounds[1])
    if backend == 'jax':
        from skrobot.kinematics._base_placement_jax import JaxLevenbergMarquardt
        problem.jax_lm = JaxLevenbergMarquardt(
            problem, thre, rthre, max_pos_step, max_rot_step, bounds)
    elif backend != 'numpy':
        raise ValueError(
            "backend must be 'numpy' or 'jax', got {!r}".format(backend))

    floor = None if margin_target is None else np.full(n_cand, margin_target)

    def solve_rows(rows, active_rows, weight):
        """Run LM on ``rows`` only, writing back into the batch."""
        sub_bases, sub_q = bases[rows].copy(), q[rows].copy()
        sub_err = _levenberg_marquardt(
            problem, sub_bases, sub_q, active_rows, stop, thre, rthre,
            weight, bounds, max_pos_step, max_rot_step,
            None if floor is None else floor[rows])
        bases[rows], q[rows] = sub_bases, sub_q
        return sub_err

    active = np.ones((n_cand, problem.n_targets), dtype=bool)
    rounds = problem.n_targets if drop_unreachable else 1
    rows = np.arange(n_cand)
    err = None
    for _ in range(rounds):
        # Only the starts that dropped targets last round have anything
        # new to solve; the rest keep their state.
        sub_err = solve_rows(rows, active[rows], base_weight)
        if err is None:
            err = sub_err
        else:
            err[rows] = sub_err
        if not drop_unreachable:
            break
        pos_err, rot_err = _errors(err)
        failing = active & ~((pos_err < thre) & (rot_err < rthre))
        stalled = np.any(failing, axis=1)
        if not np.any(stalled):
            break
        # The worst quarter of the failing targets, at least one, so many
        # targets take a few rounds rather than one round each.
        score = pos_err / thre + rot_err / rthre
        for row in np.where(stalled)[0]:
            fail = np.where(failing[row])[0]
            n_drop = max(1, len(fail) // 4)
            worst = fail[np.argsort(-score[row, fail])[:n_drop]]
            active[row, worst] = False
        rows = np.where(stalled)[0]

    dropped = np.where(~np.all(active, axis=1))[0]
    if len(dropped):
        # Base held: only the dropped targets' arms still have work to do.
        err[dropped] = solve_rows(
            dropped, np.ones((len(dropped), problem.n_targets), dtype=bool),
            np.zeros(3))

    if maximize_margin:
        err = _climb_margin(
            problem, bases, q, err, stop, thre, rthre, base_weight, bounds,
            max_pos_step, max_rot_step,
            0.0 if margin_target is None else margin_target, margin_step,
            max_margin_steps, verbose)
    bases[:, 2] = normalize_angle(bases[:, 2])
    pos_err, rot_err = _errors(err)
    return bases[:, :3], q, pos_err, rot_err


def _resolve_link_list(robot_model, move_target, link_list, joint_list):
    if link_list is not None and joint_list is not None:
        raise ValueError('Pass either link_list or joint_list, not both')
    if joint_list is not None:
        link_list = [j.child_link for j in joint_list]
    elif link_list is None:
        link_list = robot_model.link_lists(move_target.parent)
    # Fixed joints carry no DoF; extract_fk_parameters folds the links
    # between movable joints into the static transforms.
    link_list = [link for link in link_list
                 if link.joint is not None
                 and link.joint.joint_type != 'fixed']
    if not link_list:
        raise ValueError('The chain to move_target has no movable joint')
    return link_list


def set_base_pose(robot_model, base_pose):
    """Move ``robot_model`` so its root link sits at ``base_pose``.

    The root link is often a child of the model's own coordinates, so the
    model is moved rather than the link.
    """
    root_link = robot_model._find_fullbody_root_link()
    if root_link is robot_model:
        robot_model.newcoords(base_pose)
        return
    # Composed from local poses only: going through the model's world pose
    # would feed its rounding error back in, and it grows with every call.
    model_to_root = Coordinates()
    node = root_link
    while node is not robot_model:
        if node is None:
            raise ValueError(
                'The root link is not attached under the robot model')
        model_to_root = Coordinates(
            pos=node.translation, rot=node.rotation).transform(model_to_root)
        node = node.parent
    robot_model.newcoords(
        base_pose.copy_worldcoords().transform(
            model_to_root.inverse_transformation()))


def _joint_margins(fk_params, q):
    """``(K, N)`` smallest normalized distance to a joint limit.

    Joints without limits (continuous) cannot run into one and are left
    out; a chain with none bounded has margin 0.5 everywhere.
    """
    mimic_parent = fk_params['mimic_parent_indices']
    non_mimic = np.array([i for i in range(fk_params['n_joints'])
                          if mimic_parent[i] < 0], dtype=np.int64)
    lower = fk_params['joint_limits_lower'][non_mimic]
    upper = fk_params['joint_limits_upper'][non_mimic]
    bounded = ~fk_params['joint_limits_unbounded'][non_mimic] \
        & (upper > lower)
    if not np.any(bounded):
        return np.full(q.shape[:2], 0.5)
    span = upper[bounded] - lower[bounded]
    qb = q[..., bounded]
    margin = np.minimum(qb - lower[bounded], upper[bounded] - qb) / span
    return np.min(margin, axis=-1)


def _base_to_array(coords):
    pos = coords.worldpos()
    return np.array([pos[0], pos[1], _yaw_of(coords.worldrot())])


def _evaluate_target(fn, pose):
    """A base-following target's world pose at one root-link ``pose``."""
    if getattr(fn, 'vectorized', False):
        mat = np.asarray(fn(pose.T()[None]), dtype=np.float64)[0]
        return Coordinates(pos=mat[:3, 3], rot=mat[:3, :3])
    return fn(pose)


def _rank(results):
    """Best of ``results``, with all of them, ranked, as its candidates."""
    order = np.lexsort((
        [np.sum(r.position_errors) + np.sum(r.rotation_errors)
         for r in results],
        [-r.score for r in results],
        [-r.n_reached for r in results],
        [not r.base_valid for r in results]))
    candidates = [results[k] for k in order]
    best = candidates[0]
    best.candidates = candidates
    return best


def find_base_placement(
        robot_model, target_coords, move_target=None, link_list=None,
        joint_list=None, initial_base_poses=None, initial_angles='current',
        position_mask=True, rotation_mask=True, stop=200, thre=1e-3,
        rthre=np.deg2rad(1.0), base_weight=1.0, base_bounds=None,
        drop_unreachable=True, base_height=0.0, margin_target=None,
        margin_weight=0.1, rotation_tolerance=None, maximize_margin=False,
        margin_step=0.01, max_margin_steps=None, target_frame='world',
        score=None, irm=None, n_candidates=10, is_valid_base=None,
        apply=False, verbose=False, backend='numpy'):
    """Find one planar base pose from which every target is reachable.

    Solves for a single base pose ``(x, y, yaw)`` of the robot's root link
    together with one arm configuration per target, so that the end
    effector reaches every target from the same base. Each entry of
    ``initial_base_poses`` starts an independent solve; all of them run
    vectorized and the best is returned: most targets reached, then the
    highest ``score`` -- by default the joint-limit margin, so among
    placements that all work the one with the most room wins.

    The optimization is local: from a poor start it can settle where only
    some targets are reached. ``result.success`` says which; nothing is
    hidden. Passing an :class:`InverseReachabilityMap` as ``irm`` gives it
    global starts: the map proposes the base cells most targets can be
    reached from.

    Parameters
    ----------
    robot_model : skrobot.model.RobotModel
        Robot. Its root link is the mobile base.
    target_coords : Coordinates or list of Coordinates or callables
        End-effector targets, in the world frame unless ``target_frame``
        says otherwise. An entry can also be a function that takes a
        root-link pose and returns the target's world pose, for a target
        that depends on where the base stands -- a fixture that moves
        with the base along one axis only, say. Its share of the base
        Jacobian is taken by finite differences, so it should be smooth.
        Such a function is called once per candidate and iteration; mark
        it with :func:`vectorized_target` to take every base at once as
        ``(K, 4, 4)`` matrices, which is much faster with many targets or
        starts. ``irm`` leaves such targets out of the vote.
    move_target : CascadedCoords, optional
        End effector. Defaults to ``robot_model.end_coords``.
    link_list : list of Link, optional
        Links whose joints are actuated. Defaults to the chain from the
        root to ``move_target``.
    joint_list : list of Joint, optional
        Joints to actuate; alternative to ``link_list``.
    initial_base_poses : Coordinates or list of Coordinates, optional
        Starting base poses, solved in addition to those ``irm``
        proposes; only their ``(x, y, yaw)`` is used. Without either,
        the root link's current pose.
    initial_angles : 'current' or numpy.ndarray
        Arm angles to start every target from: ``'current'`` uses the
        model's angles, an array is ``(n_opt,)`` over the chain's
        non-mimic joints, and ``(n_seeds, n_opt)`` solves every base
        start once from each seed. The result depends on these: the
        solve is local in the arm as much as in the base, and on one
        74-target problem the smallest margin found ran from 0.18 to
        0.22 over four seeds. Several distinct seeds cost a solve each
        and make the outcome less of a draw; three to five spread over
        the arm's working poses -- the reset pose, the arm reaching
        forward low and high -- are a good default. ``'current'`` reads the
        model at call time, so the same call can answer differently
        after the model has been moved.
    position_mask, rotation_mask : bool, str, sequence or list
        End-effector axes to constrain, with the meaning they have in
        :meth:`~skrobot.model.RobotModel.inverse_kinematics`: the error
        is taken in the end-effector frame and the unmasked components
        are dropped. ``rotation_mask=False`` gives position-only targets,
        ``rotation_mask='xz'`` leaves the hand free to turn about its own
        y axis.
        A list with one entry per target, such as ``[True, 'z', False]``,
        sets them target by target; three scalars like ``[1, 1, 0]`` are
        one mask for every target.
    stop : int
        Maximum number of iterations.
    thre : float
        Position threshold in meters.
    rthre : float
        Rotation threshold in radians.
    base_weight : float or sequence of 3
        Relative step size of ``(x, y, yaw)`` against the arm joints.
    base_bounds : tuple, optional
        ``((x_min, x_max), (y_min, y_max))`` region the base must stay in.
    drop_unreachable : bool
        When a start stalls with targets unreached, stop weighing the
        worst quarter of them and solve again, so the base settles for the
        targets it can serve instead of a compromise that serves none.
        Dropped targets are still reported in ``result.success``.
    base_height : float or sequence of float
        Height of the root link, held fixed while ``(x, y, yaw)`` are
        solved. 0 puts it on the floor as ``use_base='planar'`` does; a
        root link above the floor, or a base raised or lowered against
        the targets, takes its actual height. A sequence -- the heights a
        pedestal can be set to, say -- solves the starts of every height
        together in one batch and returns the best over all of them, with
        ``result.candidates`` drawn from every height.
    margin_target : float, optional
        Keep every joint at least this normalized margin (0 to 0.5, see
        :attr:`BasePlacementResult.joint_margins`) from its limits: a
        hinge term of weight ``margin_weight`` pushes back the joints
        that come closer, and moves the base too when that is what makes
        room. It competes with the targets, so an unreachable margin can
        cost accuracy; ``result.success`` still judges the targets alone.
        None (default) solves for the targets only.
    margin_weight : float
        Weight of that term against errors in meters and radians.
    maximize_margin : bool
        After the targets are reached, keep raising the margin target a
        ``margin_step`` above the smallest margin each candidate has, base
        free to move, for as long as every reached target stays reached
        and the smallest margin grows. The hinge alone stops pushing once
        every joint clears ``margin_target``; this climbs to the largest
        smallest margin the targets allow near each candidate, at the
        cost of more solves.
    margin_step : float
        How far each climbing step raises the margin target.
    max_margin_steps : int, optional
        Cap on the climbing rounds, which take most of the time when
        ``maximize_margin`` is on; None climbs until no candidate gains.
    target_frame : 'world', 'base', or list of them
        Frame of each target: ``'world'``, or ``'base'`` for a pose
        relative to the root link -- a fixture carried on the base, which
        moves with it and is reached by the arm alone. One value for all
        targets or one per target. ``irm`` ignores base-frame targets
        when proposing bases, since every base serves them alike.
    rotation_tolerance : sequence of 3, or list of them, optional
        Per-axis rotation error in the end-effector frame that counts as
        reached, as in :meth:`~skrobot.model.RobotModel.batch_inverse_kinematics`
        -- one triple for all targets or one per target. Where a mask
        frees an axis completely, a tolerance frees it within a range:
        ``rotation_tolerance=(0, np.deg2rad(60), 0)`` lets the hand turn
        up to 60 degrees about its y axis.
    score : callable, optional
        ``f(BasePlacementResult) -> float``, higher is better, ranking
        placements that reach the same number of targets. Defaults to
        :attr:`BasePlacementResult.min_joint_margin`.
    irm : InverseReachabilityMap, optional
        Map built for the same chain and end effector; its candidate
        bases and seeds start the solve.
    n_candidates : int
        How many bases to take from ``irm``. Each is solved once per seed
        in ``initial_angles``, with every arm starting there, and once
        with each arm starting from the map sample that voted for that
        base. Which start does better depends on the robot and the
        targets.
    is_valid_base : callable, optional
        ``f(Coordinates) -> bool`` for where the base may stand. ``irm``
        proposes only valid cells, and among the solved placements the
        valid ones rank first; ``result.base_valid`` tells whether the
        returned one is.
    apply : bool
        If True, move the robot to the found base pose and the first
        target's angle vector.
    verbose : bool
        Print progress: each height, the number of starts, the climb.
    backend : {'numpy', 'jax'}
        Where the solver iterates. ``'jax'`` jits each iteration and is
        worth it on a GPU only; on a CPU NumPy is faster. See
        :func:`solve_shared_base_ik`.

    Returns
    -------
    BasePlacementResult
        Best placement; ``result.candidates`` holds every start's result,
        ranked: valid base, most targets reached, highest ``score``,
        smallest error.

    Examples
    --------
    >>> from skrobot.coordinates import Coordinates
    >>> from skrobot.kinematics import find_base_placement
    >>> from skrobot.models import Fetch
    >>> robot = Fetch()
    >>> robot.reset_pose()
    >>> targets = [Coordinates(pos=[1.5, 0.3, 0.8]),
    ...            Coordinates(pos=[1.6, -0.2, 1.0])]
    >>> result = find_base_placement(robot, targets, rotation_mask=False)
    >>> result.all_reached
    True
    """
    heights = np.atleast_1d(np.asarray(base_height, dtype=np.float64))
    if move_target is None:
        move_target = robot_model.end_coords
    if isinstance(move_target, (list, tuple)):
        raise ValueError(
            'find_base_placement solves for one end effector; pass '
            'move_target explicitly')
    link_list = _resolve_link_list(
        robot_model, move_target, link_list, joint_list)
    fk_params = _chain_fk_params(robot_model, link_list, move_target)
    if isinstance(target_coords, Coordinates) or callable(target_coords):
        target_coords = [target_coords]
    target_coords = list(target_coords)
    target_fns = {i: t for i, t in enumerate(target_coords) if callable(t)}
    target_pos, target_rot = _targets_to_pos_rot(
        [Coordinates() if callable(t) else t for t in target_coords])
    frames = ([target_frame] * len(target_pos)
              if isinstance(target_frame, str) else list(target_frame))
    if len(frames) != len(target_pos) or \
            any(f not in ('world', 'base') for f in frames):
        raise ValueError(
            "target_frame must be 'world', 'base', or one of them per "
            'target, got {!r}'.format(target_frame))
    base_frame = np.array([f == 'base' for f in frames])
    if any(base_frame[i] for i in target_fns):
        raise ValueError(
            "A target given as a function returns a world pose; it cannot "
            "also have target_frame='base'")

    mimic_parent = fk_params['mimic_parent_indices']
    non_mimic = [i for i in range(len(link_list)) if mimic_parent[i] < 0]
    if isinstance(initial_angles, str):
        if initial_angles != 'current':
            raise ValueError(
                "initial_angles must be 'current' or an array, got {!r}"
                .format(initial_angles))
        arm_seeds = np.array(
            [[link_list[i].joint.joint_angle() for i in non_mimic]])
    else:
        arm_seeds = np.atleast_2d(
            np.asarray(initial_angles, dtype=np.float64))
        if arm_seeds.ndim != 2 or arm_seeds.shape[1] != len(non_mimic):
            raise ValueError(
                'initial_angles must be (n_opt,) or (n_seeds, n_opt) with '
                'n_opt={}, got shape {}'.format(
                    len(non_mimic), np.shape(initial_angles)))
    initial_q = arm_seeds[0]
    n_targets = len(target_pos)

    def from_seeds(bases):
        """Every base once per arm seed, with every arm at that seed."""
        out_bases, out_q = [], []
        for seed in arm_seeds:
            out_bases.extend(bases)
            out_q.extend(np.broadcast_to(
                seed, (len(bases), n_targets, len(seed))))
        return out_bases, out_q

    if isinstance(initial_base_poses, Coordinates):
        initial_base_poses = [initial_base_poses]
    start_bases = []
    start_q = []
    start_heights = []
    if irm is not None:
        names = [fk_params['joint_names'][i] for i in non_mimic]
        if names != irm.joint_names:
            raise ValueError(
                'irm was built for joints {}, not {}'.format(
                    irm.joint_names, names))
        world = np.array([i for i in range(len(target_pos))
                          if not base_frame[i] and i not in target_fns],
                         dtype=np.int64)
        if len(world) == 0:
            raise ValueError(
                'irm needs at least one fixed world-frame target to place '
                'the base against')
        masks = _resolve_masks(rotation_mask, len(target_pos))[world]
        coords = [target_coords[i] for i in world]
    # Every height's starts go into one batch: the solver steps (x, y,
    # yaw) and carries each start's height along.
    for height in heights:
        n_before = len(start_bases)
        if irm is not None:
            proposed = irm.candidate_bases(
                coords, n_candidates=n_candidates, rotation_mask=list(masks),
                base_bounds=base_bounds, base_height=height,
                is_valid_base=is_valid_base)
            if len(proposed) == 0:
                warnings.warn(
                    'The inverse reachability map proposed no base for '
                    'these targets at base_height {}.'.format(height),
                    RuntimeWarning)
            # Targets on or following the base cast no votes and get no
            # sample seed.
            seeds = np.full(
                (len(proposed), len(target_pos), len(initial_q)), np.nan)
            seeds[:, world] = proposed.seeds
            # Neither seeding wins everywhere, so each proposed base is
            # solved from both: the arms from each of initial_angles, and
            # each arm from the sample that voted for the base.
            b, qs = from_seeds(list(proposed.bases))
            start_bases.extend(b)
            start_q.extend(qs)
            start_bases.extend(proposed.bases)
            start_q.extend(np.where(np.isnan(seeds), initial_q, seeds))
        poses = initial_base_poses
        if poses is None and len(start_bases) == n_before:
            poses = [robot_model._find_fullbody_root_link().worldcoords()]
        b, qs = from_seeds([_base_to_array(c) for c in poses or []])
        start_bases.extend(b)
        start_q.extend(qs)
        start_heights.extend([height] * (len(start_bases) - n_before))
    initial_bases = np.array(start_bases)
    initial_q = np.array(start_q)
    start_heights = np.array(start_heights)
    if verbose:
        print('find_base_placement: {} starts over {} base height(s)'.format(
            len(initial_bases), len(heights)))

    bases, q, pos_err, rot_err = solve_shared_base_ik(
        fk_params, target_pos, target_rot, initial_bases, initial_q,
        position_mask=position_mask, rotation_mask=rotation_mask,
        stop=stop, thre=thre, rthre=rthre, base_weight=base_weight,
        base_bounds=base_bounds, drop_unreachable=drop_unreachable,
        base_height=start_heights, margin_target=margin_target,
        margin_weight=margin_weight, rotation_tolerance=rotation_tolerance,
        base_frame=base_frame, target_fns=target_fns,
        maximize_margin=maximize_margin, margin_step=margin_step,
        max_margin_steps=max_margin_steps, verbose=verbose,
        backend=backend)

    success = (pos_err < thre) & (rot_err < rthre)
    margins = _joint_margins(fk_params, q)

    robot_joints = list(robot_model.joint_list)
    robot_index = [robot_joints.index(link_list[i].joint)
                   for i in non_mimic]
    current_av = robot_model.angle_vector().copy()

    results = []
    for k in range(len(bases)):
        avs = np.tile(current_av, (len(target_pos), 1))
        avs[:, robot_index] = q[k]
        pose = _planar_coords(*bases[k], z=start_heights[k])
        valid = True if is_valid_base is None else bool(is_valid_base(pose))
        if base_bounds is None:
            bound_distance = np.inf
        else:
            bound_distance = float(np.min(
                np.abs(bases[k, :2, None] - np.asarray(base_bounds))))
        at_bound = bound_distance < 1e-6
        solved_targets = [_evaluate_target(t, pose) if callable(t) else t
                          for t in target_coords]
        result = BasePlacementResult(
            pose, avs, success[k], pos_err[k], rot_err[k], margins[k], k,
            base_valid=valid, at_bound=at_bound,
            bound_distance=bound_distance, target_coords=solved_targets)
        result.score = float(result.min_joint_margin if score is None
                             else score(result))
        results.append(result)
    best = _rank(results)

    if apply:
        set_base_pose(robot_model, best.base_pose)
        robot_model.angle_vector(best.angle_vectors[0])
    return best

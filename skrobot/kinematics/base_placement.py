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

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.kinematics.differentiable import compute_geometric_jacobian_batched_numpy
from skrobot.kinematics.differentiable import extract_fk_parameters
from skrobot.kinematics.differentiable import forward_kinematics_ee_batched_numpy
from skrobot.kinematics.differentiable import rotation_error_so3_log_batch_numpy


class BasePlacementResult(object):
    """Result of :func:`find_base_placement`.

    Attributes
    ----------
    base_pose : skrobot.coordinates.Coordinates
        Pose of the robot's root link. Planar: on ``z = 0`` and turned
        about ``z`` only, the same convention as ``use_base='planar'``.
    angle_vectors : numpy.ndarray
        ``(n_targets, n_robot_joints)`` full angle vectors, one per target.
        Joints outside the solved chain keep the angles the model had when
        the solve started.
    success : numpy.ndarray
        ``(n_targets,)`` bool, whether each target was reached within the
        thresholds. A placement that misses targets is still returned;
        check this before using it.
    position_errors : numpy.ndarray
        ``(n_targets,)`` position error in meters (masked axes only).
    rotation_errors : numpy.ndarray
        ``(n_targets,)`` rotation error in radians (masked axes only).
    candidate_index : int
        Which initial base pose the returned placement started from.
    candidates : list of BasePlacementResult
        Every candidate's result, best first. Empty on the entries of this
        list themselves.
    """

    def __init__(self, base_pose, angle_vectors, success, position_errors,
                 rotation_errors, candidate_index):
        self.base_pose = base_pose
        self.angle_vectors = angle_vectors
        self.success = success
        self.position_errors = position_errors
        self.rotation_errors = rotation_errors
        self.candidate_index = candidate_index
        self.candidates = []

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
                'base_yaw={:.3f}>'.format(
                    self.n_reached, len(self.success),
                    np.round(self.base_pose.worldpos()[:2], 4),
                    _yaw_of(self.base_pose.worldrot())))


def _yaw_of(rot):
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def _planar_coords(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Coordinates(pos=np.array([x, y, 0.0]), rot=rot)


def _base_rotations(yaw):
    """``(..., 3, 3)`` rotations about z for an array of yaw angles."""
    c, s = np.cos(yaw), np.sin(yaw)
    zeros = np.zeros_like(yaw)
    ones = np.ones_like(yaw)
    return np.stack([
        np.stack([c, -s, zeros], axis=-1),
        np.stack([s, c, zeros], axis=-1),
        np.stack([zeros, zeros, ones], axis=-1),
    ], axis=-2)


def _resolve_mask(mask):
    if mask is True or mask is None:
        return np.ones(3)
    if mask is False:
        return np.zeros(3)
    if isinstance(mask, str):
        return np.array([float(a in mask.lower()) for a in 'xyz'])
    mask = np.asarray(mask, dtype=np.float64)
    if mask.shape != (3,):
        raise ValueError('mask must have 3 entries, got {}'.format(mask))
    return mask


def _targets_to_pos_rot(target_coords):
    if isinstance(target_coords, Coordinates):
        target_coords = [target_coords]
    if len(target_coords) == 0:
        raise ValueError('target_coords is empty')
    pos = np.array([c.worldpos() for c in target_coords], dtype=np.float64)
    rot = np.array([c.worldrot() for c in target_coords], dtype=np.float64)
    return pos, rot


def _root_link(robot_model):
    root_link = getattr(robot_model, 'root_link', None)
    if root_link is not None:
        return root_link
    for link in robot_model.link_list:
        if link.parent_link is None:
            return link
    raise RuntimeError('Cannot determine the root link of the robot model')


def _chain_fk_params(robot_model, link_list, move_target):
    """FK parameters of the arm chain, expressed in the root-link frame."""
    fk_params = extract_fk_parameters(robot_model, link_list, move_target)
    root_inv = _root_link(robot_model).worldcoords().inverse_transformation()
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
                 rotation_mask):
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
        self.n_targets = len(target_pos)
        self.row_mask = np.concatenate([position_mask, rotation_mask])

    def _full_angles(self, q):
        """Expand optimized angles to the chain's full joint vector."""
        flat = q.reshape(-1, self.n_opt)
        angles = np.zeros((flat.shape[0], self.fk_params['n_joints']))
        angles[:, self.non_mimic] = flat
        return angles

    def clip(self, q):
        clipped = np.clip(q, self.lower, self.upper)
        return np.where(self.unbounded, q, clipped)

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
        base_rot = _base_rotations(base[:, 2])  # (K, 3, 3)
        base_pos = np.concatenate(
            [base[:, :2], np.zeros((n_cand, 1))], axis=1)  # (K, 3)
        world_pos = np.einsum('kij,knj->kni', base_rot, pos) \
            + base_pos[:, None, :]
        world_rot = np.einsum('kij,knjl->knil', base_rot, rot)
        return world_pos, world_rot, jac, base_rot, base_pos

    def residual(self, base, q):
        """``(K, N, 6)`` masked world-frame error, target minus actual."""
        world_pos, world_rot, _, _, _ = self._ee_world(base, q, False)
        return self._residual(world_pos, world_rot)

    def _residual(self, world_pos, world_rot):
        n_cand = world_pos.shape[0]
        pos_err = self.target_pos[None] - world_pos
        tgt_rot = np.broadcast_to(self.target_rot[None], world_rot.shape)
        body_log = rotation_error_so3_log_batch_numpy(
            world_rot.reshape(-1, 3, 3), tgt_rot.reshape(-1, 3, 3))
        rot_err = np.einsum(
            'bij,bj->bi', world_rot.reshape(-1, 3, 3), body_log)
        rot_err = rot_err.reshape(n_cand, self.n_targets, 3)
        return np.concatenate([pos_err, rot_err], axis=-1) * self.row_mask

    def linearize(self, base, q):
        """Residual, base Jacobian ``(K, N, 6, 3)``, arm Jacobian."""
        world_pos, world_rot, jac, base_rot, base_pos = self._ee_world(
            base, q, True)
        err = self._residual(world_pos, world_rot)

        # The chain Jacobian is in the root frame; rotate both halves.
        arm_jac = np.concatenate([
            np.einsum('kij,knjm->knim', base_rot, jac[:, :, :3]),
            np.einsum('kij,knjm->knim', base_rot, jac[:, :, 3:]),
        ], axis=2)

        rel = world_pos - base_pos[:, None, :]
        base_jac = np.zeros(err.shape[:2] + (6, 3))
        base_jac[:, :, 0, 0] = 1.0
        base_jac[:, :, 1, 1] = 1.0
        base_jac[:, :, 0, 2] = -rel[..., 1]
        base_jac[:, :, 1, 2] = rel[..., 0]
        base_jac[:, :, 5, 2] = 1.0

        mask = self.row_mask[:, None]
        return err, base_jac * mask, arm_jac * mask


def _cost(err):
    return np.sum(err ** 2, axis=(1, 2))


def _clip_rows(err, max_pos_step, max_rot_step):
    """Cap each target's error so one far target cannot drag the base."""
    pos_norm = np.linalg.norm(err[..., :3], axis=-1, keepdims=True)
    rot_norm = np.linalg.norm(err[..., 3:], axis=-1, keepdims=True)
    pos_scale = np.minimum(1.0, max_pos_step / np.maximum(pos_norm, 1e-12))
    rot_scale = np.minimum(1.0, max_rot_step / np.maximum(rot_norm, 1e-12))
    return np.concatenate(
        [err[..., :3] * pos_scale, err[..., 3:] * rot_scale], axis=-1)


def _schur_step(err, base_jac, arm_jac, damping, base_weight):
    """Solve the arrow-shaped damped normal equations.

    The unknowns are one base step shared by every target and one arm step
    per target. Eliminating the arm steps leaves a 3x3 system for the base.
    """
    n_opt = arm_jac.shape[-1]
    base_jac = base_jac * base_weight
    lam = damping[:, None, None, None]

    h_aa = np.einsum('knri,knrj->knij', arm_jac, arm_jac) \
        + lam * np.eye(n_opt)
    h_ba = np.einsum('knri,knrj->knij', base_jac, arm_jac)  # (K,N,3,n)
    g_a = np.einsum('knri,knr->kni', arm_jac, err)
    g_b = np.einsum('knri,knr->ki', base_jac, err)
    h_bb = np.einsum('knri,knrj->kij', base_jac, base_jac) \
        + damping[:, None, None] * np.eye(3)

    # H_aa^-1 [H_ab, g_a] in one solve per target.
    rhs = np.concatenate(
        [np.swapaxes(h_ba, -1, -2), g_a[..., None]], axis=-1)
    sol = np.linalg.solve(h_aa, rhs)
    inv_h_ab = sol[..., :3]
    inv_g_a = sol[..., 3]

    schur = h_bb - np.einsum('knij,knjl->kil', h_ba, inv_h_ab)
    reduced = g_b - np.einsum('knij,knj->ki', h_ba, inv_g_a)
    d_base = np.linalg.solve(schur, reduced[..., None])[..., 0]
    d_arm = inv_g_a - np.einsum('knij,kj->kni', inv_h_ab, d_base)
    return d_base * base_weight, d_arm


def _errors(err):
    return (np.linalg.norm(err[..., :3], axis=-1),
            np.linalg.norm(err[..., 3:], axis=-1))


def _levenberg_marquardt(problem, bases, q, active, stop, thre, rthre,
                         base_weight, bounds, max_pos_step, max_rot_step):
    """Iterate on the active targets of every candidate, in place."""
    n_cand = bases.shape[0]
    weight = active[..., None].astype(np.float64)
    damping = np.full(n_cand, 1e-2)
    err, base_jac, arm_jac = problem.linearize(bases, q)
    cost = _cost(err * weight)
    for _ in range(stop):
        pos_norm, rot_norm = _errors(err)
        reached = (pos_norm < thre) & (rot_norm < rthre)
        done = np.all(reached | ~active, axis=1)
        if np.all(done):
            break
        d_base, d_arm = _schur_step(
            _clip_rows(err, max_pos_step, max_rot_step) * weight,
            base_jac * weight[..., None], arm_jac * weight[..., None],
            damping, base_weight)
        new_bases = bases + d_base
        if bounds is not None:
            new_bases[:, :2] = np.clip(new_bases[:, :2], bounds[0], bounds[1])
        new_q = problem.clip(q + d_arm)
        new_cost = _cost(problem.residual(new_bases, new_q) * weight)
        accept = (new_cost < cost) & ~done
        damping = np.where(accept, np.maximum(damping * 0.5, 1e-6),
                           np.minimum(damping * 4.0, 1e4))
        if np.any(accept):
            bases[accept] = new_bases[accept]
            q[accept] = new_q[accept]
            err, base_jac, arm_jac = problem.linearize(bases, q)
            cost = _cost(err * weight)
    return err


def solve_shared_base_ik(
        fk_params, target_pos, target_rot, initial_bases, initial_q,
        position_mask=None, rotation_mask=None, stop=200, thre=1e-3,
        rthre=np.deg2rad(1.0), base_weight=1.0, base_bounds=None,
        drop_unreachable=True, max_pos_step=0.1, max_rot_step=0.5):
    """Solve shared-base IK from several initial bases at once.

    Levenberg-Marquardt on the arrow-shaped system: each candidate keeps
    its own damping, which shrinks after a step that lowers the cost and
    grows after one that does not (the step is then discarded).

    Least squares lets a target that cannot be reached pull the base away
    from the ones that can. With ``drop_unreachable``, a candidate that
    stalls with targets unreached stops weighing the worst of them and
    solves again, until the targets it still weighs are all reached. A
    last pass then holds the base and moves each dropped target's arm as
    close as it gets, so the reported errors are those of the final base.

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
    position_mask, rotation_mask : bool, str or sequence, optional
        Which world axes to constrain. ``True`` (default) constrains all.
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
    problem = _SharedBaseProblem(
        fk_params, target_pos, target_rot,
        _resolve_mask(position_mask), _resolve_mask(rotation_mask))
    bases = np.array(initial_bases, dtype=np.float64).reshape(-1, 3)
    n_cand = bases.shape[0]
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

    active = np.ones((n_cand, problem.n_targets), dtype=bool)
    rounds = problem.n_targets if drop_unreachable else 1
    for _ in range(rounds):
        err = _levenberg_marquardt(
            problem, bases, q, active, stop, thre, rthre, base_weight,
            bounds, max_pos_step, max_rot_step)
        if not drop_unreachable:
            break
        pos_err, rot_err = _errors(err)
        failing = active & ~((pos_err < thre) & (rot_err < rthre))
        stalled = np.any(failing, axis=1)
        if not np.any(stalled):
            break
        score = np.where(failing, pos_err / thre + rot_err / rthre, -1.0)
        worst = np.argmax(score, axis=1)
        rows = np.where(stalled)[0]
        active[rows, worst[rows]] = False

    if not np.all(active):
        # Base held: only the dropped targets' arms still have work to do.
        err = _levenberg_marquardt(
            problem, bases, q, np.ones_like(active), stop, thre, rthre,
            np.zeros(3), bounds, max_pos_step, max_rot_step)
    bases[:, 2] = np.arctan2(np.sin(bases[:, 2]), np.cos(bases[:, 2]))
    pos_err, rot_err = _errors(err)
    return bases, q, pos_err, rot_err


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
    root_link = _root_link(robot_model)
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


def _base_to_array(coords):
    pos = coords.worldpos()
    return np.array([pos[0], pos[1], _yaw_of(coords.worldrot())])


def find_base_placement(
        robot_model, target_coords, move_target=None, link_list=None,
        joint_list=None, initial_base_poses=None, initial_angles='current',
        position_mask=True, rotation_mask=True, stop=200, thre=1e-3,
        rthre=np.deg2rad(1.0), base_weight=1.0, base_bounds=None,
        drop_unreachable=True, apply=False):
    """Find one planar base pose from which every target is reachable.

    Solves for a single base pose ``(x, y, yaw)`` of the robot's root link
    together with one arm configuration per target, so that the end
    effector reaches every target from the same base. Each entry of
    ``initial_base_poses`` starts an independent solve; all of them run
    vectorized and the best is returned (most targets reached, then the
    smallest total error).

    The optimization is local: from a poor start it can settle where only
    some targets are reached. ``result.success`` says which; nothing is
    hidden. Several spread-out initial poses make a full solution more
    likely.

    Parameters
    ----------
    robot_model : skrobot.model.RobotModel
        Robot. Its root link is the mobile base.
    target_coords : Coordinates or list of Coordinates
        End-effector targets in the world frame.
    move_target : CascadedCoords, optional
        End effector. Defaults to ``robot_model.end_coords``.
    link_list : list of Link, optional
        Links whose joints are actuated. Defaults to the chain from the
        root to ``move_target``.
    joint_list : list of Joint, optional
        Joints to actuate; alternative to ``link_list``.
    initial_base_poses : Coordinates or list of Coordinates, optional
        Starting base poses. Defaults to the root link's current pose,
        projected onto the floor plane.
    initial_angles : 'current' or numpy.ndarray
        Arm angles to start every target from: ``'current'`` uses the
        model's angles, an array is ``(n_opt,)`` over the chain's
        non-mimic joints.
    position_mask, rotation_mask : bool, str or sequence
        World axes to constrain, e.g. ``rotation_mask=False`` for
        position-only targets or ``'z'`` for a vertical approach axis.
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
        worst of them and solve again, so the base settles for the
        targets it can serve instead of a compromise that serves none.
        Dropped targets are still reported in ``result.success``.
    apply : bool
        If True, move the robot to the found base pose and the first
        target's angle vector.

    Returns
    -------
    BasePlacementResult
        Best placement; ``result.candidates`` holds every start's result.

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
    if move_target is None:
        move_target = robot_model.end_coords
    if isinstance(move_target, (list, tuple)):
        raise ValueError(
            'find_base_placement solves for one end effector; pass '
            'move_target explicitly')
    link_list = _resolve_link_list(
        robot_model, move_target, link_list, joint_list)
    fk_params = _chain_fk_params(robot_model, link_list, move_target)
    target_pos, target_rot = _targets_to_pos_rot(target_coords)

    if initial_base_poses is None:
        initial_base_poses = [_root_link(robot_model).worldcoords()]
    elif isinstance(initial_base_poses, Coordinates):
        initial_base_poses = [initial_base_poses]
    initial_bases = np.array([_base_to_array(c) for c in initial_base_poses])

    mimic_parent = fk_params['mimic_parent_indices']
    non_mimic = [i for i in range(len(link_list)) if mimic_parent[i] < 0]
    if isinstance(initial_angles, str):
        if initial_angles != 'current':
            raise ValueError(
                "initial_angles must be 'current' or an array, got {!r}"
                .format(initial_angles))
        initial_q = np.array(
            [link_list[i].joint.joint_angle() for i in non_mimic])
    else:
        initial_q = np.asarray(initial_angles, dtype=np.float64)

    bases, q, pos_err, rot_err = solve_shared_base_ik(
        fk_params, target_pos, target_rot, initial_bases, initial_q,
        position_mask=position_mask, rotation_mask=rotation_mask,
        stop=stop, thre=thre, rthre=rthre, base_weight=base_weight,
        base_bounds=base_bounds, drop_unreachable=drop_unreachable)

    success = (pos_err < thre) & (rot_err < rthre)
    n_reached = np.count_nonzero(success, axis=1)
    total_err = np.sum(pos_err, axis=1) + np.sum(rot_err, axis=1)
    order = np.lexsort((total_err, -n_reached))

    robot_joints = list(robot_model.joint_list)
    robot_index = [robot_joints.index(link_list[i].joint)
                   for i in non_mimic]
    current_av = robot_model.angle_vector().copy()

    candidates = []
    for k in order:
        avs = np.tile(current_av, (len(target_pos), 1))
        avs[:, robot_index] = q[k]
        candidates.append(BasePlacementResult(
            _planar_coords(*bases[k]), avs, success[k], pos_err[k],
            rot_err[k], int(k)))
    best = candidates[0]
    best.candidates = candidates

    if apply:
        set_base_pose(robot_model, best.base_pose)
        robot_model.angle_vector(best.angle_vectors[0])
    return best

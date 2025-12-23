import numpy as np
import scipy

from skrobot.coordinates.base import slerp_coordinates
from skrobot.planner.sqp_based import construct_smoothcost_fullmat
from skrobot.planner.utils import scipinize
from skrobot.pycompat import lru_cache


def interpolate_waypoints(waypoint_coords, n_divisions=10, closed_loop=False):
    """Interpolate between waypoint coordinates using SLERP.

    Parameters
    ----------
    waypoint_coords : list[Coordinates]
        List of waypoint coordinates (e.g., 4 corners of a rectangle)
    n_divisions : int
        Number of divisions between each pair of waypoints
    closed_loop : bool
        If True, also interpolate from last waypoint back to first

    Returns
    -------
    interpolated_coords : list[Coordinates]
        List of interpolated coordinates including original waypoints
    """
    if len(waypoint_coords) < 2:
        return list(waypoint_coords)

    interpolated = []
    n_waypoints = len(waypoint_coords)

    pairs = list(range(n_waypoints - 1))
    if closed_loop:
        pairs.append(n_waypoints - 1)

    for i in pairs:
        c1 = waypoint_coords[i]
        c2 = waypoint_coords[(i + 1) % n_waypoints]

        for j in range(n_divisions):
            t = j / n_divisions
            interpolated.append(slerp_coordinates(c1, c2, t))

    if not closed_loop:
        interpolated.append(waypoint_coords[-1])

    return interpolated


def generate_initial_trajectory_seeded_ik(
        robot,
        move_target,
        target_coords_list,
        link_list=None,
        rotation_axis=True,
        translation_axis=True,
        stop=100,
        thre=0.001,
        rthre=np.deg2rad(1.0)):
    """Generate initial trajectory using seeded IK.

    Each IK solution uses the previous solution as initial guess,
    which helps maintain continuity in joint space.

    Parameters
    ----------
    robot : RobotModel
        Robot model instance
    move_target : Coordinates
        End-effector coordinates
    target_coords_list : list[Coordinates]
        List of target coordinates for the trajectory
    link_list : list[Link], optional
        Link list for IK. If None, automatically determined
    rotation_axis : bool or str
        Rotation constraint for IK
    translation_axis : bool or str
        Translation constraint for IK
    stop : int
        Maximum IK iterations
    thre : float
        Position threshold
    rthre : float
        Rotation threshold in radians

    Returns
    -------
    trajectory : np.ndarray (n_waypoints, n_dof)
        Joint angle trajectory
    success_flags : list[bool]
        Success flag for each waypoint
    """
    if move_target is None:
        move_target = robot.end_coords
    if link_list is None:
        link_list = robot.link_lists(move_target.parent)

    n_waypoints = len(target_coords_list)
    joint_list = robot.joint_list_from_link_list(link_list, ignore_fixed_joint=True)
    n_dof = len(joint_list)

    trajectory = np.zeros((n_waypoints, n_dof))
    success_flags = []

    initial_angles = robot.angle_vector().copy()

    for i, target_coords in enumerate(target_coords_list):
        result = robot.inverse_kinematics(
            target_coords,
            move_target=move_target,
            link_list=link_list,
            rotation_axis=rotation_axis,
            translation_axis=translation_axis,
            stop=stop,
            thre=thre,
            rthre=rthre)

        if result is not False and result is not None:
            joint_angles = np.array([j.joint_angle() for j in joint_list])
            trajectory[i] = joint_angles
            success_flags.append(True)
        else:
            # IK failed, use previous solution or initial angles
            if i > 0:
                # Set joints to previous solution
                for j_idx, joint in enumerate(joint_list):
                    joint.joint_angle(trajectory[i - 1, j_idx])
            joint_angles = np.array([j.joint_angle() for j in joint_list])
            trajectory[i] = joint_angles
            success_flags.append(False)

    robot.angle_vector(initial_angles)
    return trajectory, success_flags


def _compute_fk_error(robot, move_target, link_list, joint_list,
                      joint_angles, target_coords,
                      rotation_axis=True, translation_axis=True):
    """Compute forward kinematics error for a single configuration.

    Returns position and rotation errors between current and target poses.
    """
    for j, joint in enumerate(joint_list):
        joint.joint_angle(joint_angles[j])

    current_pos = move_target.worldpos()
    current_rot = move_target.worldrot()
    target_pos = target_coords.worldpos()
    target_rot = target_coords.worldrot()

    pos_error = np.zeros(3)
    rot_error = np.zeros(3)

    if translation_axis is True:
        pos_error = target_pos - current_pos
    elif translation_axis is not False:
        axis_str = str(translation_axis).lower()
        if 'x' in axis_str:
            pos_error[0] = target_pos[0] - current_pos[0]
        if 'y' in axis_str:
            pos_error[1] = target_pos[1] - current_pos[1]
        if 'z' in axis_str:
            pos_error[2] = target_pos[2] - current_pos[2]

    if rotation_axis is True:
        rel_rot = np.dot(target_rot, current_rot.T)
        trace_val = np.clip((np.trace(rel_rot) - 1) / 2, -1, 1)
        angle = np.arccos(trace_val)
        if angle > 1e-6:
            axis = np.array([
                rel_rot[2, 1] - rel_rot[1, 2],
                rel_rot[0, 2] - rel_rot[2, 0],
                rel_rot[1, 0] - rel_rot[0, 1]
            ]) / (2 * np.sin(angle))
            rot_error = axis * angle
    elif rotation_axis is not False:
        axis_str = str(rotation_axis).lower()
        rel_rot = np.dot(target_rot, current_rot.T)
        trace_val = np.clip((np.trace(rel_rot) - 1) / 2, -1, 1)
        angle = np.arccos(trace_val)
        if angle > 1e-6:
            axis = np.array([
                rel_rot[2, 1] - rel_rot[1, 2],
                rel_rot[0, 2] - rel_rot[2, 0],
                rel_rot[1, 0] - rel_rot[0, 1]
            ]) / (2 * np.sin(angle))
            full_rot_error = axis * angle
            if 'x' not in axis_str:
                full_rot_error[0] = 0
            if 'y' not in axis_str:
                full_rot_error[1] = 0
            if 'z' not in axis_str:
                full_rot_error[2] = 0
            rot_error = full_rot_error

    return pos_error, rot_error


def _compute_fk_jacobian(robot, move_target, link_list, joint_list,
                         rotation_axis=True, translation_axis=True):
    """Compute Jacobian matrix for the end-effector."""
    n_dof = len(joint_list)
    jacobian = np.zeros((6, n_dof))

    ee_pos = move_target.worldpos()

    for j, joint in enumerate(joint_list):
        if joint.joint_type == 'rotational':
            joint_pos = joint.child_link.worldpos()
            joint_axis = joint.child_link.worldrot()[:, joint.axis].flatten()

            jacobian[:3, j] = np.cross(joint_axis, ee_pos - joint_pos)
            jacobian[3:6, j] = joint_axis
        elif joint.joint_type == 'linear':
            joint_axis = joint.child_link.worldrot()[:, joint.axis].flatten()
            jacobian[:3, j] = joint_axis
            jacobian[3:6, j] = 0

    return jacobian


@lru_cache(maxsize=100)
def _get_task_weight_matrix(n_wp, pos_weight, rot_weight):
    """Construct weight matrix for task space constraints."""
    single_weights = np.array([pos_weight] * 3 + [rot_weight] * 3)
    return np.tile(single_weights, n_wp)


def plan_smooth_trajectory_ik(
        robot,
        move_target,
        waypoint_coords,
        link_list=None,
        n_divisions=10,
        closed_loop=False,
        rotation_axis=True,
        translation_axis=True,
        position_tolerance=0.005,
        rotation_tolerance=np.deg2rad(3.0),
        weights=None,
        position_weight=100.0,
        rotation_weight=50.0,
        slsqp_options=None,
        stop=100,
        thre=0.001,
        rthre=np.deg2rad(1.0),
        verbose=False):
    """Plan smooth trajectory through waypoints using SQP optimization.

    This function generates a smooth joint trajectory through multiple
    Cartesian waypoints. It first generates an initial trajectory using
    seeded IK, then optimizes the entire trajectory to minimize joint
    accelerations while maintaining end-effector position/orientation
    constraints.

    Parameters
    ----------
    robot : RobotModel
        Robot model instance
    move_target : Coordinates
        End-effector coordinates
    waypoint_coords : list[Coordinates]
        List of waypoint coordinates (e.g., 4 corners of a rectangle)
    link_list : list[Link], optional
        Link list for IK. If None, automatically determined
    n_divisions : int
        Number of interpolation divisions between each waypoint pair
    closed_loop : bool
        If True, create closed trajectory (last point connects to first)
    rotation_axis : bool or str
        Rotation constraint. True for full 3-axis, False for none,
        or string like 'z' for single axis
    translation_axis : bool or str
        Translation constraint. True for full 3-axis, False for none,
        or string like 'xy' for planar
    position_tolerance : float
        Position constraint tolerance in meters
    rotation_tolerance : float
        Rotation constraint tolerance in radians
    weights : list[float], optional
        Joint weights for smoothness cost. If None, uses uniform weights
    position_weight : float
        Weight for position constraint violation penalty
    rotation_weight : float
        Weight for rotation constraint violation penalty
    slsqp_options : dict, optional
        Options for scipy SLSQP optimizer
    stop : int
        Maximum IK iterations for initial trajectory
    thre : float
        Position threshold for IK
    rthre : float
        Rotation threshold for IK
    verbose : bool
        If True, print optimization progress

    Returns
    -------
    trajectory : np.ndarray (n_waypoints, n_dof)
        Optimized joint angle trajectory
    interpolated_coords : list[Coordinates]
        List of interpolated target coordinates
    success : bool
        True if optimization succeeded
    info : dict
        Dictionary containing:
        - 'initial_trajectory': trajectory before optimization
        - 'ik_success_flags': success flag for each initial IK
        - 'optimization_result': scipy optimization result
        - 'position_errors': final position errors
        - 'rotation_errors': final rotation errors

    Examples
    --------
    >>> from skrobot.models import PR2
    >>> from skrobot.coordinates import Coordinates
    >>> import numpy as np
    >>>
    >>> robot = PR2()
    >>> robot.reset_manip_pose()
    >>>
    >>> # Define 4 corners of a rectangle
    >>> corners = [
    ...     Coordinates(pos=[0.6, 0.2, 0.8]),
    ...     Coordinates(pos=[0.6, -0.2, 0.8]),
    ...     Coordinates(pos=[0.6, -0.2, 1.0]),
    ...     Coordinates(pos=[0.6, 0.2, 1.0]),
    ... ]
    >>>
    >>> trajectory, coords, success, info = plan_smooth_trajectory_ik(
    ...     robot,
    ...     robot.rarm_end_coords,
    ...     corners,
    ...     link_list=robot.rarm.link_list,
    ...     n_divisions=10,
    ...     closed_loop=True,
    ... )
    """
    if move_target is None:
        move_target = robot.end_coords
    if link_list is None:
        link_list = robot.link_lists(move_target.parent)

    interpolated_coords = interpolate_waypoints(
        waypoint_coords, n_divisions, closed_loop)
    n_wp = len(interpolated_coords)

    joint_list = robot.joint_list_from_link_list(link_list, ignore_fixed_joint=True)
    n_dof = len(joint_list)

    initial_robot_angles = robot.angle_vector().copy()

    if verbose:
        print(f"Generating initial trajectory with {n_wp} waypoints...")

    initial_trajectory, ik_success_flags = generate_initial_trajectory_seeded_ik(
        robot, move_target, interpolated_coords, link_list,
        rotation_axis=rotation_axis,
        translation_axis=translation_axis,
        stop=stop, thre=thre, rthre=rthre)

    n_ik_success = sum(ik_success_flags)
    if verbose:
        print(f"Initial IK: {n_ik_success}/{n_wp} succeeded")

    if n_ik_success < n_wp * 0.5:
        robot.angle_vector(initial_robot_angles)
        return initial_trajectory, interpolated_coords, False, {
            'initial_trajectory': initial_trajectory.copy(),
            'ik_success_flags': ik_success_flags,
            'optimization_result': None,
            'position_errors': None,
            'rotation_errors': None,
            'message': 'Too many IK failures in initial trajectory'
        }

    if weights is None:
        weights = [1.0] * n_dof
    weights = tuple(weights)

    smoothness_matrix = construct_smoothcost_fullmat(n_wp, n_dof, weights)

    joint_limits = np.array([[j.min_angle, j.max_angle] for j in joint_list])

    def objective_function(xi):
        """Smoothness objective: minimize acceleration."""
        f = (0.5 * smoothness_matrix.dot(xi).dot(xi)) / n_wp
        grad = smoothness_matrix.dot(xi) / n_wp
        return f, grad

    def task_constraint(xi):
        """Task space constraint: end-effector must be near targets."""
        av_seq = xi.reshape(n_wp, n_dof)

        pos_errors = np.zeros((n_wp, 3))
        rot_errors = np.zeros((n_wp, 3))

        for i in range(n_wp):
            for j, joint in enumerate(joint_list):
                joint.joint_angle(av_seq[i, j])

            pos_err, rot_err = _compute_fk_error(
                robot, move_target, link_list, joint_list,
                av_seq[i], interpolated_coords[i],
                rotation_axis, translation_axis)
            pos_errors[i] = pos_err
            rot_errors[i] = rot_err

        pos_norms = np.linalg.norm(pos_errors, axis=1)
        rot_norms = np.linalg.norm(rot_errors, axis=1)

        pos_constraint = position_tolerance - pos_norms
        rot_constraint = rotation_tolerance - rot_norms

        constraint_values = np.concatenate([pos_constraint, rot_constraint])

        jacobian = np.zeros((2 * n_wp, n_dof * n_wp))
        for i in range(n_wp):
            for j, joint in enumerate(joint_list):
                joint.joint_angle(av_seq[i, j])

            J = _compute_fk_jacobian(
                robot, move_target, link_list, joint_list,
                rotation_axis, translation_axis)

            if pos_norms[i] > 1e-8:
                dpos_dq = -pos_errors[i] / pos_norms[i]
                jacobian[i, i * n_dof:(i + 1) * n_dof] = -dpos_dq @ J[:3, :]

            if rot_norms[i] > 1e-8:
                drot_dq = -rot_errors[i] / rot_norms[i]
                jacobian[n_wp + i, i * n_dof:(i + 1) * n_dof] = -drot_dq @ J[3:6, :]

        return constraint_values, jacobian

    if slsqp_options is None:
        slsqp_options = {'ftol': 1e-6, 'disp': verbose, 'maxiter': 200}

    bounds = list(zip(joint_limits[:, 0], joint_limits[:, 1])) * n_wp

    obj_scipy, obj_jac_scipy = scipinize(objective_function)
    ineq_scipy, ineq_jac_scipy = scipinize(task_constraint)

    ineq_dict = {
        'type': 'ineq',
        'fun': ineq_scipy,
        'jac': ineq_jac_scipy
    }

    xi_init = initial_trajectory.reshape(-1)

    if verbose:
        print("Starting SQP optimization...")

    result = scipy.optimize.minimize(
        obj_scipy, xi_init,
        method='SLSQP',
        jac=obj_jac_scipy,
        bounds=bounds,
        constraints=[ineq_dict],
        options=slsqp_options)

    optimized_trajectory = result.x.reshape(n_wp, n_dof)

    final_pos_errors = np.zeros(n_wp)
    final_rot_errors = np.zeros(n_wp)

    for i in range(n_wp):
        for j, joint in enumerate(joint_list):
            joint.joint_angle(optimized_trajectory[i, j])

        pos_err, rot_err = _compute_fk_error(
            robot, move_target, link_list, joint_list,
            optimized_trajectory[i], interpolated_coords[i],
            rotation_axis, translation_axis)
        final_pos_errors[i] = np.linalg.norm(pos_err)
        final_rot_errors[i] = np.linalg.norm(rot_err)

    robot.angle_vector(initial_robot_angles)

    success = result.success and np.all(final_pos_errors < position_tolerance * 2)

    if verbose:
        print(f"Optimization {'succeeded' if success else 'failed'}")
        print(f"Max position error: {np.max(final_pos_errors):.4f} m")
        print(f"Max rotation error: {np.rad2deg(np.max(final_rot_errors)):.2f} deg")

    info = {
        'initial_trajectory': initial_trajectory.copy(),
        'ik_success_flags': ik_success_flags,
        'optimization_result': result,
        'position_errors': final_pos_errors,
        'rotation_errors': final_rot_errors
    }

    return optimized_trajectory, interpolated_coords, success, info


def compute_trajectory_smoothness(trajectory):
    """Compute smoothness metrics for a joint trajectory.

    Parameters
    ----------
    trajectory : np.ndarray (n_wp, n_dof)
        Joint angle trajectory

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'max_velocity': maximum joint velocity (rad/step)
        - 'max_acceleration': maximum joint acceleration (rad/step^2)
        - 'total_path_length': sum of joint angle changes
        - 'velocity_variance': variance of velocities
    """
    velocities = np.diff(trajectory, axis=0)
    accelerations = np.diff(velocities, axis=0)

    return {
        'max_velocity': np.max(np.abs(velocities)),
        'max_acceleration': np.max(np.abs(accelerations)),
        'total_path_length': np.sum(np.abs(velocities)),
        'velocity_variance': np.var(velocities),
        'mean_velocity': np.mean(np.abs(velocities)),
        'mean_acceleration': np.mean(np.abs(accelerations))
    }

"""High-level inverse kinematics via trajectory optimization.

This module wraps :class:`skrobot.planner.trajectory_optimization.TrajectoryProblem`
so that a caller can pose a single-shot or trajectory IK problem
(optionally with sphere obstacles) without having to wire residuals,
endpoint constraints and a solver by hand.

The key entry point is :func:`solve_ik`. It always builds a multi-waypoint
problem; when the caller passes a single :class:`Coordinates` target the
problem is treated as ``start_config -> target`` with the start fixed,
and only the final waypoint angle vector is returned as the IK answer.
When a list of targets is given, the targets are tracked at the
corresponding waypoints.
"""

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.planner.trajectory_optimization import create_solver
from skrobot.planner.trajectory_optimization import TrajectoryProblem


def _coerce_targets(target_coords):
    """Return a list of Coordinates regardless of input shape."""
    if isinstance(target_coords, Coordinates):
        return [target_coords]
    return list(target_coords)


def _stack_targets(targets, n_waypoints):
    """Build (n_waypoints, 3) positions and (n_waypoints, 3, 3) rotations.

    If only one target is given, it is broadcast to every waypoint.
    Otherwise the list must already match n_waypoints.
    """
    if len(targets) == 1:
        positions = np.tile(targets[0].worldpos(), (n_waypoints, 1))
        rotations = np.tile(targets[0].worldrot(), (n_waypoints, 1, 1))
    else:
        if len(targets) != n_waypoints:
            raise ValueError(
                'len(target_coords) ({}) must equal n_waypoints ({}) '
                'when passing a list of targets'.format(
                    len(targets), n_waypoints))
        positions = np.stack([t.worldpos() for t in targets])
        rotations = np.stack([t.worldrot() for t in targets])
    return positions, rotations


def _current_angles(joint_list):
    return np.array(
        [j.joint_angle() for j in joint_list], dtype=np.float64)


def _apply_angles(joint_list, angles):
    for j, q in zip(joint_list, angles):
        j.joint_angle(q)


def solve_ik(
    robot_model,
    link_list,
    move_target,
    target_coords,
    *,
    n_waypoints=None,
    initial_angles=None,
    collision_link_list=None,
    world_obstacles=None,
    solver='jaxls',
    max_iterations=200,
    position_weight=10.0,
    rotation_weight=1.0,
    smoothness_weight=0.01,
    collision_weight=1000.0,
    collision_activation=0.05,
    collision_as_constraint=True,
    n_spheres_per_link=5,
    position_mask=None,
    rotation_mask=None,
    apply_result=True,
    compute_torque=True,
    verbose=False,
):
    """Solve IK by minimising a TrajectoryProblem.

    The problem keeps the start waypoint fixed at ``initial_angles``
    (defaults to the robot's current joint configuration) and lets the
    remaining waypoints float. End-effector pose tracking is done with
    :meth:`TrajectoryProblem.add_cartesian_path_cost`.

    Parameters
    ----------
    robot_model : skrobot.model.RobotModel
        Robot model whose joints will be optimised.
    link_list : list of skrobot.model.Link
        Kinematic chain (used as the IK DoF) in root-to-tip order.
    move_target : skrobot.coordinates.CascadedCoords
        End-effector frame attached to the chain.
    target_coords : Coordinates or list of Coordinates
        Goal pose(s). A single ``Coordinates`` is treated as a static
        target replicated over every waypoint. A list is treated as a
        per-waypoint trajectory and must have length ``n_waypoints``.
    n_waypoints : int, optional
        Number of waypoints. Defaults to ``len(target_coords)`` when a
        list is given, else 2 (start + goal).
    initial_angles : array-like, optional
        Joint angles to seed the start waypoint with. Defaults to the
        current ``joint_angle()`` of each joint in the chain.
    collision_link_list : list of Link, optional
        Links whose swept-sphere approximation participates in the
        collision cost. Required iff ``world_obstacles`` is given.
    world_obstacles : list of dict, optional
        Sphere obstacles, each ``{'type': 'sphere', 'center': (3,),
        'radius': float}``. Box obstacles are not handled by the
        underlying solvers; approximate them as one or more spheres
        before passing them in.
    solver : str
        ``'jaxls'`` (default; Levenberg-Marquardt least squares with
        hard-constraint support), ``'augmented_lagrangian'``,
        ``'gradient_descent'``, or ``'scipy'``.
    max_iterations : int
        Iteration budget passed to the solver.
    position_weight, rotation_weight : float
        Weights for the Cartesian path cost.
    smoothness_weight : float
        Weight for the smoothness cost between consecutive waypoints.
    collision_weight, collision_activation : float
        Forwarded to ``add_collision_cost`` when obstacles are given.
    collision_as_constraint : bool
        Treat collision avoidance as a hard inequality constraint when
        the solver supports it (jaxls / augmented_lagrangian). When
        False, collision becomes a soft penalty controlled by
        ``collision_weight``.
    n_spheres_per_link : int
        Number of swept spheres approximating each collision link.
        Higher values mean tighter collision approximation. The
        scikit-robot historical default is 3; we bump to 5 to give
        roughly optmotiongen-comparable accuracy on Tycoon-sized
        cubes.
    position_mask, rotation_mask : array-like length 3, optional
        Per-axis selection of which translation / rotation error
        components contribute to the Cartesian cost. ``[0, 0, 1]``
        keeps only Z, mirroring optmotiongen's
        ``:translation-axis :z`` style. ``rotation_mask=[0, 0, 0]``
        disables rotation tracking.
    apply_result : bool
        If True (default), set the robot joints to the final waypoint
        angles before returning. Set to False to leave the model
        untouched.
    compute_torque : bool
        If True, compute the joint torque vector at the final pose
        using ``RobotModel.torque_vector``. Stored under
        ``torque_vector`` in the returned dict.
    verbose : bool
        Forwarded to the solver.

    Returns
    -------
    dict
        Keys:

        - ``success`` : bool reported by the solver.
        - ``angle_vector`` : (n_joints,) array of the final waypoint.
        - ``angle_vector_list`` : (n_waypoints, n_joints) array.
        - ``cost`` : final objective value.
        - ``iterations`` : iteration count.
        - ``position_error`` : ``||ee_pos - last_target_pos||`` after
          applying the final waypoint.
        - ``rotation_error`` : ``angle of (ee_rot)^T * target_rot`` (rad).
        - ``torque_vector`` : (n_joints,) array of joint torques at
          the final pose, or ``None`` if ``compute_torque=False``.
        - ``message`` : solver message string.
    """
    multi_ee = isinstance(move_target, (list, tuple))

    if multi_ee:
        # Branched / Y-shape multi-EE with shared joints. Use the
        # union chain as the variable space, the first EE's
        # ancestor sub-chain as the main cartesian_path cost, and
        # ``add_extra_ee_cost`` for every additional EE. Collision
        # avoidance, smoothness, n_waypoints, jaxls hard-constraint
        # support, etc. all work through the standard
        # TrajectoryProblem pipeline.
        return _solve_ik_multi_ee_shared(
            robot_model=robot_model,
            link_list=link_list,
            move_target=move_target,
            target_coords=target_coords,
            n_waypoints=n_waypoints,
            initial_angles=initial_angles,
            collision_link_list=collision_link_list,
            world_obstacles=world_obstacles,
            solver=solver,
            max_iterations=max_iterations,
            position_weight=position_weight,
            rotation_weight=rotation_weight,
            smoothness_weight=smoothness_weight,
            collision_weight=collision_weight,
            collision_activation=collision_activation,
            collision_as_constraint=collision_as_constraint,
            n_spheres_per_link=n_spheres_per_link,
            position_mask=position_mask,
            rotation_mask=rotation_mask,
            apply_result=apply_result,
            compute_torque=compute_torque,
            verbose=verbose,
        )

    targets = _coerce_targets(target_coords)
    if n_waypoints is None:
        n_waypoints = len(targets) if len(targets) > 1 else 2

    if n_waypoints < 2:
        raise ValueError(
            'n_waypoints must be >= 2 (got {}); the start waypoint is '
            'always fixed.'.format(n_waypoints))

    joint_list = [link.joint for link in link_list]
    if initial_angles is None:
        initial_angles = _current_angles(joint_list)
    initial_angles = np.asarray(initial_angles, dtype=np.float64)

    track_rotation = (rotation_weight > 0.0
                      and (rotation_mask is None
                           or any(rotation_mask)))

    problem = TrajectoryProblem(
        robot_model=robot_model,
        link_list=link_list,
        n_waypoints=n_waypoints,
        move_target=move_target,
    )
    problem.set_fixed_endpoints(start=True, end=False)
    target_positions, target_rotations = _stack_targets(
        targets, n_waypoints)
    if not track_rotation:
        target_rotations = None
    problem.add_cartesian_path_cost(
        target_positions=target_positions,
        target_rotations=target_rotations,
        weight=position_weight,
        rotation_weight=rotation_weight / max(position_weight, 1e-12),
        position_mask=position_mask,
        rotation_mask=rotation_mask,
    )
    if smoothness_weight > 0.0 and n_waypoints >= 2:
        problem.add_smoothness_cost(weight=smoothness_weight)

    if world_obstacles:
        if not collision_link_list:
            raise ValueError(
                'collision_link_list is required when world_obstacles '
                'is provided')
        problem.add_collision_cost(
            collision_link_list=collision_link_list,
            world_obstacles=world_obstacles,
            weight=collision_weight,
            activation_distance=collision_activation,
            as_constraint=collision_as_constraint,
            n_spheres_per_link=n_spheres_per_link,
        )

    init_traj = np.tile(initial_angles, (n_waypoints, 1))
    solver_obj = create_solver(
        solver, max_iterations=max_iterations, verbose=verbose)
    result = solver_obj.solve(problem, init_traj)

    angle_vector_list = np.asarray(result.trajectory)
    final_angles = angle_vector_list[-1]

    saved_angles = _current_angles(joint_list)
    _apply_angles(joint_list, final_angles)
    robot_model.update()

    ee_pos = move_target.worldpos()
    ee_rot = move_target.worldrot()
    per_ee_pos_err = [float(np.linalg.norm(
        (ee_pos - target_positions[-1])
        * (np.asarray(position_mask, dtype=np.float64)
           if position_mask is not None
           else np.ones(3))))]
    position_error = per_ee_pos_err[0]
    if track_rotation:
        target_rot = _stack_targets(targets, n_waypoints)[1][-1]
        rel = ee_rot.T @ target_rot
        cos_angle = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
        rotation_error = float(np.arccos(cos_angle))
    else:
        rotation_error = 0.0
    per_ee_rot_err = [rotation_error]

    torque_vector = None
    if compute_torque:
        try:
            torque_vector = np.asarray(
                robot_model.torque_vector(), dtype=np.float64)
        except Exception:
            torque_vector = None

    if not apply_result:
        _apply_angles(joint_list, saved_angles)
        robot_model.update()

    return {
        'success': bool(result.success),
        'angle_vector': np.asarray(final_angles),
        'angle_vector_list': angle_vector_list,
        'cost': float(result.cost),
        'iterations': int(result.iterations),
        'position_error': position_error,
        'rotation_error': rotation_error,
        'per_ee_position_error': per_ee_pos_err,
        'per_ee_rotation_error': per_ee_rot_err,
        'torque_vector': torque_vector,
        'message': str(result.message),
    }


def _solve_ik_multi_ee_shared(
    *,
    robot_model,
    link_list,
    move_target,
    target_coords,
    n_waypoints,
    initial_angles,
    collision_link_list,
    world_obstacles,
    solver,
    max_iterations,
    position_weight,
    rotation_weight,
    smoothness_weight,
    collision_weight,
    collision_activation,
    collision_as_constraint,
    n_spheres_per_link,
    position_mask,
    rotation_mask,
    apply_result,
    compute_torque,
    verbose,
):
    """Multi-EE IK with shared joints + collision via TrajectoryProblem.

    The first chain becomes the union variable space; subsequent EEs
    are added with ``add_extra_ee_cost``. The longest sub-chain (the
    one with the most ancestor joints) is taken as the union, so all
    other sub-chains' joint indices map cleanly into it.
    """
    if not (isinstance(link_list, list) and len(link_list) > 0
            and isinstance(link_list[0], list)):
        raise ValueError(
            'multi-EE solve requires link_list to be a list of '
            'chains (list of list of Link)')
    if len(link_list) != len(move_target):
        raise ValueError(
            'len(link_list) ({}) must equal len(move_target) ({}) '
            'in multi-EE mode'.format(len(link_list), len(move_target)))
    if not (isinstance(target_coords, (list, tuple))
            and len(target_coords) == len(move_target)):
        raise ValueError(
            'target_coords must be a list of {} entries (one per '
            'EE) in multi-EE mode'.format(len(move_target)))

    targets_per_chain = [_coerce_targets(tc) for tc in target_coords]
    if n_waypoints is None:
        longest = max(len(ts) for ts in targets_per_chain)
        n_waypoints = longest if longest > 1 else 2
    if n_waypoints < 2:
        raise ValueError(
            'n_waypoints must be >= 2 (got {}); the start waypoint is '
            'always fixed.'.format(n_waypoints))

    # Build the union chain by joint name from longest sub-chain plus
    # any joints uniquely contributed by other sub-chains.
    sub_chains = list(link_list)
    longest_idx = max(range(len(sub_chains)),
                      key=lambda i: len(sub_chains[i]))
    union_chain = list(sub_chains[longest_idx])
    union_names = {l.joint.name for l in union_chain}
    for ci, sub in enumerate(sub_chains):
        if ci == longest_idx:
            continue
        for link in sub:
            if link.joint.name not in union_names:
                # Insert at the position implied by ancestor order. To
                # keep the topological invariant simple we append; the
                # extra_ee chain_indices use joint-name match anyway.
                union_chain.append(link)
                union_names.add(link.joint.name)

    union_joint_list = [l.joint for l in union_chain]
    if initial_angles is None:
        initial_angles = _current_angles(union_joint_list)
    initial_angles = np.asarray(initial_angles, dtype=np.float64)

    track_rotation = (rotation_weight > 0.0
                      and (rotation_mask is None
                           or any(rotation_mask)))

    problem = TrajectoryProblem(
        robot_model=robot_model,
        link_list=union_chain,
        n_waypoints=n_waypoints,
        move_target=move_target[longest_idx],
    )
    problem.set_fixed_endpoints(start=True, end=False)

    # Main cost: longest sub-chain's EE.
    main_pos, main_rot = _stack_targets(
        targets_per_chain[longest_idx], n_waypoints)
    if not track_rotation:
        main_rot = None
    problem.add_cartesian_path_cost(
        target_positions=main_pos,
        target_rotations=main_rot,
        weight=position_weight,
        rotation_weight=rotation_weight / max(position_weight, 1e-12),
        position_mask=position_mask,
        rotation_mask=rotation_mask,
    )

    # Extra costs for the remaining EEs.
    for ci, sub in enumerate(sub_chains):
        if ci == longest_idx:
            continue
        sub_pos, sub_rot = _stack_targets(
            targets_per_chain[ci], n_waypoints)
        if not track_rotation:
            sub_rot = None
        problem.add_extra_ee_cost(
            move_target=move_target[ci],
            sub_chain_link_list=sub,
            target_positions=sub_pos,
            target_rotations=sub_rot,
            weight=position_weight,
            rotation_weight=(rotation_weight / max(position_weight, 1e-12)
                             if track_rotation else 0.0),
            position_mask=position_mask,
            rotation_mask=rotation_mask,
        )

    if smoothness_weight > 0.0 and n_waypoints >= 2:
        problem.add_smoothness_cost(weight=smoothness_weight)

    if world_obstacles:
        if not collision_link_list:
            raise ValueError(
                'collision_link_list is required when world_obstacles '
                'is provided')
        problem.add_collision_cost(
            collision_link_list=collision_link_list,
            world_obstacles=world_obstacles,
            weight=collision_weight,
            activation_distance=collision_activation,
            as_constraint=collision_as_constraint,
            n_spheres_per_link=n_spheres_per_link,
        )

    init_traj = np.tile(initial_angles, (n_waypoints, 1))
    solver_obj = create_solver(
        solver, max_iterations=max_iterations, verbose=verbose)
    result = solver_obj.solve(problem, init_traj)

    angle_vector_list = np.asarray(result.trajectory)
    final_angles = angle_vector_list[-1]

    saved_angles = _current_angles(union_joint_list)
    _apply_angles(union_joint_list, final_angles)
    robot_model.update()

    per_ee_pos_err = []
    per_ee_rot_err = []
    for ee, ts_list in zip(move_target, targets_per_chain):
        last_pos = ts_list[-1].worldpos()
        per_ee_pos_err.append(float(
            np.linalg.norm(ee.worldpos() - last_pos)))
        if track_rotation:
            rel = ee.worldrot().T @ ts_list[-1].worldrot()
            cos_angle = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
            per_ee_rot_err.append(float(np.arccos(cos_angle)))
        else:
            per_ee_rot_err.append(0.0)
    position_error = float(max(per_ee_pos_err))
    rotation_error = (float(max(per_ee_rot_err))
                      if track_rotation else 0.0)

    torque_vector = None
    if compute_torque:
        try:
            torque_vector = np.asarray(
                robot_model.torque_vector(), dtype=np.float64)
        except Exception:
            torque_vector = None

    if not apply_result:
        _apply_angles(union_joint_list, saved_angles)
        robot_model.update()

    return {
        'success': bool(result.success),
        'angle_vector': np.asarray(final_angles),
        'angle_vector_list': angle_vector_list,
        'cost': float(result.cost),
        'iterations': int(result.iterations),
        'position_error': position_error,
        'rotation_error': rotation_error,
        'per_ee_position_error': per_ee_pos_err,
        'per_ee_rotation_error': per_ee_rot_err,
        'torque_vector': torque_vector,
        'message': str(result.message),
    }


def _solve_ik_multi_ee(
    *,
    robot_model,
    link_list,
    move_target,
    target_coords,
    initial_angles,
    position_weight,
    rotation_weight,
    position_mask,
    rotation_mask,
    apply_result,
    compute_torque,
    verbose,
):
    """Multi-EE single-pose IK using ``RobotModel.inverse_kinematics``.

    Trajectory-optimization-based multi-EE (``add_multi_ee_waypoint_cost``)
    assumes each chain has *disjoint* joints. For branched chains (e.g.
    Y-shaped Tycoon compositions) the chains share joints, and the
    correct solver is the legacy SR-inverse Newton method that
    aggregates per-task Jacobians over a unified joint vector. This
    helper wraps that path so :func:`solve_ik` users get a single
    multi-EE entry point.
    """
    if not (isinstance(link_list, list) and len(link_list) > 0
            and isinstance(link_list[0], list)):
        raise ValueError(
            'multi-EE solve requires link_list to be a list of '
            'chains (list of list of Link)')
    if len(link_list) != len(move_target):
        raise ValueError(
            'len(link_list) ({}) must equal len(move_target) ({}) '
            'in multi-EE mode'.format(
                len(link_list), len(move_target)))
    if not (isinstance(target_coords, (list, tuple))
            and len(target_coords) == len(move_target)):
        raise ValueError(
            'target_coords must be a list of {} entries (one per '
            'EE) in multi-EE mode'.format(len(move_target)))

    targets_single = []
    for tc in target_coords:
        ts = _coerce_targets(tc)
        targets_single.append(ts[-1])

    union_joint_list = []
    seen = set()
    for chain in link_list:
        for link in chain:
            if link.joint is not None and link.joint.name not in seen:
                seen.add(link.joint.name)
                union_joint_list.append(link.joint)

    saved_angles = np.array(
        [j.joint_angle() for j in union_joint_list], dtype=np.float64)
    if initial_angles is not None:
        for j, q in zip(union_joint_list, np.asarray(initial_angles)):
            j.joint_angle(q)
    robot_model.update()

    pmask = (np.array(position_mask, dtype=np.int64)
             if position_mask is not None else np.array([1, 1, 1]))
    rmask = (np.array(rotation_mask, dtype=np.int64)
             if rotation_mask is not None
             else (np.array([1, 1, 1])
                   if rotation_weight > 0.0
                   else np.array([0, 0, 0])))
    pmasks = [pmask] * len(move_target)
    rmasks = [rmask] * len(move_target)

    success = robot_model.inverse_kinematics(
        target_coords=list(targets_single),
        move_target=list(move_target),
        link_list=list(link_list),
        position_mask=pmasks,
        rotation_mask=rmasks,
        revert_if_fail=False,
        stop=200,
        thre=[0.001] * len(move_target),
        rthre=[np.deg2rad(1.0)] * len(move_target),
    )
    success = bool(success is not False and success is not True
                   or success)
    robot_model.update()

    final_angles = np.array(
        [j.joint_angle() for j in union_joint_list], dtype=np.float64)

    per_ee_pos_err = []
    per_ee_rot_err = []
    for ee, t in zip(move_target, targets_single):
        per_ee_pos_err.append(float(np.linalg.norm(
            ee.worldpos() - t.worldpos())))
        rel = ee.worldrot().T @ t.worldrot()
        cos_angle = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
        per_ee_rot_err.append(float(np.arccos(cos_angle)))
    position_error = float(max(per_ee_pos_err))
    rotation_error = (float(max(per_ee_rot_err))
                      if rotation_weight > 0.0 else 0.0)

    torque_vector = None
    if compute_torque:
        try:
            torque_vector = np.asarray(
                robot_model.torque_vector(), dtype=np.float64)
        except Exception:
            torque_vector = None

    if not apply_result:
        for j, q in zip(union_joint_list, saved_angles):
            j.joint_angle(q)
        robot_model.update()

    n_waypoints = max(
        max(len(_coerce_targets(tc)) for tc in target_coords), 2)
    angle_vector_list = np.tile(final_angles, (n_waypoints, 1))

    return {
        'success': success,
        'angle_vector': final_angles,
        'angle_vector_list': angle_vector_list,
        'cost': 0.0,
        'iterations': 0,
        'position_error': position_error,
        'rotation_error': rotation_error,
        'per_ee_position_error': per_ee_pos_err,
        'per_ee_rotation_error': per_ee_rot_err,
        'torque_vector': torque_vector,
        'message': 'multi-EE SR-inverse',
    }


def box_to_spheres(center, extents, n_per_axis=2, rotation=None):
    """Approximate a box with a uniform sphere lattice.

    The trajectory optimization residuals only consume sphere
    obstacles, so a box has to be discretised before
    :func:`solve_ik` can avoid it. Each sphere covers the full
    diagonal of its cell, which is conservative (overestimates the
    box's footprint) but simple.

    Parameters
    ----------
    center : array-like (3,)
        Box centre in world coordinates (metres).
    extents : array-like (3,)
        Full box extents along x, y, z (metres) in the box's local
        frame.
    n_per_axis : int or sequence of 3 ints
        Number of spheres along each axis. Use a sequence to refine
        the long sides of an elongated box.
    rotation : (3, 3) ndarray, optional
        Rotation of the box. If provided, the lattice is generated in
        the box's local frame and rotated into world frame before being
        offset by ``center``. Defaults to identity (axis-aligned box).

    Returns
    -------
    list of dict
        Sphere obstacles consumable by :func:`solve_ik`.
    """
    center = np.asarray(center, dtype=np.float64)
    extents = np.asarray(extents, dtype=np.float64)
    if np.isscalar(n_per_axis):
        n = np.array([n_per_axis] * 3, dtype=np.int64)
    else:
        n = np.asarray(n_per_axis, dtype=np.int64)
    if n.shape != (3,):
        raise ValueError('n_per_axis must be int or length-3 sequence')

    cell = extents / n
    radius = float(np.linalg.norm(cell) / 2.0)

    if rotation is None:
        rotation_mat = np.eye(3)
    else:
        rotation_mat = np.asarray(rotation, dtype=np.float64)
        if rotation_mat.shape != (3, 3):
            raise ValueError('rotation must be a 3x3 matrix')

    starts_local = -extents / 2.0 + cell / 2.0
    spheres = []
    for ix in range(n[0]):
        for iy in range(n[1]):
            for iz in range(n[2]):
                local = starts_local + cell * np.array([ix, iy, iz])
                c = center + rotation_mat @ local
                spheres.append({
                    'type': 'sphere',
                    'center': c.tolist(),
                    'radius': radius,
                })
    return spheres

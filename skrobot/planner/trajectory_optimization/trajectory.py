"""Backend-agnostic trajectory utilities.

This module provides trajectory interpolation and smoothness cost functions
that work with any backend (NumPy, JAX, etc.).
"""

import numpy as np


def interpolate_trajectory(start_angles, end_angles, n_waypoints):
    """Create linear interpolation between start and end configurations.

    Parameters
    ----------
    start_angles : array-like
        Starting joint angles (n_joints,).
    end_angles : array-like
        Ending joint angles (n_joints,).
    n_waypoints : int
        Number of waypoints including start and end.

    Returns
    -------
    numpy.ndarray
        Interpolated trajectory (n_waypoints, n_joints).
    """
    start = np.array(start_angles)
    end = np.array(end_angles)
    t = np.linspace(0, 1, n_waypoints)[:, np.newaxis]
    return start + t * (end - start)


def compute_velocity(backend, trajectory, dt):
    """Compute velocity along trajectory using central differences.

    Parameters
    ----------
    backend : DifferentiableBackend
        Backend to use for computation.
    trajectory : array
        Joint trajectory (T, n_joints).
    dt : float
        Time step between waypoints.

    Returns
    -------
    array
        Velocity at each interior waypoint (T-2, n_joints).
    """
    # Central difference: v[t] = (q[t+1] - q[t-1]) / (2*dt)
    return (trajectory[2:] - trajectory[:-2]) / (2 * dt)


def compute_acceleration(backend, trajectory, dt):
    """Compute acceleration along trajectory using central differences.

    Parameters
    ----------
    backend : DifferentiableBackend
        Backend to use for computation.
    trajectory : array
        Joint trajectory (T, n_joints).
    dt : float
        Time step between waypoints.

    Returns
    -------
    array
        Acceleration at each interior waypoint (T-2, n_joints).
    """
    # Central difference: a[t] = (q[t+1] - 2*q[t] + q[t-1]) / dt^2
    return (trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]) / (dt ** 2)


def compute_jerk(backend, trajectory, dt):
    """Compute jerk along trajectory using finite differences.

    Parameters
    ----------
    backend : DifferentiableBackend
        Backend to use for computation.
    trajectory : array
        Joint trajectory (T, n_joints).
    dt : float
        Time step between waypoints.

    Returns
    -------
    array
        Jerk at interior waypoints (T-3, n_joints).
    """
    # Jerk from acceleration differences
    acc = compute_acceleration(backend, trajectory, dt)
    return (acc[1:] - acc[:-1]) / dt


def velocity_cost(backend, trajectory, dt, weight=1.0):
    """Compute velocity minimization cost.

    Parameters
    ----------
    backend : DifferentiableBackend
        Backend to use for computation.
    trajectory : array
        Joint trajectory (T, n_joints).
    dt : float
        Time step.
    weight : float
        Cost weight.

    Returns
    -------
    float
        Total velocity cost.
    """
    vel = compute_velocity(backend, trajectory, dt)
    return weight * backend.sum(vel ** 2)


def acceleration_cost(backend, trajectory, dt, weight=1.0):
    """Compute acceleration minimization cost.

    Parameters
    ----------
    backend : DifferentiableBackend
        Backend to use for computation.
    trajectory : array
        Joint trajectory (T, n_joints).
    dt : float
        Time step.
    weight : float
        Cost weight.

    Returns
    -------
    float
        Total acceleration cost.
    """
    acc = compute_acceleration(backend, trajectory, dt)
    return weight * backend.sum(acc ** 2)


def jerk_cost(backend, trajectory, dt, weight=1.0):
    """Compute jerk minimization cost.

    Parameters
    ----------
    backend : DifferentiableBackend
        Backend to use for computation.
    trajectory : array
        Joint trajectory (T, n_joints).
    dt : float
        Time step.
    weight : float
        Cost weight.

    Returns
    -------
    float
        Total jerk cost.
    """
    jrk = compute_jerk(backend, trajectory, dt)
    return weight * backend.sum(jrk ** 2)


def smoothness_cost(backend, trajectory, dt, vel_weight=0.1, acc_weight=1.0, jerk_weight=0.1):
    """Compute combined smoothness cost.

    Parameters
    ----------
    backend : DifferentiableBackend
        Backend to use for computation.
    trajectory : array
        Joint trajectory (T, n_joints).
    dt : float
        Time step.
    vel_weight : float
        Velocity cost weight.
    acc_weight : float
        Acceleration cost weight.
    jerk_weight : float
        Jerk cost weight.

    Returns
    -------
    float
        Total smoothness cost.
    """
    cost = 0.0
    if vel_weight > 0:
        cost = cost + velocity_cost(backend, trajectory, dt, vel_weight)
    if acc_weight > 0:
        cost = cost + acceleration_cost(backend, trajectory, dt, acc_weight)
    if jerk_weight > 0 and trajectory.shape[0] >= 5:
        cost = cost + jerk_cost(backend, trajectory, dt, jerk_weight)
    return cost


def velocity_limit_cost(backend, trajectory, dt, velocity_limits, weight=10.0):
    """Compute velocity limit violation cost.

    Parameters
    ----------
    backend : DifferentiableBackend
        Backend to use for computation.
    trajectory : array
        Joint trajectory (T, n_joints).
    dt : float
        Time step.
    velocity_limits : array
        Maximum velocity per joint (n_joints,).
    weight : float
        Cost weight.

    Returns
    -------
    float
        Total velocity limit violation cost.
    """
    vel = compute_velocity(backend, trajectory, dt)
    violations = backend.maximum(0.0, backend.abs(vel) - velocity_limits)
    return weight * backend.sum(violations ** 2)


# Legacy API compatibility - deprecated functions

def create_trajectory_optimizer(*args, **kwargs):
    """Deprecated: Use TrajectoryProblem with create_solver instead.

    This function is deprecated and will be removed in a future version.
    Please use the unified trajectory optimization interface:

        from skrobot.planner.trajectory_optimization import TrajectoryProblem
        from skrobot.planner.trajectory_optimization.solvers import create_solver

        problem = TrajectoryProblem(robot, link_list, n_waypoints)
        problem.add_smoothness_cost(weight=1.0)
        problem.add_collision_cost(collision_link_list, obstacles, weight=100.0)

        solver = create_solver('gradient_descent')  # or 'jaxls', 'scipy'
        result = solver.solve(problem, initial_trajectory)
    """
    import warnings
    warnings.warn(
        "create_trajectory_optimizer is deprecated. "
        "Use TrajectoryProblem with create_solver('gradient_descent') instead. "
        "See skrobot.planner.trajectory_optimization module.",
        DeprecationWarning,
        stacklevel=2
    )

    # Redirect to new unified interface
    from skrobot.planner.trajectory_optimization import TrajectoryProblem
    from skrobot.planner.trajectory_optimization.solvers import create_solver

    robot_model = args[0]
    link_list = args[1]
    move_target = args[2]
    dt = kwargs.get('dt', 0.1)
    collision_weight = kwargs.get('collision_weight', 0.0)
    self_collision_weight = kwargs.get('self_collision_weight', 0.0)
    world_obstacles = kwargs.get('world_obstacles', None)
    collision_link_list = kwargs.get('collision_link_list', None)
    activation_distance = kwargs.get('activation_distance', 0.02)
    acc_weight = kwargs.get('acc_weight', 1.0)
    jerk_weight = kwargs.get('jerk_weight', 0.1)

    # Create solver wrapper that mimics old API
    solver = create_solver('gradient_descent', max_iterations=300)

    def optimizer_wrapper(initial_trajectory, target_positions, target_rotations,
                          max_iterations=100, learning_rate=0.01, fix_endpoints=True):
        """Wrapper that mimics old optimizer API."""
        n_waypoints = initial_trajectory.shape[0]

        problem = TrajectoryProblem(
            robot_model=robot_model,
            link_list=link_list,
            n_waypoints=n_waypoints,
            dt=dt,
            move_target=move_target,
        )

        problem.add_smoothness_cost(weight=1.0)
        problem.add_acceleration_cost(weight=acc_weight)
        if jerk_weight > 0:
            problem.add_jerk_cost(weight=jerk_weight)

        if collision_weight > 0 and collision_link_list and world_obstacles:
            problem.add_collision_cost(
                collision_link_list=collision_link_list,
                world_obstacles=world_obstacles,
                weight=collision_weight,
                activation_distance=activation_distance,
            )

        if self_collision_weight > 0 and collision_link_list:
            problem.add_self_collision_cost(
                weight=self_collision_weight,
                activation_distance=activation_distance,
            )

        problem.fixed_start = fix_endpoints
        problem.fixed_end = fix_endpoints

        result = solver.solve(
            problem,
            np.array(initial_trajectory),
            max_iterations=max_iterations,
            learning_rate=learning_rate,
        )

        return result.trajectory

    return optimizer_wrapper


def smooth_trajectory(*args, **kwargs):
    """Deprecated: Use TrajectoryProblem with create_solver instead.

    See create_trajectory_optimizer for migration guide.
    """
    import warnings
    warnings.warn(
        "smooth_trajectory is deprecated. "
        "Use TrajectoryProblem with create_solver('gradient_descent') instead.",
        DeprecationWarning,
        stacklevel=2
    )

    robot_model = args[0]
    link_list = args[1]
    move_target = args[2]
    trajectory = args[3]
    target_coords_list = args[4]
    dt = kwargs.get('dt', 0.1)
    max_iterations = kwargs.get('max_iterations', 50)
    learning_rate = kwargs.get('learning_rate', 0.01)

    optimizer = create_trajectory_optimizer(
        robot_model, link_list, move_target, dt=dt
    )

    target_positions = np.array([c.worldpos() for c in target_coords_list])
    target_rotations = np.array([c.worldrot() for c in target_coords_list])

    return optimizer(
        np.array(trajectory),
        target_positions,
        target_rotations,
        max_iterations,
        learning_rate,
        True,
    )

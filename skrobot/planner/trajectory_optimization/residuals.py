"""Backend-agnostic residual computation functions.

These functions compute residuals for trajectory optimization.
They are designed to work with any array library (numpy, jax.numpy, torch).

Each residual function takes:
- trajectory: (T, n_joints) array of joint angles
- params: dict of precomputed parameters
- array_module: numpy-like module (np, jnp, torch)

Returns:
- residual: array of residual values (to be squared and summed by solver)
"""


def compute_smoothness_residual(
    q_curr,
    q_prev,
    weight,
    xp,
):
    """Compute smoothness residual between consecutive waypoints.

    Parameters
    ----------
    q_curr : array
        Current waypoint joint angles (n_joints,).
    q_prev : array
        Previous waypoint joint angles (n_joints,).
    weight : float
        Residual weight.
    xp : module
        Array module (numpy, jax.numpy, torch).

    Returns
    -------
    array
        Weighted difference residual.
    """
    return weight * (q_curr - q_prev)


def compute_acceleration_residual(
    q_curr,
    q_next,
    q_prev,
    dt,
    weight,
    xp,
):
    """Compute acceleration residual using central difference.

    Parameters
    ----------
    q_curr : array
        Current waypoint (n_joints,).
    q_next : array
        Next waypoint (n_joints,).
    q_prev : array
        Previous waypoint (n_joints,).
    dt : float
        Time step.
    weight : float
        Residual weight.
    xp : module
        Array module.

    Returns
    -------
    array
        Acceleration residual.
    """
    acc = (q_next - 2 * q_curr + q_prev) / (dt ** 2)
    return weight * acc


def compute_joint_limit_residual(
    q,
    lower_limits,
    upper_limits,
    xp,
):
    """Compute joint limit constraint residual.

    Returns positive values when limits are violated.

    Parameters
    ----------
    q : array
        Joint angles (n_joints,).
    lower_limits : array
        Lower joint limits (n_joints,).
    upper_limits : array
        Upper joint limits (n_joints,).
    xp : module
        Array module.

    Returns
    -------
    array
        Constraint residual (>= 0 means feasible).
    """
    lower_violation = lower_limits - q  # positive when q < lower
    upper_violation = q - upper_limits  # positive when q > upper
    return xp.concatenate([
        xp.maximum(0.0, lower_violation),
        xp.maximum(0.0, upper_violation),
    ])


def compute_collision_residual(
    sphere_positions,
    sphere_radii,
    obstacle_center,
    obstacle_radius,
    activation_distance,
    xp,
):
    """Compute collision avoidance residual.

    Parameters
    ----------
    sphere_positions : array
        World positions of collision spheres (n_spheres, 3).
    sphere_radii : array
        Radii of collision spheres (n_spheres,).
    obstacle_center : array
        Obstacle center (3,).
    obstacle_radius : float
        Obstacle radius.
    activation_distance : float
        Distance below which collision cost activates.
    xp : module
        Array module.

    Returns
    -------
    array
        Collision residual for each sphere.
    """
    diff = sphere_positions - obstacle_center[None, :]
    dists = xp.sqrt(xp.sum(diff ** 2, axis=-1) + 1e-10)
    signed_dists = dists - sphere_radii - obstacle_radius

    # Residual is positive when too close (collision cost)
    residuals = xp.maximum(0.0, activation_distance - signed_dists)
    return residuals


def compute_self_collision_residual(
    sphere_positions,
    sphere_radii,
    pair_indices_i,
    pair_indices_j,
    activation_distance,
    xp,
):
    """Compute self-collision avoidance residual.

    Parameters
    ----------
    sphere_positions : array
        World positions of collision spheres (n_spheres, 3).
    sphere_radii : array
        Radii of collision spheres (n_spheres,).
    pair_indices_i : array
        First sphere indices for each collision pair.
    pair_indices_j : array
        Second sphere indices for each collision pair.
    activation_distance : float
        Distance below which collision cost activates.
    xp : module
        Array module.

    Returns
    -------
    array
        Self-collision residual for each pair.
    """
    if len(pair_indices_i) == 0:
        return xp.array([])

    pos_i = sphere_positions[pair_indices_i]
    pos_j = sphere_positions[pair_indices_j]
    rad_i = sphere_radii[pair_indices_i]
    rad_j = sphere_radii[pair_indices_j]

    diff = pos_i - pos_j
    dists = xp.sqrt(xp.sum(diff ** 2, axis=-1) + 1e-10)
    signed_dists = dists - rad_i - rad_j

    residuals = xp.maximum(0.0, activation_distance - signed_dists)
    return residuals


def compute_pose_residual(
    position,
    rotation,
    target_position,
    target_rotation,
    position_weight,
    rotation_weight,
    xp,
):
    """Compute end-effector pose tracking residual.

    Parameters
    ----------
    position : array
        Current end-effector position (3,).
    rotation : array
        Current end-effector rotation matrix (3, 3).
    target_position : array
        Target position (3,).
    target_rotation : array
        Target rotation matrix (3, 3).
    position_weight : float
        Position error weight.
    rotation_weight : float
        Rotation error weight.
    xp : module
        Array module.

    Returns
    -------
    array
        Pose residual.
    """
    pos_residual = position_weight * (position - target_position)
    rot_residual = rotation_weight * (rotation - target_rotation).flatten()
    return xp.concatenate([pos_residual, rot_residual])


class ResidualSpec:
    """Specification for a residual function.

    This class holds the residual function and its parameters,
    allowing solvers to construct optimization problems.
    """

    def __init__(
        self,
        name,
        residual_fn,
        params,
        kind='soft',
        weight=1.0,
    ):
        """Initialize residual specification.

        Parameters
        ----------
        name : str
            Residual name for debugging.
        residual_fn : callable
            Function computing residual from variables and params.
        params : dict
            Parameters passed to residual function.
        kind : str
            'soft' for soft cost, 'eq' for equality constraint,
            'geq' for >= 0 constraint.
        weight : float
            Residual weight (for soft costs).
        """
        self.name = name
        self.residual_fn = residual_fn
        self.params = params
        self.kind = kind
        self.weight = weight

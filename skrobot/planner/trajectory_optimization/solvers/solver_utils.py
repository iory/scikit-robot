"""Shared utilities for trajectory optimization solvers."""

import hashlib


def get_problem_structure_key(problem):
    """Generate cache key from problem structure.

    This key captures the structure of the problem (shapes, cost types)
    but not the specific values (obstacle positions, targets).
    Used for JIT compilation caching.

    Parameters
    ----------
    problem : TrajectoryProblem
        Problem definition.

    Returns
    -------
    tuple
        Cache key tuple based on problem structure.
    """
    residual_names = tuple(sorted(r.name for r in problem.residuals))
    residual_weights = tuple(
        (r.name, r.weight) for r in problem.residuals
    )

    # Only include waypoint constraint *indices*, not values
    wp_constraint_indices = tuple(
        idx for idx, _ in problem.waypoint_constraints
    )

    # Include obstacle *count*, not positions (structure-based)
    n_obstacles = 0
    if problem.world_obstacles:
        n_obstacles = len(problem.world_obstacles)

    # Include EE waypoint cost *structure*, not target values
    ee_wp_structure = tuple(
        (c['waypoint_index'], c['position_weight'], c['rotation_weight'])
        for c in problem.ee_waypoint_costs
    )

    # Check if cartesian path exists (structure, not values)
    has_cartesian = any(
        spec.name == 'cartesian_path' for spec in problem.residuals
    )

    return (
        problem.n_waypoints,
        problem.n_joints,
        residual_names,
        residual_weights,
        problem.fixed_start,
        problem.fixed_end,
        problem.collision_spheres is not None,
        wp_constraint_indices,
        n_obstacles,
        ee_wp_structure,
        has_cartesian,
    )


def get_problem_value_hash(problem):
    """Generate hash of problem values (obstacle positions, targets, etc.).

    Combined with structure key to detect when functions need rebuilding.

    Parameters
    ----------
    problem : TrajectoryProblem
        Problem definition.

    Returns
    -------
    str or None
        MD5 hash of problem values, or None if no values to hash.
    """
    hash_parts = []

    # Hash obstacle positions
    if problem.world_obstacles:
        for obs in problem.world_obstacles:
            hash_parts.append(str(obs.get('center', [])))
            hash_parts.append(str(obs.get('radius', 0)))

    # Hash EE waypoint targets
    for c in problem.ee_waypoint_costs:
        hash_parts.append(c['target_position'].tobytes())
        hash_parts.append(c['target_rotation'].tobytes())

    # Hash cartesian path targets
    for spec in problem.residuals:
        if spec.name == 'cartesian_path':
            params = spec.params
            target_pos = params.get('target_positions')
            if target_pos is not None:
                hash_parts.append(target_pos.tobytes())
            target_rot = params.get('target_rotations')
            if target_rot is not None:
                hash_parts.append(target_rot.tobytes())

    # Hash waypoint constraint values
    for _, angles in problem.waypoint_constraints:
        hash_parts.append(angles.tobytes())

    if not hash_parts:
        return None

    combined = b''.join(
        p if isinstance(p, bytes) else p.encode() for p in hash_parts
    )
    return hashlib.md5(combined).hexdigest()[:16]


__all__ = [
    'get_problem_structure_key',
    'get_problem_value_hash',
]

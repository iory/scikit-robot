"""Shared utilities for trajectory optimization solvers."""

import hashlib

import numpy as np


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

    # The self-collision representation, and for 'gridsdf' the shape of its
    # residual block, decide how the cost is compiled -- so they are structure,
    # not values.
    self_collision_structure = tuple(
        _self_collision_structure(spec) for spec in problem.residuals
        if spec.name == 'self_collision'
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
        self_collision_structure,
    )


def _self_collision_structure(spec):
    """Cache-key contribution of one self-collision residual spec."""
    mode = spec.params.get('mode', 'sphere')
    if mode != 'gridsdf':
        return (mode, len(spec.params['pair_indices'][0]))
    data = spec.params['gridsdf_data']
    return (mode, len(data['pairs_a']), data['surface_points'].shape[1])


def _feed_value(md5, value):
    """Fold one problem value into ``md5``.

    Walks containers so a residual's params are covered whatever shape they
    take. Anything that is not a container is folded in through ``repr``,
    which is stable for the scalars and strings these params hold.
    """
    if isinstance(value, np.ndarray):
        md5.update(b'a')
        md5.update(repr((value.shape, value.dtype.str)).encode())
        md5.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        md5.update(b'd')
        for key in sorted(value, key=repr):
            md5.update(repr(key).encode())
            _feed_value(md5, value[key])
    elif isinstance(value, (list, tuple)):
        md5.update(b'l')
        for item in value:
            _feed_value(md5, item)
    else:
        md5.update(b's')
        md5.update(repr(value).encode())


def get_problem_value_hash(problem):
    """Hash every problem value the compiled functions close over.

    Combined with the structure key this decides whether a cached pair of
    compiled functions may serve a problem. The FK arrays are excluded: they
    travel as a runtime argument, so one executable serves every problem with
    the same structure regardless of kinematics.

    Everything else the residuals hold -- targets, obstacle geometry, sphere
    radii, axis masks, activation distances, grid contents -- is still baked
    into the trace, so it has to be part of the key. Rather than listing those
    fields, this walks the residual params wholesale: a field added by a later
    residual is then covered without anyone remembering to extend this
    function. Two earlier additions (the box obstacles' extents and rotation,
    and the Cartesian axis masks) were missed by the field-by-field version
    that came before, which would have let one problem's values serve another.

    Parameters
    ----------
    problem : TrajectoryProblem
        Problem definition.

    Returns
    -------
    str or None
        Hash of the problem's values, or None when it holds none.
    """
    md5 = hashlib.md5()
    empty = True

    if problem.world_obstacles:
        empty = False
        _feed_value(md5, problem.world_obstacles)

    for spec in problem.residuals:
        empty = False
        md5.update(repr(spec.name).encode())
        _feed_value(md5, spec.weight)
        _feed_value(md5, spec.params)

    for cost in problem.ee_waypoint_costs:
        empty = False
        _feed_value(md5, cost)

    for idx, angles in problem.waypoint_constraints:
        empty = False
        _feed_value(md5, idx)
        _feed_value(md5, angles)

    if problem.collision_spheres is not None:
        empty = False
        _feed_value(md5, problem.collision_spheres)

    if empty:
        return None
    return md5.hexdigest()[:16]


def build_gridsdf_self_distance_fn(problem, fk_data, backend,
                                   parameterized=False):
    """Build the GridSDF self-collision distance function of a problem.

    Parameters
    ----------
    problem : TrajectoryProblem
        Problem definition.
    fk_data : dict
        Output of
        :func:`~skrobot.planner.trajectory_optimization.fk_utils.prepare_fk_data`.
    backend : module
        Array module (``numpy`` or ``jax.numpy``).

    Returns
    -------
    callable or None
        ``f(angles) -> (n_pairs, n_surface)`` signed distances, or None if the
        problem has no self-collision cost with ``mode='gridsdf'``.
    """
    from skrobot.planner.trajectory_optimization.gridsdf_collision import make_gridsdf_self_distance_fn

    for spec in problem.residuals:
        if spec.name == 'self_collision' \
                and spec.params.get('mode') == 'gridsdf':
            return make_gridsdf_self_distance_fn(
                fk_data, spec.params['gridsdf_data'], backend,
                parameterized=parameterized)
    return None


__all__ = [
    'build_gridsdf_self_distance_fn',
    'get_problem_structure_key',
    'get_problem_value_hash',
]

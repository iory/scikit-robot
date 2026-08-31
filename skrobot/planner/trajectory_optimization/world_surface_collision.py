"""Surface-point based world-collision distances (backend-agnostic).

The default world-collision cost approximates every robot link with a few
spheres and measures the distance from those sphere centres to the obstacles.
For links that are far from spherical -- a boxy modular-robot module, say --
that is heavily over-conservative: the enclosing spheres stick out well beyond
the real geometry, so a link that physically fits through a gap reads as
colliding.

This module instead samples points on each collision link's own surface,
transforms them into the world frame with the link transforms, and evaluates
the obstacle's *exact* signed distance function at those points.  Combined
with the analytic box SDF in
:func:`~skrobot.planner.trajectory_optimization.fk_utils.compute_box_obstacle_distances`
neither side of the pair is inflated.

Build the static data once with :func:`build_world_surface_data` (NumPy), then
get a plain ``angles -> signed distances`` callable from
:func:`make_world_surface_distance_fn` and feed it to
:func:`~skrobot.planner.trajectory_optimization.fk_utils.compute_collision_residuals`,
exactly as for the sphere model.

Querying sampled surface points of the robot against a signed distance field
of the environment is the scheme used by recent GPU trajectory optimizers such
as cuRobo [1]_; the hinge residual ``max(activation_distance - d, 0)`` on top
of it is the obstacle cost of CHOMP [2]_.

References
----------
.. [1] B. Sundaralingam, S. K. S. Hari, A. Fishman, C. Garrett, K. Van Wyk,
   V. Blukis, A. Millane, H. Oleynikova, A. Handa, F. Ramos, N. Ratliff and
   D. Fox.  "cuRobo: Parallelized Collision-Free Robot Motion Generation."
   IEEE International Conference on Robotics and Automation (ICRA), 2024.
.. [2] N. Ratliff, M. Zucker, J. A. Bagnell and S. Srinivasa.
   "CHOMP: Gradient Optimization Techniques for Efficient Motion Planning."
   IEEE International Conference on Robotics and Automation (ICRA), 2009.
"""
import numpy as np

from skrobot.planner.trajectory_optimization.fk_utils import build_collision_link_transform_fn
from skrobot.planner.trajectory_optimization.fk_utils import compute_world_obstacle_distances


def build_world_surface_data(collision_link_list, n_surface=None):
    """Sample surface points on each collision link.

    Points are taken from the link's **convex hull**, not from the raw
    collision mesh.  Randomly sampling raw vertices misses the extreme
    points that actually decide a near-miss: a TYCOON module housing has
    ~19k vertices, so even 500 random samples cover 2.6% of them and the
    deepest point of a 1 cm penetration is almost never among them.  The
    hull has a few hundred vertices, contains every extreme point, and
    encloses the mesh -- so the distance it yields is conservative (never
    optimistic) for convex obstacles.

    Parameters
    ----------
    collision_link_list : list[skrobot.model.Link]
        Collision links, in the same order handed to
        :meth:`TrajectoryProblem.add_collision_cost`.
    n_surface : int, optional
        Cap on points per link.  ``None`` (default) keeps every hull
        vertex.  When given and the hull has more, a deterministic
        farthest-point subset of that size is used.

    Returns
    -------
    dict
        ``surface_points`` (n_meshed, S, 3) in link-local coordinates,
        where ``S`` is the largest per-link count (shorter links are
        padded by repeating a point, which only duplicates residuals),
        and ``link_indices`` (n_meshed,) giving each entry's position in
        ``collision_link_list``.  Links without a collision mesh carry no
        geometry to sample and are skipped.

    Raises
    ------
    ValueError
        If no collision link has a collision mesh.
    """
    per_link, kept = [], []
    for i, link in enumerate(collision_link_list):
        if link.collision_mesh is None:
            continue
        hull = np.asarray(link.collision_mesh.convex_hull.vertices,
                          dtype=np.float64)
        if n_surface is not None and len(hull) > n_surface:
            hull = _farthest_point_subset(hull, n_surface)
        per_link.append(hull)
        kept.append(i)
    if not per_link:
        raise ValueError(
            "mode='surface' needs a collision mesh on at least one "
            'collision link, but none of the {} links has one'.format(
                len(collision_link_list)))

    size = max(len(p) for p in per_link)
    padded = np.stack([
        p if len(p) == size
        else np.concatenate([p, np.repeat(p[:1], size - len(p), axis=0)])
        for p in per_link])
    return {'surface_points': padded,
            'link_indices': np.asarray(kept, dtype=np.int64)}


def _farthest_point_subset(points, k):
    """Deterministic farthest-point subset of ``points`` (k, 3).

    Greedy farthest-point sampling keeps the extremes, which is what the
    collision distance depends on; a random subset does not.
    """
    chosen = [int(np.argmax(points[:, 0]))]
    d = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return points[np.asarray(chosen)]


def world_surface_distances(link_positions, link_rotations, surface_points,
                            obstacle_arrays, backend, link_indices=None):
    """Signed distances from link surface points to the world obstacles.

    Parameters
    ----------
    link_positions : array
        Collision link origins in the world frame (n_links, 3).
    link_rotations : array
        Collision link rotations in the world frame (n_links, 3, 3).
    surface_points : array
        Link-local surface samples (n_links, n_surface, 3).
    obstacle_arrays : dict
        Output of
        :func:`~skrobot.planner.trajectory_optimization.fk_utils.prepare_world_obstacle_arrays`.
    backend : module
        Array module (``numpy`` or ``jax.numpy``).
    link_indices : array, optional
        Rows of ``link_positions`` / ``link_rotations`` that
        ``surface_points`` refers to.  Needed when some collision links
        were skipped for having no mesh.

    Returns
    -------
    array
        Signed distances (n_meshed * n_surface, n_obstacles).
        Positive = separated, negative = penetrating.
    """
    xp = backend
    if link_indices is not None:
        link_positions = link_positions[link_indices]
        link_rotations = link_rotations[link_indices]
    # (n_meshed, n_surface, 3) local -> world
    world_pts = link_positions[:, None, :] \
        + xp.einsum('lij,lsj->lsi', link_rotations, surface_points)
    flat = world_pts.reshape((-1, 3))
    # Surface points have no radius of their own: the link geometry is
    # already represented by where the points are.
    radii = xp.zeros(flat.shape[0])
    return compute_world_obstacle_distances(
        flat, radii, obstacle_arrays, xp)


def make_world_surface_distance_fn(fk_data, surface_data, obstacle_arrays,
                                   backend, parameterized=False):
    """Build the ``angles -> signed distances`` callable.

    Parameters
    ----------
    fk_data : dict
        Output of
        :func:`~skrobot.planner.trajectory_optimization.fk_utils.prepare_fk_data`.
    surface_data : dict
        Output of :func:`build_world_surface_data`.
    obstacle_arrays : dict
        Output of
        :func:`~skrobot.planner.trajectory_optimization.fk_utils.prepare_world_obstacle_arrays`.
    backend : module
        Array module (``numpy`` or ``jax.numpy``).

    Returns
    -------
    callable
        ``f(angles) -> (n_links * n_surface, n_obstacles)`` signed distances.
    """
    xp = backend
    get_link_transforms = build_collision_link_transform_fn(
        fk_data, backend, parameterized=parameterized)
    surface_points = xp.asarray(surface_data['surface_points'])
    link_indices = xp.asarray(surface_data['link_indices'])

    if parameterized:
        def distance_fn(angles, fk):
            link_pos, link_rot = get_link_transforms(angles, fk)
            return world_surface_distances(
                link_pos, link_rot, surface_points, obstacle_arrays, xp,
                link_indices=link_indices)
    else:
        def distance_fn(angles):
            link_pos, link_rot = get_link_transforms(angles)
            return world_surface_distances(
                link_pos, link_rot, surface_points, obstacle_arrays, xp,
                link_indices=link_indices)

    return distance_fn

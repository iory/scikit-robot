"""Grid-SDF based self-collision distances (backend-agnostic).

The default self-collision cost approximates every link with a few spheres.
For links whose shape is far from a sphere/capsule (e.g. the bulky Panda link
housings) that is heavily over-conservative: even a legitimate rest pose reads
as "in collision".  This module instead represents each collision link by a
:class:`skrobot.sdf.GridSDF` and measures penetration by transforming one
link's surface points into another link's SDF frame and looking the signed
distance up with (backend-agnostic) trilinear interpolation.

Build the static data once with :func:`build_gridsdf_self_data` (NumPy), then
evaluate :func:`gridsdf_self_distances` inside a numpy/jax cost with the world
transforms of the collision links.  Solvers usually want
:func:`make_gridsdf_self_distance_fn`, which wraps the forward kinematics and
returns a plain ``angles -> signed distances`` callable; the hinge is then
applied with
:func:`~skrobot.planner.trajectory_optimization.fk_utils.compute_collision_residuals`,
as for the sphere model.

That hinge residual ``max(activation_distance - d, 0)`` on a precomputed
signed distance field follows the obstacle cost of CHOMP and the collision
penalty of TrajOpt; representing each robot link by its own signed distance
field (rather than by spheres) and querying sampled surface points against it
is the scheme used by recent GPU trajectory optimizers such as cuRobo.

References
----------
.. [1] N. Ratliff, M. Zucker, J. A. Bagnell and S. Srinivasa.
   "CHOMP: Gradient Optimization Techniques for Efficient Motion Planning."
   IEEE International Conference on Robotics and Automation (ICRA), 2009.
.. [2] J. Schulman, Y. Duan, J. Ho, A. Lee, I. Awwal, H. Bradlow, J. Pan,
   S. Patil, K. Goldberg and P. Abbeel.
   "Motion Planning with Sequential Convex Optimization and Convex Collision
   Checking." International Journal of Robotics Research, 33(9), 2014.
.. [3] B. Sundaralingam, S. K. S. Hari, A. Fishman, C. Garrett, K. Van Wyk,
   V. Blukis, A. Millane, H. Oleynikova, A. Handa, F. Ramos, N. Ratliff and
   D. Fox.  "cuRobo: Parallelized Collision-Free Robot Motion Generation."
   IEEE International Conference on Robotics and Automation (ICRA), 2024.
"""
import numpy as np

from skrobot.planner.trajectory_optimization.collision import create_self_collision_pairs


def build_gridsdf_self_data(robot_model, collision_link_list,
                            dim_grid=40, n_surface=48):
    """Precompute per-collision-link GridSDFs and surface samples.

    Parameters
    ----------
    robot_model : skrobot.model.RobotModel
        Robot model.  Unused except for documentation symmetry with the other
        collision helpers; the geometry is taken from the links themselves.
    collision_link_list : list[skrobot.model.Link]
        Collision links.  Must be the same list handed to
        :meth:`TrajectoryProblem.add_collision_cost`, i.e. ordered along the
        kinematic chain, because link pairs are selected by
        :func:`~skrobot.planner.trajectory_optimization.collision.create_self_collision_pairs`,
        which skips neighbours in this list as "adjacent".
    dim_grid : int
        GridSDF resolution per axis.  Voxelization runs once per (mesh,
        resolution) pair and is then served from
        :func:`skrobot.data.get_cache_dir`, so only the first build pays for
        it.
    n_surface : int
        Number of surface sample points kept per link.

    Returns
    -------
    dict
        Static arrays consumed by :func:`gridsdf_self_distances`:
        ``grids`` (n, Dx, Dy, Dz), ``origins`` (n, 3), ``resolutions`` (n,),
        ``dims`` (n, 3), ``surface_points`` (n, n_surface, 3) in link-local
        coordinates, and the ordered link pairs ``pairs_a`` / ``pairs_b``.

    Raises
    ------
    ValueError
        If any collision link has no collision mesh.
    """
    from skrobot.sdf.signed_distance_function import trimesh2sdf

    missing = [link.name for link in collision_link_list
               if link.collision_mesh is None]
    if missing:
        raise ValueError(
            "mode='gridsdf' needs a collision mesh on every collision link, "
            "but these have none: {}".format(missing))

    grids, origins, resolutions, dims, surfs = [], [], [], [], []
    rng = np.random.RandomState(0)
    for link in collision_link_list:
        # Always build a *voxel* GridSDF, even for links whose collision mesh
        # carries primitive metadata (box/cylinder/sphere): trimesh2sdf would
        # otherwise return an analytic primitive SDF, which has no grid to
        # look up and is placed in its own primitive frame.  Dropping the
        # metadata keeps every link's field in the link frame, which is the
        # frame the sampled surface points live in as well.
        mesh = link.collision_mesh.copy()
        if 'shape' in mesh.metadata:
            mesh.metadata = {k: v for k, v in mesh.metadata.items()
                             if k != 'shape'}
        sdf = trimesh2sdf(mesh, dim_grid=dim_grid)
        data = np.asarray(sdf._data, dtype=np.float64)
        grids.append(data)
        origins.append(np.asarray(sdf.origin, dtype=np.float64))
        resolutions.append(float(sdf._resolution))
        dims.append(np.array(data.shape, dtype=np.int64))
        verts = np.asarray(link.collision_mesh.vertices, dtype=np.float64)
        idx = rng.choice(len(verts), size=n_surface,
                         replace=len(verts) < n_surface)
        surfs.append(verts[idx])

    dims = np.stack(dims)
    # Grids differ in shape per link; pad to a common shape so they can be
    # stacked and gathered in one batched lookup.  Padding is never weighted
    # (queries are clamped to ``dims - 1``), 'edge' just keeps it finite.
    maxd = dims.max(axis=0)
    padded = np.stack([
        np.pad(g, [(0, int(maxd[k]) - g.shape[k]) for k in range(3)],
               mode='edge')
        for g in grids])

    # Ordered pairs: the lookup is asymmetric (surface points of A against the
    # field of B), so each unordered pair is evaluated in both directions.
    link_pairs = create_self_collision_pairs(collision_link_list)
    pa = [a for a, b in link_pairs] + [b for a, b in link_pairs]
    pb = [b for a, b in link_pairs] + [a for a, b in link_pairs]

    return {
        'grids': padded,                              # (n, Dx, Dy, Dz)
        'origins': np.stack(origins),                 # (n, 3)
        'resolutions': np.array(resolutions),         # (n,)
        'dims': dims.astype(np.float64),              # (n, 3)
        'surface_points': np.stack(surfs),            # (n, S, 3) link-local
        'pairs_a': np.array(pa, dtype=np.int64),
        'pairs_b': np.array(pb, dtype=np.int64),
    }


def _trilinear(grid_batch, grid_coords, dims, resolutions, backend):
    """Batched trilinear lookup into a stack of grids.

    Parameters
    ----------
    grid_batch : array
        Stacked signed distance grids (n_pairs, Dx, Dy, Dz).
    grid_coords : array
        Query points in fractional grid coordinates (n_pairs, n_points, 3).
    dims : array
        Valid (unpadded) extent of each grid (n_pairs, 3).
    resolutions : array
        Grid resolution [m] of each grid (n_pairs,).
    backend : module
        ``numpy`` or ``jax.numpy``.

    Returns
    -------
    array
        Signed distances (n_pairs, n_points).  Outside the grid the clamped
        edge value plus the Euclidean distance from the query point to the
        grid boundary is returned, so the field stays finite and keeps a
        usable gradient pointing back toward the surface (a constant fill
        value would make the cost flat, and hence gradient-free, exactly where
        links are far apart).
    """
    xp = backend
    shape = grid_batch.shape[1:]

    # Clamp query coordinates into the valid cell range so the interpolation
    # returns a true edge value; the outside offset is added separately.
    hi = dims[:, None, :] - 1.0
    clamped = xp.clip(grid_coords, 0.0, hi)
    lo = xp.floor(clamped)
    frac = clamped - lo
    lo = lo.astype(xp.int32)

    pair_idx = xp.arange(grid_batch.shape[0])[:, None]

    def gather(dx, dy, dz):
        ix = xp.clip(lo[..., 0] + dx, 0, shape[0] - 1)
        iy = xp.clip(lo[..., 1] + dy, 0, shape[1] - 1)
        iz = xp.clip(lo[..., 2] + dz, 0, shape[2] - 1)
        return grid_batch[pair_idx, ix, iy, iz]

    fx, fy, fz = frac[..., 0], frac[..., 1], frac[..., 2]
    c00 = gather(0, 0, 0) * (1 - fx) + gather(1, 0, 0) * fx
    c10 = gather(0, 1, 0) * (1 - fx) + gather(1, 1, 0) * fx
    c01 = gather(0, 0, 1) * (1 - fx) + gather(1, 0, 1) * fx
    c11 = gather(0, 1, 1) * (1 - fx) + gather(1, 1, 1) * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    value = c0 * (1 - fz) + c1 * fz

    # Distance beyond the grid boundary (exactly 0 inside), in metres.  The
    # doubled ``where`` keeps the square root away from 0, where its derivative
    # is undefined and jax would propagate NaN.
    squared = xp.sum((grid_coords - clamped) ** 2, axis=-1)
    outside = squared > 0.0
    beyond = xp.where(
        outside, xp.sqrt(xp.where(outside, squared, 1.0)), 0.0)
    return value + beyond * resolutions[:, None]


def gridsdf_self_distances(link_positions, link_rotations, data, backend):
    """Signed distances of every (link pair, surface point) for one pose.

    Parameters
    ----------
    link_positions : array
        World positions of the collision links (n_links, 3).
    link_rotations : array
        World rotations of the collision links (n_links, 3, 3).
    data : dict
        Output of :func:`build_gridsdf_self_data`, with the arrays already
        moved onto ``backend``.
    backend : module
        ``numpy`` or ``jax.numpy``.

    Returns
    -------
    array
        Signed distances (n_pairs, n_surface); negative means penetration.
    """
    xp = backend
    a = data['pairs_a']
    b = data['pairs_b']
    surface_a = data['surface_points'][a]                 # (P, S, 3), local A
    rot_a, pos_a = link_rotations[a], link_positions[a]
    rot_b, pos_b = link_rotations[b], link_positions[b]

    # Surface points of A: local A -> world -> local B.
    world = pos_a[:, None, :] + xp.einsum('pij,psj->psi', rot_a, surface_a)
    local_b = xp.einsum('pij,psj->psi', xp.swapaxes(rot_b, 1, 2),
                        world - pos_b[:, None, :])
    grid_coords = (local_b - data['origins'][b][:, None, :]) \
        / data['resolutions'][b][:, None, None]
    return _trilinear(data['grids'][b], grid_coords, data['dims'][b],
                      data['resolutions'][b], xp)


def make_gridsdf_self_distance_fn(fk_data, gridsdf_data, backend):
    """Build ``angles -> signed distances`` for a single configuration.

    Moves the static GridSDF data onto ``backend`` once and closes over the
    forward kinematics, so solvers can simply map the returned callable over
    the waypoints of a trajectory.

    Parameters
    ----------
    fk_data : dict
        Output of
        :func:`~skrobot.planner.trajectory_optimization.fk_utils.prepare_fk_data`.
        Must contain the collision link entries, i.e.
        :meth:`TrajectoryProblem.add_collision_cost` must have been called.
    gridsdf_data : dict
        Output of :func:`build_gridsdf_self_data` (NumPy arrays).
    backend : module
        ``numpy`` or ``jax.numpy``.

    Returns
    -------
    callable
        ``f(angles) -> (n_pairs, n_surface)`` signed distances.
    """
    from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions

    xp = backend
    data = {key: xp.asarray(value) for key, value in gridsdf_data.items()}
    chain_idx = xp.asarray(fk_data['collision_link_to_chain_idx'])
    offsets_pos = xp.asarray(fk_data['collision_link_offsets_pos'])
    offsets_rot = xp.asarray(fk_data['collision_link_offsets_rot'])
    get_link_transforms = build_fk_functions(fk_data, xp)[0]

    def gridsdf_self_distances_from_angles(angles):
        link_pos, link_rot = get_link_transforms(angles)
        chain_pos = link_pos[chain_idx]
        chain_rot = link_rot[chain_idx]
        world_pos = chain_pos + xp.einsum('cij,cj->ci', chain_rot, offsets_pos)
        world_rot = xp.einsum('cij,cjk->cik', chain_rot, offsets_rot)
        return gridsdf_self_distances(world_pos, world_rot, data, xp)

    return gridsdf_self_distances_from_angles


__all__ = [
    'build_gridsdf_self_data',
    'gridsdf_self_distances',
    'make_gridsdf_self_distance_fn',
]

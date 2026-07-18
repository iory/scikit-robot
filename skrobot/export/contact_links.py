import numpy as np


def contact_links(robot, trajectory, targets, threshold=0.05,
                  max_vertices=500):
    """Names of links whose geometry comes within ``threshold`` of a target.

    Runs a joint trajectory kinematically (no dynamics) and reports which links
    actually approach the objects the robot interacts with. Use it to pick
    ``decompose_links`` for :func:`skrobot.export.usd.urdf_to_usd` from the task
    itself: only the links that come near the manipulated object need accurate
    (decomposed) colliders, while the rest keep the cheap single convex hull.
    This is much cheaper than decomposing every concave link when only the
    gripper touches anything.

    Parameters
    ----------
    robot : skrobot.model.RobotModel
        Robot whose links are tested. Its configuration is changed during the
        call (set to each trajectory sample).
    trajectory : iterable of array_like
        Joint configurations, each a full angle-vector applied with
        ``robot.angle_vector``.
    targets : array_like
        World-frame points the robot interacts with, shape (3,) or (M, 3)
        (e.g. object centres).
    threshold : float
        A link counts if any of its collision-mesh vertices comes within this
        many metres of any target, over the whole trajectory.
    max_vertices : int
        Per-link vertex budget; dense meshes are uniformly subsampled to keep
        the sweep fast. The min distance is only weakly sensitive to it.

    Returns
    -------
    list of str
        Link names within ``threshold`` of a target, ordered by closest
        approach (nearest first).
    """
    tgt = np.atleast_2d(np.asarray(targets, dtype=float))
    if tgt.shape[1] != 3:
        raise ValueError('targets must be (3,) or (M, 3)')

    # cache each link's local collision vertices (subsampled) once
    local = {}
    for link in robot.link_list:
        mesh = getattr(link, 'collision_mesh', None)
        if mesh is None or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            continue
        v = np.asarray(mesh.vertices, dtype=float)
        if len(v) > max_vertices:
            idx = np.linspace(0, len(v) - 1, max_vertices).astype(int)
            v = v[idx]
        local[link.name] = v

    best = {}
    for cfg in trajectory:
        robot.angle_vector(np.asarray(cfg, dtype=float))
        for name, v in local.items():
            link = getattr(robot, name)
            T = np.asarray(link.worldcoords().T(), dtype=float)
            world = v @ T[:3, :3].T + T[:3, 3]
            # min distance from any link vertex to any target
            d = float(np.min(np.linalg.norm(
                world[:, None, :] - tgt[None, :, :], axis=2)))
            if d < best.get(name, np.inf):
                best[name] = d

    hit = sorted((d, n) for n, d in best.items() if d <= threshold)
    return [n for _, n in hit]

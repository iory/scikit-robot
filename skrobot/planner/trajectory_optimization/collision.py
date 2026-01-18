"""Collision utilities for trajectory optimization."""

import numpy as np


def extract_collision_spheres(robot_model, link_list, n_spheres_per_link=3):
    """Extract collision spheres for robot links.

    Approximates each link with spheres along its bounding capsule.

    Parameters
    ----------
    robot_model : RobotModel
        Robot model.
    link_list : list
        List of links in kinematic chain.
    n_spheres_per_link : int
        Number of spheres per link.

    Returns
    -------
    dict
        Dictionary containing:
        - 'sphere_centers_local': Local sphere centers per link (n_spheres, 3)
        - 'sphere_radii': Sphere radii (n_spheres,)
        - 'link_indices': Link index for each sphere (n_spheres,)
    """
    try:
        import trimesh
    except ImportError:
        trimesh = None

    sphere_centers = []
    sphere_radii = []
    link_indices = []

    for link_idx, link in enumerate(link_list):
        # Get collision mesh if available
        mesh = getattr(link, 'collision_mesh', None)
        if trimesh is not None and mesh is not None:
            if isinstance(mesh, trimesh.Trimesh) and not mesh.is_empty:
                # Compute bounding capsule
                try:
                    result = trimesh.bounds.minimum_cylinder(mesh)
                    height = result['height']
                    radius = result['radius']
                    transform = result['transform']

                    # Generate sphere centers along capsule axis
                    t_values = np.linspace(-0.5, 0.5, n_spheres_per_link)
                    for t in t_values:
                        local_pos = np.array([0, 0, t * height])
                        world_pos = (transform[:3, :3] @ local_pos
                                     + transform[:3, 3])
                        sphere_centers.append(world_pos)
                        sphere_radii.append(radius)
                        link_indices.append(link_idx)
                    continue
                except Exception:
                    pass

        # Fallback: single sphere at link origin
        sphere_centers.append(np.zeros(3))
        sphere_radii.append(0.05)  # Default radius
        link_indices.append(link_idx)

    return {
        'sphere_centers_local': np.array(sphere_centers),
        'sphere_radii': np.array(sphere_radii),
        'link_indices': np.array(link_indices),
    }


def create_self_collision_pairs(link_list, ignore_adjacent=True):
    """Create pairs of links for self-collision checking.

    Parameters
    ----------
    link_list : list
        List of links.
    ignore_adjacent : bool
        If True, ignore collisions between adjacent (parent-child) links.

    Returns
    -------
    list of tuple
        List of (i, j) pairs of link indices to check.
    """
    n_links = len(link_list)
    pairs = []

    for i in range(n_links):
        for j in range(i + 2, n_links):  # Skip i and i+1 (adjacent)
            pairs.append((i, j))

    return pairs

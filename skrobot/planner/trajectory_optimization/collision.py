"""Collision utilities for trajectory optimization."""

import numpy as np


def point_to_sphere_distance(point, sphere_center, sphere_radius):
    """Compute signed distance from a point to a sphere surface.

    Parameters
    ----------
    point : array-like
        Query point (3,).
    sphere_center : array-like
        Sphere center (3,).
    sphere_radius : float
        Sphere radius.

    Returns
    -------
    float
        Signed distance (negative if inside sphere).
    """
    point = np.asarray(point)
    sphere_center = np.asarray(sphere_center)
    return np.linalg.norm(point - sphere_center) - sphere_radius


def point_to_box_distance(point, box_center, box_rotation, box_half_extents):
    """Compute distance from a point to a box surface.

    Parameters
    ----------
    point : array-like
        Query point in world frame (3,).
    box_center : array-like
        Box center in world frame (3,).
    box_rotation : array-like
        Box rotation matrix (3, 3).
    box_half_extents : array-like
        Half extents of the box (3,).

    Returns
    -------
    float
        Distance to box surface (0 if inside).
    """
    point = np.asarray(point)
    box_center = np.asarray(box_center)
    box_rotation = np.asarray(box_rotation)
    box_half_extents = np.asarray(box_half_extents)

    # Transform point to box local frame
    local_point = box_rotation.T @ (point - box_center)

    # Compute closest point on box surface
    closest = np.clip(local_point, -box_half_extents, box_half_extents)
    return np.linalg.norm(local_point - closest)


def point_to_cylinder_distance(point, cyl_center, cyl_rotation,
                               cyl_radius, cyl_half_height):
    """Compute distance from a point to a cylinder surface.

    The cylinder axis is along the local Z axis.

    Parameters
    ----------
    point : array-like
        Query point in world frame (3,).
    cyl_center : array-like
        Cylinder center in world frame (3,).
    cyl_rotation : array-like
        Cylinder rotation matrix (3, 3).
    cyl_radius : float
        Cylinder radius.
    cyl_half_height : float
        Half height of the cylinder.

    Returns
    -------
    float
        Distance to cylinder surface (0 if inside).
    """
    point = np.asarray(point)
    cyl_center = np.asarray(cyl_center)
    cyl_rotation = np.asarray(cyl_rotation)

    # Transform point to cylinder local frame
    local_point = cyl_rotation.T @ (point - cyl_center)

    # Distance to cylinder axis (in XY plane)
    xy_dist = np.sqrt(local_point[0]**2 + local_point[1]**2)
    z_abs = abs(local_point[2])

    if xy_dist <= cyl_radius:
        # Inside cylinder radius
        if z_abs <= cyl_half_height:
            # Inside cylinder - return 0
            return 0.0
        else:
            # Above or below cylinder
            return z_abs - cyl_half_height
    else:
        # Outside cylinder radius
        if z_abs <= cyl_half_height:
            # Side of cylinder
            return xy_dist - cyl_radius
        else:
            # Edge of cylinder (corner distance)
            return np.sqrt(
                (xy_dist - cyl_radius)**2
                + (z_abs - cyl_half_height)**2
            )


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

"""JAX-based collision detection for batch IK solving.

This module provides differentiable collision detection primitives
that can be used with JAX autodiff for optimization.
"""

import os
import platform

import numpy as np


def _ensure_jax_cpu_on_mac():
    """Ensure JAX uses CPU backend on macOS."""
    if platform.system() == 'Darwin':
        if 'JAX_PLATFORMS' not in os.environ:
            os.environ['JAX_PLATFORMS'] = 'cpu'


# Ensure CPU on Mac before any JAX imports
_ensure_jax_cpu_on_mac()


def sphere_sphere_distance(center1, radius1, center2, radius2):
    """Compute signed distance between two spheres.

    Parameters
    ----------
    center1 : jax.numpy.ndarray
        Center of first sphere (..., 3).
    radius1 : float or jax.numpy.ndarray
        Radius of first sphere.
    center2 : jax.numpy.ndarray
        Center of second sphere (..., 3).
    radius2 : float or jax.numpy.ndarray
        Radius of second sphere.

    Returns
    -------
    jax.numpy.ndarray
        Signed distance. Positive = separation, negative = penetration.
    """
    _ensure_jax_cpu_on_mac()
    import jax.numpy as jnp

    diff = center1 - center2
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)
    return dist - radius1 - radius2


def capsule_capsule_distance(p1, q1, r1, p2, q2, r2):
    """Compute signed distance between two capsules.

    A capsule is defined by a line segment (p to q) and a radius.

    Parameters
    ----------
    p1, q1 : jax.numpy.ndarray
        Endpoints of first capsule segment (..., 3).
    r1 : float or jax.numpy.ndarray
        Radius of first capsule.
    p2, q2 : jax.numpy.ndarray
        Endpoints of second capsule segment (..., 3).
    r2 : float or jax.numpy.ndarray
        Radius of second capsule.

    Returns
    -------
    jax.numpy.ndarray
        Signed distance between capsules.
    """
    _ensure_jax_cpu_on_mac()
    import jax.numpy as jnp

    # Line segment to line segment distance
    d1 = q1 - p1  # Direction of segment 1
    d2 = q2 - p2  # Direction of segment 2
    r = p1 - p2

    a = jnp.sum(d1 * d1, axis=-1) + 1e-10  # |d1|^2
    e = jnp.sum(d2 * d2, axis=-1) + 1e-10  # |d2|^2
    f = jnp.sum(d2 * r, axis=-1)

    b = jnp.sum(d1 * d2, axis=-1)
    c = jnp.sum(d1 * r, axis=-1)

    denom = a * e - b * b + 1e-10

    # Compute s and t parameters
    s = jnp.clip((b * f - c * e) / denom, 0.0, 1.0)
    t = jnp.clip((b * s + f) / e, 0.0, 1.0)

    # Recompute s based on clamped t
    s = jnp.clip((b * t - c) / a, 0.0, 1.0)

    # Closest points on segments
    closest1 = p1 + s[..., None] * d1
    closest2 = p2 + t[..., None] * d2

    # Distance between closest points
    diff = closest1 - closest2
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)

    return dist - r1 - r2


def sphere_halfspace_distance(center, radius, point, normal):
    """Compute signed distance from sphere to halfspace (plane).

    Parameters
    ----------
    center : jax.numpy.ndarray
        Sphere center (..., 3).
    radius : float or jax.numpy.ndarray
        Sphere radius.
    point : jax.numpy.ndarray
        Point on the plane (..., 3).
    normal : jax.numpy.ndarray
        Outward normal of the plane (..., 3).

    Returns
    -------
    jax.numpy.ndarray
        Signed distance. Positive = sphere above plane.
    """
    _ensure_jax_cpu_on_mac()
    import jax.numpy as jnp

    # Distance from sphere center to plane
    to_center = center - point
    plane_dist = jnp.sum(to_center * normal, axis=-1)

    return plane_dist - radius


def capsule_halfspace_distance(p1, p2, radius, point, normal):
    """Compute signed distance from capsule to halfspace (plane).

    A capsule is defined by two endpoints and a radius.

    Parameters
    ----------
    p1 : jax.numpy.ndarray
        First endpoint of capsule axis (..., 3).
    p2 : jax.numpy.ndarray
        Second endpoint of capsule axis (..., 3).
    radius : float or jax.numpy.ndarray
        Capsule radius.
    point : jax.numpy.ndarray
        Point on the plane (..., 3).
    normal : jax.numpy.ndarray
        Outward normal of the plane (..., 3).

    Returns
    -------
    jax.numpy.ndarray
        Signed distance. Positive = capsule above plane.
    """
    _ensure_jax_cpu_on_mac()
    import jax.numpy as jnp

    # Distance from each endpoint (as sphere) to plane
    dist1 = sphere_halfspace_distance(p1, radius, point, normal)
    dist2 = sphere_halfspace_distance(p2, radius, point, normal)

    return jnp.minimum(dist1, dist2)


def sphere_capsule_distance(sphere_center, sphere_radius, p1, p2, capsule_radius):
    """Compute signed distance between sphere and capsule.

    Parameters
    ----------
    sphere_center : jax.numpy.ndarray
        Sphere center (..., 3).
    sphere_radius : float or jax.numpy.ndarray
        Sphere radius.
    p1 : jax.numpy.ndarray
        First endpoint of capsule axis (..., 3).
    p2 : jax.numpy.ndarray
        Second endpoint of capsule axis (..., 3).
    capsule_radius : float or jax.numpy.ndarray
        Capsule radius.

    Returns
    -------
    jax.numpy.ndarray
        Signed distance. Positive = separated, negative = penetrating.
    """
    _ensure_jax_cpu_on_mac()
    import jax.numpy as jnp

    # Find closest point on capsule segment to sphere center
    segment = p2 - p1
    segment_len_sq = jnp.sum(segment ** 2, axis=-1, keepdims=True) + 1e-10

    # Project sphere center onto segment
    t = jnp.sum((sphere_center - p1) * segment, axis=-1, keepdims=True) / segment_len_sq
    t = jnp.clip(t, 0.0, 1.0)

    # Closest point on segment
    closest = p1 + t * segment

    # Distance from sphere center to closest point on capsule axis
    diff = sphere_center - closest
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)

    return dist - sphere_radius - capsule_radius


def sphere_box_distance(sphere_center, sphere_radius, box_center, box_half_extents,
                         box_rotation=None):
    """Compute signed distance between sphere and box.

    Parameters
    ----------
    sphere_center : jax.numpy.ndarray
        Sphere center (..., 3).
    sphere_radius : float or jax.numpy.ndarray
        Sphere radius.
    box_center : jax.numpy.ndarray
        Box center (..., 3).
    box_half_extents : jax.numpy.ndarray
        Box half-extents (half width, half height, half depth) (..., 3).
    box_rotation : jax.numpy.ndarray, optional
        Box rotation matrix (..., 3, 3). If None, box is axis-aligned.

    Returns
    -------
    jax.numpy.ndarray
        Signed distance. Positive = separated, negative = penetrating.
    """
    _ensure_jax_cpu_on_mac()
    import jax.numpy as jnp

    # Transform sphere center to box's local frame
    local_center = sphere_center - box_center
    if box_rotation is not None:
        # Apply inverse rotation (transpose for rotation matrix)
        local_center = jnp.einsum('...ij,...j->...i',
                                  jnp.swapaxes(box_rotation, -1, -2),
                                  local_center)

    # Clamp to box bounds to find closest point on box surface
    closest = jnp.clip(local_center, -box_half_extents, box_half_extents)

    # Distance from sphere center to closest point
    diff = local_center - closest
    dist_to_surface = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)

    return dist_to_surface - sphere_radius


def capsule_box_distance(p1, p2, capsule_radius, box_center, box_half_extents,
                          box_rotation=None):
    """Compute signed distance between capsule and box.

    Uses iterative closest point refinement.

    Parameters
    ----------
    p1 : jax.numpy.ndarray
        First endpoint of capsule axis (..., 3).
    p2 : jax.numpy.ndarray
        Second endpoint of capsule axis (..., 3).
    capsule_radius : float or jax.numpy.ndarray
        Capsule radius.
    box_center : jax.numpy.ndarray
        Box center (..., 3).
    box_half_extents : jax.numpy.ndarray
        Box half-extents (..., 3).
    box_rotation : jax.numpy.ndarray, optional
        Box rotation matrix (..., 3, 3). If None, box is axis-aligned.

    Returns
    -------
    jax.numpy.ndarray
        Signed distance. Positive = separated, negative = penetrating.
    """
    _ensure_jax_cpu_on_mac()
    import jax.numpy as jnp

    # Transform capsule endpoints to box's local frame
    p1_local = p1 - box_center
    p2_local = p2 - box_center
    if box_rotation is not None:
        rot_inv = jnp.swapaxes(box_rotation, -1, -2)
        p1_local = jnp.einsum('...ij,...j->...i', rot_inv, p1_local)
        p2_local = jnp.einsum('...ij,...j->...i', rot_inv, p2_local)

    # Helper: find closest point on segment to target point
    def closest_segment_point(seg_p1, seg_p2, target):
        segment = seg_p2 - seg_p1
        seg_len_sq = jnp.sum(segment ** 2, axis=-1, keepdims=True) + 1e-10
        t = jnp.sum((target - seg_p1) * segment, axis=-1, keepdims=True) / seg_len_sq
        t = jnp.clip(t, 0.0, 1.0)
        return seg_p1 + t * segment

    # Iteration 1: closest point on segment to box center (origin)
    origin = jnp.zeros_like(p1_local)
    pt_seg = closest_segment_point(p1_local, p2_local, origin)
    pt_box = jnp.clip(pt_seg, -box_half_extents, box_half_extents)

    # Iteration 2: refine with closest point on segment to pt_box
    pt_seg = closest_segment_point(p1_local, p2_local, pt_box)

    # Compute SDF from segment point to box
    # Clamp to find closest point on box surface
    closest = jnp.clip(pt_seg, -box_half_extents, box_half_extents)
    diff = pt_seg - closest
    dist_to_surface = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)

    # For inside case: compute penetration depth as distance to nearest face
    face_dists = box_half_extents - jnp.abs(pt_seg)
    min_face_dist = jnp.min(face_dists, axis=-1)

    # If inside (dist ~= 0), return negative penetration
    is_inside = dist_to_surface < 1e-5
    sdf = jnp.where(is_inside, -min_face_dist, dist_to_surface)

    return sdf - capsule_radius


def box_halfspace_distance(box_center, box_half_extents, box_rotation,
                            point, normal):
    """Compute signed distance from box to halfspace (plane).

    Parameters
    ----------
    box_center : jax.numpy.ndarray
        Box center (..., 3).
    box_half_extents : jax.numpy.ndarray
        Box half-extents (..., 3).
    box_rotation : jax.numpy.ndarray
        Box rotation matrix (..., 3, 3). Can be identity for axis-aligned box.
    point : jax.numpy.ndarray
        Point on the plane (..., 3).
    normal : jax.numpy.ndarray
        Outward normal of the plane (..., 3).

    Returns
    -------
    jax.numpy.ndarray
        Signed distance. Positive = box above plane.
    """
    _ensure_jax_cpu_on_mac()
    import jax.numpy as jnp

    # Get 8 corners of the box
    # Signs for the 8 corners
    signs = jnp.array([
        [-1, -1, -1],
        [-1, -1, +1],
        [-1, +1, -1],
        [-1, +1, +1],
        [+1, -1, -1],
        [+1, -1, +1],
        [+1, +1, -1],
        [+1, +1, +1],
    ])  # (8, 3)

    # Corners in local frame
    corners_local = box_half_extents[..., None, :] * signs  # (..., 8, 3)

    # Transform to world frame
    corners_world = jnp.einsum('...ij,...kj->...ki', box_rotation, corners_local)
    corners_world = corners_world + box_center[..., None, :]  # (..., 8, 3)

    # Compute distance from each corner to plane
    to_corners = corners_world - point[..., None, :]  # (..., 8, 3)
    corner_dists = jnp.sum(to_corners * normal[..., None, :], axis=-1)  # (..., 8)

    # Return minimum distance (most penetrating corner)
    return jnp.min(corner_dists, axis=-1)


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
        - 'sphere_centers_local': Local sphere centers per link (n_links, n_spheres, 3)
        - 'sphere_radii': Sphere radii per link (n_links, n_spheres)
        - 'link_indices': Link index for each sphere
    """
    import trimesh

    sphere_centers = []
    sphere_radii = []
    link_indices = []

    for link_idx, link in enumerate(link_list):
        # Get collision mesh if available
        if hasattr(link, 'collision_mesh') and link.collision_mesh is not None:
            mesh = link.collision_mesh
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
                        world_pos = transform[:3, :3] @ local_pos + transform[:3, 3]
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


def create_collision_cost_fn(fk_params, collision_spheres, self_collision_pairs,
                             activation_distance=0.02):
    """Create a JIT-compiled self-collision cost function.

    Parameters
    ----------
    fk_params : dict
        FK parameters from extract_fk_parameters.
    collision_spheres : dict
        Collision sphere parameters from extract_collision_spheres.
    self_collision_pairs : list
        List of (i, j) link index pairs to check.
    activation_distance : float
        Distance threshold below which cost activates.

    Returns
    -------
    callable
        Collision cost function: cost_fn(angles) -> cost
    """
    _ensure_jax_cpu_on_mac()
    import jax
    import jax.numpy as jnp

    # Convert to JAX arrays
    sphere_centers_local = jnp.array(collision_spheres['sphere_centers_local'])
    sphere_radii = jnp.array(collision_spheres['sphere_radii'])
    link_indices = jnp.array(collision_spheres['link_indices'])

    n_spheres = len(sphere_radii)

    # Get link transforms for sphere computation
    link_translations = jnp.array(fk_params['link_translations'])
    link_rotations = jnp.array(fk_params['link_rotations'])
    joint_axes = jnp.array(fk_params['joint_axes'])
    base_position = jnp.array(fk_params['base_position'])
    base_rotation = jnp.array(fk_params['base_rotation'])
    n_joints = fk_params['n_joints']

    def rotation_matrix_axis_angle(axis, theta):
        axis = axis / jnp.sqrt(jnp.dot(axis, axis) + 1e-10)
        K = jnp.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        I = jnp.eye(3)
        return I + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)

    def get_link_transforms(angles):
        """Compute world transforms for each link."""
        transforms = []
        current_pos = base_position
        current_rot = base_rotation

        for i in range(n_joints):
            link_trans = link_translations[i]
            link_rot = link_rotations[i]

            current_pos = current_pos + current_rot @ link_trans
            current_rot = current_rot @ link_rot

            joint_rot = rotation_matrix_axis_angle(joint_axes[i], angles[i])
            current_rot = current_rot @ joint_rot

            transforms.append((current_pos.copy(), current_rot.copy()))

        return transforms

    def get_sphere_positions(angles):
        """Compute world positions of all collision spheres."""
        link_transforms = get_link_transforms(angles)

        sphere_positions = []
        for sphere_idx in range(n_spheres):
            link_idx = link_indices[sphere_idx]
            link_pos, link_rot = link_transforms[link_idx]
            local_center = sphere_centers_local[sphere_idx]
            world_center = link_pos + link_rot @ local_center
            sphere_positions.append(world_center)

        return jnp.stack(sphere_positions, axis=0)

    def collision_cost(angles):
        """Compute self-collision cost."""
        sphere_positions = get_sphere_positions(angles)

        total_cost = 0.0
        for i, j in self_collision_pairs:
            for si in range(n_spheres):
                if link_indices[si] != i:
                    continue
                for sj in range(n_spheres):
                    if link_indices[sj] != j:
                        continue

                    dist = sphere_sphere_distance(
                        sphere_positions[si], sphere_radii[si],
                        sphere_positions[sj], sphere_radii[sj]
                    )

                    # Smooth activation: cost increases as distance decreases
                    cost = jnp.maximum(0.0, activation_distance - dist)
                    total_cost = total_cost + cost ** 2

        return total_cost

    return jax.jit(collision_cost)


def compute_self_collision_distances(fk_fn, angles, link_list, n_spheres_per_link=3):
    """Compute self-collision distances for a robot configuration.

    Parameters
    ----------
    fk_fn : callable
        Forward kinematics function.
    angles : jax.numpy.ndarray
        Joint angles.
    link_list : list
        List of links.
    n_spheres_per_link : int
        Number of spheres per link.

    Returns
    -------
    jax.numpy.ndarray
        Minimum distance for each collision pair.
    """
    _ensure_jax_cpu_on_mac()
    import jax.numpy as jnp

    # This is a simplified version - full implementation would compute
    # sphere positions at each link and check all pairs
    # For now, return empty array if no collision checking needed
    return jnp.array([])


def create_world_collision_cost_fn(fk_params, collision_spheres, obstacles):
    """Create a world collision cost function.

    Parameters
    ----------
    fk_params : dict
        FK parameters.
    collision_spheres : dict
        Collision sphere parameters.
    obstacles : list of dict
        List of obstacles. Each obstacle is a dict with:
        - 'type': 'sphere', 'box', or 'halfspace'
        - For sphere: 'center', 'radius'
        - For halfspace: 'point', 'normal'

    Returns
    -------
    callable
        World collision cost function.
    """
    _ensure_jax_cpu_on_mac()
    import jax
    import jax.numpy as jnp

    sphere_centers_local = jnp.array(collision_spheres['sphere_centers_local'])
    sphere_radii = jnp.array(collision_spheres['sphere_radii'])
    link_indices = jnp.array(collision_spheres['link_indices'])
    n_spheres = len(sphere_radii)

    link_translations = jnp.array(fk_params['link_translations'])
    link_rotations = jnp.array(fk_params['link_rotations'])
    joint_axes = jnp.array(fk_params['joint_axes'])
    base_position = jnp.array(fk_params['base_position'])
    base_rotation = jnp.array(fk_params['base_rotation'])
    n_joints = fk_params['n_joints']

    # Parse obstacles
    sphere_obstacles = []
    halfspace_obstacles = []
    for obs in obstacles:
        if obs['type'] == 'sphere':
            sphere_obstacles.append({
                'center': jnp.array(obs['center']),
                'radius': obs['radius']
            })
        elif obs['type'] == 'halfspace':
            sphere_obstacles.append({
                'point': jnp.array(obs['point']),
                'normal': jnp.array(obs['normal'])
            })

    def rotation_matrix_axis_angle(axis, theta):
        axis = axis / jnp.sqrt(jnp.dot(axis, axis) + 1e-10)
        K = jnp.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        I = jnp.eye(3)
        return I + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)

    def get_link_transforms(angles):
        transforms = []
        current_pos = base_position
        current_rot = base_rotation

        for i in range(n_joints):
            link_trans = link_translations[i]
            link_rot = link_rotations[i]

            current_pos = current_pos + current_rot @ link_trans
            current_rot = current_rot @ link_rot

            joint_rot = rotation_matrix_axis_angle(joint_axes[i], angles[i])
            current_rot = current_rot @ joint_rot

            transforms.append((current_pos, current_rot))

        return transforms

    def get_sphere_positions(angles):
        link_transforms = get_link_transforms(angles)
        sphere_positions = []

        for sphere_idx in range(n_spheres):
            link_idx = link_indices[sphere_idx]
            link_pos, link_rot = link_transforms[link_idx]
            local_center = sphere_centers_local[sphere_idx]
            world_center = link_pos + link_rot @ local_center
            sphere_positions.append(world_center)

        return jnp.stack(sphere_positions, axis=0)

    def world_collision_cost(angles, activation_distance=0.02):
        sphere_positions = get_sphere_positions(angles)
        total_cost = 0.0

        for si in range(n_spheres):
            # Check against sphere obstacles
            for obs in sphere_obstacles:
                dist = sphere_sphere_distance(
                    sphere_positions[si], sphere_radii[si],
                    obs['center'], obs['radius']
                )
                cost = jnp.maximum(0.0, activation_distance - dist)
                total_cost = total_cost + cost ** 2

            # Check against halfspace obstacles
            for obs in halfspace_obstacles:
                dist = sphere_halfspace_distance(
                    sphere_positions[si], sphere_radii[si],
                    obs['point'], obs['normal']
                )
                cost = jnp.maximum(0.0, activation_distance - dist)
                total_cost = total_cost + cost ** 2

        return total_cost

    return jax.jit(world_collision_cost)


def colldist_from_sdf(dist, activation_dist):
    """Convert signed distance to collision cost (smooth activation).

    Based on https://arxiv.org/pdf/2310.17274

    Parameters
    ----------
    dist : jax.numpy.ndarray
        Signed distances. Positive = separation, negative = penetration.
    activation_dist : float
        Distance threshold for cost activation.

    Returns
    -------
    jax.numpy.ndarray
        Collision cost values (<= 0).
    """
    _ensure_jax_cpu_on_mac()
    import jax.numpy as jnp

    dist = jnp.minimum(dist, activation_dist)
    dist = jnp.where(
        dist < 0,
        dist - 0.5 * activation_dist,
        -0.5 / (activation_dist + 1e-6) * (dist - activation_dist) ** 2,
    )
    dist = jnp.minimum(dist, 0.0)
    return dist

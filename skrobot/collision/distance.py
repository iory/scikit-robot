"""Backend-agnostic collision distance functions.

This module provides differentiable distance computation between collision
geometry primitives. All functions support both NumPy and JAX backends
through the `xp` parameter.

Example
-------
>>> from skrobot.collision import Sphere, Capsule, collision_distance
>>> import numpy as np
>>> s1 = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=0.5)
>>> s2 = Sphere(center=np.array([2.0, 0.0, 0.0]), radius=0.5)
>>> dist = collision_distance(s1, s2)  # returns 1.0
"""

import numpy as np

from skrobot.collision.geometry import Box
from skrobot.collision.geometry import Capsule
from skrobot.collision.geometry import HalfSpace
from skrobot.collision.geometry import Sphere


def sphere_sphere_distance(s1, s2, xp=np):
    """Compute signed distance between two spheres.

    Parameters
    ----------
    s1 : Sphere
        First sphere.
    s2 : Sphere
        Second sphere.
    xp : module
        Array module (numpy or jax.numpy).

    Returns
    -------
    array
        Signed distance. Positive = separation, negative = penetration.
    """
    diff = s1.center - s2.center
    dist = xp.sqrt(xp.sum(diff ** 2, axis=-1) + 1e-10)
    return dist - s1.radius - s2.radius


def capsule_capsule_distance(c1, c2, xp=np):
    """Compute signed distance between two capsules.

    A capsule is defined by a line segment (p1 to p2) and a radius.

    Parameters
    ----------
    c1 : Capsule
        First capsule.
    c2 : Capsule
        Second capsule.
    xp : module
        Array module (numpy or jax.numpy).

    Returns
    -------
    array
        Signed distance between capsules.
    """
    p1, q1, r1 = c1.p1, c1.p2, c1.radius
    p2, q2, r2 = c2.p1, c2.p2, c2.radius

    # Line segment to line segment distance
    d1 = q1 - p1  # Direction of segment 1
    d2 = q2 - p2  # Direction of segment 2
    r = p1 - p2

    a = xp.sum(d1 * d1, axis=-1) + 1e-10  # |d1|^2
    e = xp.sum(d2 * d2, axis=-1) + 1e-10  # |d2|^2
    f = xp.sum(d2 * r, axis=-1)

    b = xp.sum(d1 * d2, axis=-1)
    c = xp.sum(d1 * r, axis=-1)

    denom = a * e - b * b + 1e-10

    # Compute s and t parameters
    s = xp.clip((b * f - c * e) / denom, 0.0, 1.0)
    t = xp.clip((b * s + f) / e, 0.0, 1.0)

    # Recompute s based on clamped t
    s = xp.clip((b * t - c) / a, 0.0, 1.0)

    # Closest points on segments
    closest1 = p1 + s[..., None] * d1
    closest2 = p2 + t[..., None] * d2

    # Distance between closest points
    diff = closest1 - closest2
    dist = xp.sqrt(xp.sum(diff ** 2, axis=-1) + 1e-10)

    return dist - r1 - r2


def sphere_capsule_distance(sphere, capsule, xp=np):
    """Compute signed distance between sphere and capsule.

    Parameters
    ----------
    sphere : Sphere
        Sphere geometry.
    capsule : Capsule
        Capsule geometry.
    xp : module
        Array module (numpy or jax.numpy).

    Returns
    -------
    array
        Signed distance. Positive = separated, negative = penetrating.
    """
    # Find closest point on capsule segment to sphere center
    segment = capsule.p2 - capsule.p1
    segment_len_sq = xp.sum(segment ** 2, axis=-1, keepdims=True) + 1e-10

    # Project sphere center onto segment
    t = xp.sum((sphere.center - capsule.p1) * segment,
               axis=-1, keepdims=True) / segment_len_sq
    t = xp.clip(t, 0.0, 1.0)

    # Closest point on segment
    closest = capsule.p1 + t * segment

    # Distance from sphere center to closest point on capsule axis
    diff = sphere.center - closest
    dist = xp.sqrt(xp.sum(diff ** 2, axis=-1) + 1e-10)

    return dist - sphere.radius - capsule.radius


def sphere_box_distance(sphere, box, xp=np):
    """Compute signed distance between sphere and box.

    Parameters
    ----------
    sphere : Sphere
        Sphere geometry.
    box : Box
        Box geometry.
    xp : module
        Array module (numpy or jax.numpy).

    Returns
    -------
    array
        Signed distance. Positive = separated, negative = penetrating.
    """
    # Transform sphere center to box's local frame
    local_center = sphere.center - box.center
    if box.rotation is not None:
        # Apply inverse rotation (transpose for rotation matrix)
        local_center = xp.einsum('...ij,...j->...i',
                                 xp.swapaxes(box.rotation, -1, -2),
                                 local_center)

    # Clamp to box bounds to find closest point on box surface
    closest = xp.clip(local_center, -box.half_extents, box.half_extents)

    # Distance from sphere center to closest point
    diff = local_center - closest
    dist_to_surface = xp.sqrt(xp.sum(diff ** 2, axis=-1) + 1e-10)

    return dist_to_surface - sphere.radius


def capsule_box_distance(capsule, box, xp=np):
    """Compute signed distance between capsule and box.

    Uses iterative closest point refinement.

    Parameters
    ----------
    capsule : Capsule
        Capsule geometry.
    box : Box
        Box geometry.
    xp : module
        Array module (numpy or jax.numpy).

    Returns
    -------
    array
        Signed distance. Positive = separated, negative = penetrating.
    """
    # Transform capsule endpoints to box's local frame
    p1_local = capsule.p1 - box.center
    p2_local = capsule.p2 - box.center
    if box.rotation is not None:
        rot_inv = xp.swapaxes(box.rotation, -1, -2)
        p1_local = xp.einsum('...ij,...j->...i', rot_inv, p1_local)
        p2_local = xp.einsum('...ij,...j->...i', rot_inv, p2_local)

    # Helper: find closest point on segment to target point
    def closest_segment_point(seg_p1, seg_p2, target):
        segment = seg_p2 - seg_p1
        seg_len_sq = xp.sum(segment ** 2, axis=-1, keepdims=True) + 1e-10
        t = xp.sum((target - seg_p1) * segment,
                   axis=-1, keepdims=True) / seg_len_sq
        t = xp.clip(t, 0.0, 1.0)
        return seg_p1 + t * segment

    # Iteration 1: closest point on segment to box center (origin)
    origin = xp.zeros_like(p1_local)
    pt_seg = closest_segment_point(p1_local, p2_local, origin)
    pt_box = xp.clip(pt_seg, -box.half_extents, box.half_extents)

    # Iteration 2: refine with closest point on segment to pt_box
    pt_seg = closest_segment_point(p1_local, p2_local, pt_box)

    # Compute SDF from segment point to box
    # Clamp to find closest point on box surface
    closest = xp.clip(pt_seg, -box.half_extents, box.half_extents)
    diff = pt_seg - closest
    dist_to_surface = xp.sqrt(xp.sum(diff ** 2, axis=-1) + 1e-10)

    # For inside case: compute penetration depth as distance to nearest face
    face_dists = box.half_extents - xp.abs(pt_seg)
    min_face_dist = xp.min(face_dists, axis=-1)

    # If inside (dist ~= 0), return negative penetration
    is_inside = dist_to_surface < 1e-5
    sdf = xp.where(is_inside, -min_face_dist, dist_to_surface)

    return sdf - capsule.radius


def sphere_halfspace_distance(sphere, halfspace, xp=np):
    """Compute signed distance from sphere to halfspace (plane).

    Parameters
    ----------
    sphere : Sphere
        Sphere geometry.
    halfspace : HalfSpace
        Halfspace geometry.
    xp : module
        Array module (numpy or jax.numpy).

    Returns
    -------
    array
        Signed distance. Positive = sphere above plane.
    """
    # Distance from sphere center to plane
    to_center = sphere.center - halfspace.point
    plane_dist = xp.sum(to_center * halfspace.normal, axis=-1)

    return plane_dist - sphere.radius


def capsule_halfspace_distance(capsule, halfspace, xp=np):
    """Compute signed distance from capsule to halfspace (plane).

    Parameters
    ----------
    capsule : Capsule
        Capsule geometry.
    halfspace : HalfSpace
        Halfspace geometry.
    xp : module
        Array module (numpy or jax.numpy).

    Returns
    -------
    array
        Signed distance. Positive = capsule above plane.
    """
    # Create temporary spheres at capsule endpoints
    s1 = Sphere(center=capsule.p1, radius=capsule.radius)
    s2 = Sphere(center=capsule.p2, radius=capsule.radius)

    # Distance from each endpoint (as sphere) to plane
    dist1 = sphere_halfspace_distance(s1, halfspace, xp)
    dist2 = sphere_halfspace_distance(s2, halfspace, xp)

    return xp.minimum(dist1, dist2)


def box_halfspace_distance(box, halfspace, xp=np):
    """Compute signed distance from box to halfspace (plane).

    Parameters
    ----------
    box : Box
        Box geometry.
    halfspace : HalfSpace
        Halfspace geometry.
    xp : module
        Array module (numpy or jax.numpy).

    Returns
    -------
    array
        Signed distance. Positive = box above plane.
    """
    # Get rotation matrix (identity if None)
    if box.rotation is None:
        rotation = xp.eye(3)
    else:
        rotation = box.rotation

    # Signs for the 8 corners
    signs = xp.array([
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
    corners_local = box.half_extents[..., None, :] * signs  # (..., 8, 3)

    # Transform to world frame
    corners_world = xp.einsum('...ij,...kj->...ki', rotation, corners_local)
    corners_world = corners_world + box.center[..., None, :]  # (..., 8, 3)

    # Compute distance from each corner to plane
    to_corners = corners_world - halfspace.point[..., None, :]  # (..., 8, 3)
    corner_dists = xp.sum(to_corners * halfspace.normal[..., None, :],
                          axis=-1)  # (..., 8)

    # Return minimum distance (most penetrating corner)
    return xp.min(corner_dists, axis=-1)


def collision_distance(geom1, geom2, xp=np):
    """Compute signed distance between two collision geometries.

    Automatically dispatches to the appropriate distance function based
    on geometry types.

    Parameters
    ----------
    geom1 : CollisionGeometry
        First geometry.
    geom2 : CollisionGeometry
        Second geometry.
    xp : module
        Array module (numpy or jax.numpy).

    Returns
    -------
    array
        Signed distance. Positive = separated, negative = penetrating.

    Raises
    ------
    NotImplementedError
        If the geometry pair is not supported.
    """
    # Normalize order for symmetric pairs
    type1, type2 = type(geom1).__name__, type(geom2).__name__

    # Sphere-Sphere
    if isinstance(geom1, Sphere) and isinstance(geom2, Sphere):
        return sphere_sphere_distance(geom1, geom2, xp)

    # Capsule-Capsule
    if isinstance(geom1, Capsule) and isinstance(geom2, Capsule):
        return capsule_capsule_distance(geom1, geom2, xp)

    # Sphere-Capsule
    if isinstance(geom1, Sphere) and isinstance(geom2, Capsule):
        return sphere_capsule_distance(geom1, geom2, xp)
    if isinstance(geom1, Capsule) and isinstance(geom2, Sphere):
        return sphere_capsule_distance(geom2, geom1, xp)

    # Sphere-Box
    if isinstance(geom1, Sphere) and isinstance(geom2, Box):
        return sphere_box_distance(geom1, geom2, xp)
    if isinstance(geom1, Box) and isinstance(geom2, Sphere):
        return sphere_box_distance(geom2, geom1, xp)

    # Capsule-Box
    if isinstance(geom1, Capsule) and isinstance(geom2, Box):
        return capsule_box_distance(geom1, geom2, xp)
    if isinstance(geom1, Box) and isinstance(geom2, Capsule):
        return capsule_box_distance(geom2, geom1, xp)

    # Sphere-HalfSpace
    if isinstance(geom1, Sphere) and isinstance(geom2, HalfSpace):
        return sphere_halfspace_distance(geom1, geom2, xp)
    if isinstance(geom1, HalfSpace) and isinstance(geom2, Sphere):
        return sphere_halfspace_distance(geom2, geom1, xp)

    # Capsule-HalfSpace
    if isinstance(geom1, Capsule) and isinstance(geom2, HalfSpace):
        return capsule_halfspace_distance(geom1, geom2, xp)
    if isinstance(geom1, HalfSpace) and isinstance(geom2, Capsule):
        return capsule_halfspace_distance(geom2, geom1, xp)

    # Box-HalfSpace
    if isinstance(geom1, Box) and isinstance(geom2, HalfSpace):
        return box_halfspace_distance(geom1, geom2, xp)
    if isinstance(geom1, HalfSpace) and isinstance(geom2, Box):
        return box_halfspace_distance(geom2, geom1, xp)

    raise NotImplementedError(
        f"Distance computation not implemented for {type1}-{type2} pair"
    )


def colldist_from_sdf(dist, activation_dist, xp=np):
    """Convert signed distance to collision cost (smooth activation).

    Based on https://arxiv.org/pdf/2310.17274

    This function provides a smooth, differentiable cost that:
    - Is zero when distance > activation_dist
    - Increases quadratically as distance approaches activation_dist
    - Increases linearly for penetration (negative distance)

    Parameters
    ----------
    dist : array
        Signed distances. Positive = separation, negative = penetration.
    activation_dist : float
        Distance threshold for cost activation.
    xp : module
        Array module (numpy or jax.numpy).

    Returns
    -------
    array
        Collision cost values (<= 0).
    """
    dist = xp.minimum(dist, activation_dist)
    dist = xp.where(
        dist < 0,
        dist - 0.5 * activation_dist,
        -0.5 / (activation_dist + 1e-6) * (dist - activation_dist) ** 2,
    )
    dist = xp.minimum(dist, 0.0)
    return dist

"""Utilities for fitting primitive shapes to meshes."""

from logging import getLogger

import numpy as np

from skrobot._lazy_imports import _lazy_trimesh


logger = getLogger(__name__)


def compute_mesh_primitive_iou(mesh, primitive_mesh, voxel_pitch=None):
    """Compute intersection over union between mesh and primitive using voxelization.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Original mesh.
    primitive_mesh : trimesh.Trimesh
        Primitive approximation mesh.
    voxel_pitch : float, optional
        Voxel size for discretization. If None, automatically determined.

    Returns
    -------
    iou : float
        Intersection over union ratio (0-1).
    """
    _lazy_trimesh()

    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    max_dim = np.max(dimensions)

    if voxel_pitch is None:
        voxel_pitch = max_dim / 32

    mesh_voxel = mesh.voxelized(pitch=voxel_pitch)
    prim_voxel = primitive_mesh.voxelized(pitch=voxel_pitch)

    mesh_grid = mesh_voxel.matrix
    prim_grid = prim_voxel.matrix

    mesh_origin = mesh_voxel.translation
    prim_origin = prim_voxel.translation

    mesh_indices = np.argwhere(mesh_grid)
    prim_indices = np.argwhere(prim_grid)

    mesh_coords = mesh_indices * voxel_pitch + mesh_origin
    prim_coords = prim_indices * voxel_pitch + prim_origin

    mesh_set = set(map(tuple, np.round(mesh_coords / voxel_pitch).astype(int)))
    prim_set = set(map(tuple, np.round(prim_coords / voxel_pitch).astype(int)))

    intersection_count = len(mesh_set & prim_set)
    union_count = len(mesh_set | prim_set)

    if union_count == 0:
        return 0.0

    iou = intersection_count / union_count

    logger.debug("Voxel IoU: mesh=%d, prim=%d, intersection=%d, union=%d, iou=%.4f",
                 len(mesh_set), len(prim_set), intersection_count, union_count, iou)

    return iou


def fit_box_to_mesh(mesh):
    """Fit a box primitive to a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to fit a box to.

    Returns
    -------
    extents : np.ndarray
        Box dimensions [x, y, z].
    center : np.ndarray
        Center position of the box.
    rotation : np.ndarray
        Rotation matrix (3x3) of the box. Currently returns identity.
    """
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    extents = bounds[1] - bounds[0]

    return extents, center, np.eye(3)


def fit_sphere_to_mesh(mesh):
    """Fit a sphere primitive to a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to fit a sphere to.

    Returns
    -------
    radius : float
        Radius of the sphere.
    center : np.ndarray
        Center position of the sphere.
    """
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    extents = bounds[1] - bounds[0]

    radius = np.max(extents) / 2

    return radius, center


def fit_cylinder_to_mesh(mesh):
    """Fit a cylinder primitive to a mesh.

    This function analyzes the mesh bounding box and assumes the smallest
    dimension is the cylinder's height (axis direction).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to fit a cylinder to.

    Returns
    -------
    radius : float
        Radius of the cylinder.
    height : float
        Height of the cylinder.
    axis : np.ndarray
        Unit vector indicating the cylinder's axis direction.
    center : np.ndarray
        Center position of the cylinder.
    """
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    dimensions = bounds[1] - bounds[0]

    height_idx = np.argmin(dimensions)
    height = dimensions[height_idx]

    other_dims = [dimensions[i] for i in range(3) if i != height_idx]
    radius = max(other_dims) / 2

    axis = np.zeros(3)
    axis[height_idx] = 1

    return radius, height, axis, center


def fit_capsule_to_mesh(mesh):
    """Fit a capsule primitive to a mesh.

    A capsule is represented as a cylinder with hemispherical caps.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to fit a capsule to.

    Returns
    -------
    radius : float
        Radius of the capsule (both cylinder and hemisphere).
    height : float
        Height of the cylindrical section (excluding hemispheres).
    axis : np.ndarray
        Unit vector indicating the capsule's axis direction.
    center : np.ndarray
        Center position of the capsule.
    """
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    dimensions = bounds[1] - bounds[0]

    height_idx = np.argmin(dimensions)
    total_height = dimensions[height_idx]

    other_dims = [dimensions[i] for i in range(3) if i != height_idx]
    radius = max(other_dims) / 2

    height = max(0, total_height - 2 * radius)

    axis = np.zeros(3)
    axis[height_idx] = 1

    return radius, height, axis, center


def create_primitive_mesh(primitive_params):
    """Create a trimesh object from primitive parameters.

    Parameters
    ----------
    primitive_params : dict
        Dictionary with primitive parameters.

    Returns
    -------
    primitive_mesh : trimesh.Trimesh
        Mesh representation of the primitive.
    """
    trimesh = _lazy_trimesh()
    prim_type = primitive_params['type']
    center = primitive_params['center']

    if prim_type == 'box':
        extents = primitive_params['extents']
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        return mesh

    elif prim_type == 'sphere':
        radius = primitive_params['radius']
        mesh = trimesh.creation.icosphere(radius=radius, subdivisions=3)
        mesh.apply_translation(center)
        return mesh

    elif prim_type == 'cylinder':
        radius = primitive_params['radius']
        height = primitive_params['height']
        axis = primitive_params['axis']

        mesh = trimesh.creation.cylinder(radius=radius, height=height)

        z_axis = np.array([0, 0, 1])
        axis = axis / np.linalg.norm(axis)

        if not np.allclose(axis, z_axis):
            from skrobot.coordinates.math import rotation_matrix
            if np.allclose(axis, -z_axis):
                rot_matrix = rotation_matrix(np.pi, [1, 0, 0])
            else:
                rotation_axis = np.cross(z_axis, axis)
                angle = np.arccos(np.clip(np.dot(z_axis, axis), -1.0, 1.0))
                rot_matrix = rotation_matrix(angle, rotation_axis)

            transform = np.eye(4)
            transform[:3, :3] = rot_matrix
            mesh.apply_transform(transform)

        mesh.apply_translation(center)
        return mesh

    else:
        raise ValueError(f"Unknown primitive type: {prim_type}")


def estimate_best_primitive(mesh):
    """Estimate which primitive shape best fits a mesh using volume IoU.

    This function tries fitting box, cylinder, and sphere primitives and
    selects the one with the highest intersection over union (IoU) ratio.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to analyze.

    Returns
    -------
    primitive_type : str
        One of: 'box', 'cylinder', 'sphere'
    score : float
        IoU score for the best primitive type (0-1).
    """
    trimesh = _lazy_trimesh()

    if isinstance(mesh, list):
        if len(mesh) == 0:
            raise ValueError("Empty mesh list")
        combined_mesh = trimesh.util.concatenate(mesh)
    else:
        combined_mesh = mesh

    candidates = []

    try:
        box_params = fit_primitive_to_mesh(combined_mesh, primitive_type='box')
        box_mesh = create_primitive_mesh(box_params)
        box_iou = compute_mesh_primitive_iou(combined_mesh, box_mesh)
        candidates.append(('box', box_iou))
        logger.debug("Box IoU: %.4f", box_iou)
    except Exception as e:
        logger.debug("Box fitting failed: %s", e)
        candidates.append(('box', 0.0))

    try:
        sphere_params = fit_primitive_to_mesh(combined_mesh, primitive_type='sphere')
        sphere_mesh = create_primitive_mesh(sphere_params)
        sphere_iou = compute_mesh_primitive_iou(combined_mesh, sphere_mesh)
        candidates.append(('sphere', sphere_iou))
        logger.debug("Sphere IoU: %.4f", sphere_iou)
    except Exception as e:
        logger.debug("Sphere fitting failed: %s", e)
        candidates.append(('sphere', 0.0))

    try:
        cylinder_params = fit_primitive_to_mesh(combined_mesh, primitive_type='cylinder')
        cylinder_mesh = create_primitive_mesh(cylinder_params)
        cylinder_iou = compute_mesh_primitive_iou(combined_mesh, cylinder_mesh)
        candidates.append(('cylinder', cylinder_iou))
        logger.debug("Cylinder IoU: %.4f", cylinder_iou)
    except Exception as e:
        logger.debug("Cylinder fitting failed: %s", e)
        candidates.append(('cylinder', 0.0))

    best_primitive, best_score = max(candidates, key=lambda x: x[1])

    logger.info("Best primitive: %s (IoU: %.4f)", best_primitive, best_score)

    return best_primitive, best_score


def fit_primitive_to_mesh(mesh, primitive_type=None):
    """Fit a primitive shape to a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to fit a primitive to.
    primitive_type : str, optional
        Type of primitive to fit: 'box', 'cylinder', 'sphere', or 'capsule'.
        If None, automatically estimates the best primitive type.

    Returns
    -------
    primitive_params : dict
        Dictionary containing primitive parameters:
        - 'type': primitive type
        - 'center': center position
        - For 'box': 'extents', 'rotation'
        - For 'cylinder': 'radius', 'height', 'axis'
        - For 'sphere': 'radius'
        - For 'capsule': 'radius', 'height', 'axis'
    """
    trimesh = _lazy_trimesh()

    if isinstance(mesh, list):
        if len(mesh) == 0:
            raise ValueError("Empty mesh list")
        combined_mesh = trimesh.util.concatenate(mesh)
    else:
        combined_mesh = mesh

    if primitive_type is None:
        primitive_type, _ = estimate_best_primitive(combined_mesh)
        logger.info("Auto-detected primitive type: %s", primitive_type)

    if primitive_type == 'box':
        extents, center, rotation = fit_box_to_mesh(combined_mesh)
        return {
            'type': 'box',
            'extents': extents,
            'center': center,
            'rotation': rotation
        }
    elif primitive_type == 'sphere':
        radius, center = fit_sphere_to_mesh(combined_mesh)
        return {
            'type': 'sphere',
            'radius': radius,
            'center': center
        }
    elif primitive_type == 'cylinder':
        radius, height, axis, center = fit_cylinder_to_mesh(combined_mesh)
        return {
            'type': 'cylinder',
            'radius': radius,
            'height': height,
            'axis': axis,
            'center': center
        }
    elif primitive_type == 'capsule':
        radius, height, axis, center = fit_capsule_to_mesh(combined_mesh)
        return {
            'type': 'capsule',
            'radius': radius,
            'height': height,
            'axis': axis,
            'center': center
        }
    else:
        raise ValueError(f"Unknown primitive type: {primitive_type}")

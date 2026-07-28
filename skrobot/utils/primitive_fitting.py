"""Utilities for fitting primitive shapes to meshes."""

from logging import getLogger

import numpy as np

from skrobot._lazy_imports import _lazy_trimesh
from skrobot.coordinates.math import rotation_matrix_from_vectors


logger = getLogger(__name__)


__all__ = [
    'compute_mesh_primitive_iou',
    'fit_box_to_mesh',
    'fit_sphere_to_mesh',
    'fit_cylinder_to_mesh',
    'fit_capsule_to_mesh',
    'create_primitive_mesh',
    'estimate_best_primitive',
    'fit_primitive_to_mesh',
    'primitive_params_to_origin',
]


def _voxel_coord_set(mesh, voxel_pitch, fill=True):
    """Voxelize a mesh and return the set of occupied integer voxel keys.

    When ``fill`` is True the grid is solid-filled so the result represents the
    mesh *volume*, not just its surface shell -- making IoU meaningful even for
    thin, non-watertight surface meshes. Keys are ``round(point / pitch)``, the
    same lattice used by :func:`_primitive_voxel_set`.
    """
    voxel = mesh.voxelized(pitch=voxel_pitch)
    if fill:
        try:
            voxel = voxel.fill()
        except Exception as e:  # noqa: BLE001
            logger.debug("Voxel fill failed (%s); using surface voxels", e)
    coords = np.argwhere(voxel.matrix) * voxel_pitch + voxel.translation
    return set(map(tuple, np.round(coords / voxel_pitch).astype(int)))


def compute_mesh_primitive_iou(mesh, primitive_mesh, voxel_pitch=None,
                               fill=True):
    """Compute intersection over union between mesh and primitive using voxelization.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Original mesh.
    primitive_mesh : trimesh.Trimesh
        Primitive approximation mesh.
    voxel_pitch : float, optional
        Voxel size for discretization. If None, automatically determined.
    fill : bool, optional
        If True (default), solid-fill both voxelizations so the IoU reflects
        volumetric overlap. If False, compare surface voxels only (the old
        behavior), which systematically under-counts overlap for solid
        primitives versus thin surface meshes.

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

    mesh_set = _voxel_coord_set(mesh, voxel_pitch, fill=fill)
    prim_set = _voxel_coord_set(primitive_mesh, voxel_pitch, fill=fill)

    intersection_count = len(mesh_set & prim_set)
    union_count = len(mesh_set | prim_set)

    if union_count == 0:
        return 0.0

    iou = intersection_count / union_count

    logger.debug("Voxel IoU: mesh=%d, prim=%d, intersection=%d, union=%d, iou=%.4f",
                 len(mesh_set), len(prim_set), intersection_count, union_count, iou)

    return iou


def fit_box_to_mesh(mesh, oriented=False):
    """Fit a box primitive to a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to fit a box to.
    oriented : bool, optional
        If True, fit an oriented bounding box (OBB) using
        ``trimesh.bounds.oriented_bounds``, returning a real (possibly
        non-identity) rotation. If False (default), fit an axis-aligned
        bounding box (AABB) with identity rotation.

    Returns
    -------
    extents : np.ndarray
        Box dimensions [x, y, z] (in the box's own frame).
    center : np.ndarray
        Center position of the box (in the mesh frame).
    rotation : np.ndarray
        Rotation matrix (3x3) of the box in the mesh frame. Identity when
        ``oriented`` is False.
    """
    if oriented:
        trimesh = _lazy_trimesh()
        # to_origin maps the mesh so the OBB is centered at the origin and
        # axis-aligned. The box pose in the mesh frame is therefore its
        # inverse.
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        transform = np.linalg.inv(to_origin)
        rotation = transform[:3, :3].copy()
        center = transform[:3, 3].copy()
        return np.asarray(extents, dtype=float), center, rotation

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


def fit_cylinder_to_mesh(mesh, oriented=False):
    """Fit a cylinder primitive to a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to fit a cylinder to.
    oriented : bool, optional
        If True, fit a minimum-volume cylinder with an arbitrary axis
        direction using ``trimesh.bounds.minimum_cylinder``. If False
        (default), analyze the mesh bounding box and assume the smallest
        dimension is the cylinder's height (an axis-aligned axis).

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
    if oriented:
        trimesh = _lazy_trimesh()
        result = trimesh.bounds.minimum_cylinder(mesh)
        transform = np.asarray(result['transform'], dtype=float)
        radius = float(result['radius'])
        height = float(result['height'])
        # trimesh cylinders are defined along the local Z-axis; the fitted
        # axis direction is that Z-axis transformed into the mesh frame.
        axis = transform[:3, 2].copy()
        center = transform[:3, 3].copy()
        return radius, height, axis, center

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
        rotation = primitive_params.get('rotation', None)
        if rotation is not None and not np.allclose(rotation, np.eye(3)):
            transform = np.eye(4)
            transform[:3, :3] = rotation
            mesh.apply_transform(transform)
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

        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix_from_vectors([0, 0, 1], axis)
        mesh.apply_transform(transform)

        mesh.apply_translation(center)
        return mesh

    else:
        raise ValueError(f"Unknown primitive type: {prim_type}")


def _fit_type_variants(mesh, primitive_type, oriented=None):
    """Build candidate fits for one primitive type.

    ``oriented`` selects which orientation variants to build for box/cylinder:
    ``None`` builds both an axis-aligned and an oriented (OBB / minimum-volume)
    fit so the caller can keep whichever scores better; ``True`` builds only
    the oriented fit; ``False`` builds only the axis-aligned fit. Spheres and
    capsules yield a single candidate regardless. Failing fitters are skipped
    rather than raising.
    """
    orientations = (False, True) if oriented is None else (bool(oriented),)
    variants = []
    if primitive_type == 'box':
        for o in orientations:
            try:
                extents, center, rotation = fit_box_to_mesh(mesh, oriented=o)
                variants.append({'type': 'box', 'extents': extents,
                                 'center': center, 'rotation': rotation})
            except Exception as e:  # noqa: BLE001
                logger.debug("box fit (oriented=%s) failed: %s", o, e)
    elif primitive_type == 'cylinder':
        for o in orientations:
            try:
                radius, height, axis, center = fit_cylinder_to_mesh(
                    mesh, oriented=o)
                variants.append({'type': 'cylinder', 'radius': radius,
                                 'height': height, 'axis': axis,
                                 'center': center})
            except Exception as e:  # noqa: BLE001
                logger.debug("cylinder fit (oriented=%s) failed: %s", o, e)
    elif primitive_type == 'sphere':
        radius, center = fit_sphere_to_mesh(mesh)
        variants.append({'type': 'sphere', 'radius': radius, 'center': center})
    elif primitive_type == 'capsule':
        radius, height, axis, center = fit_capsule_to_mesh(mesh)
        variants.append({'type': 'capsule', 'radius': radius, 'height': height,
                         'axis': axis, 'center': center})
    else:
        raise ValueError(f"Unknown primitive type: {primitive_type}")
    return variants


def _primitive_voxel_set(params, pitch):
    """Integer voxel keys a primitive occupies, via analytic containment.

    Enumerate the integer lattice over the primitive's bounding box and test
    each voxel center (``key * pitch``) for containment analytically -- no
    voxelization. ``margin`` (half a voxel) dilates the primitive to match the
    outward inflation of trimesh surface voxelization, keeping the occupancy
    discretization-consistent with :func:`_voxel_coord_set`.
    """
    prim_type = params['type']
    center = np.asarray(params['center'], dtype=float)
    margin = pitch / 2.0
    bounds = create_primitive_mesh(params).bounds
    lo = np.floor((bounds[0] - margin) / pitch).astype(int)
    hi = np.ceil((bounds[1] + margin) / pitch).astype(int)
    ranges = [np.arange(lo[i], hi[i] + 1) for i in range(3)]
    if any(r.size == 0 for r in ranges):
        return set()
    grid = np.stack([g.ravel() for g in np.meshgrid(*ranges, indexing='ij')],
                    axis=1)
    d = grid * pitch - center
    if prim_type == 'box':
        rotation = np.asarray(params.get('rotation', np.eye(3)), dtype=float)
        half = np.asarray(params['extents'], dtype=float) / 2.0 + margin
        inside = np.all(np.abs(d @ rotation) <= half, axis=1)
    elif prim_type == 'sphere':
        r = float(params['radius']) + margin
        inside = np.sum(d * d, axis=1) <= r * r
    elif prim_type in ('cylinder', 'capsule'):
        r = float(params['radius'])
        h = float(params['height'])
        if prim_type == 'capsule':
            h = h + 2.0 * r
        axis = np.asarray(params['axis'], dtype=float)
        axis = axis / np.linalg.norm(axis)
        along = d @ axis
        inside = (np.abs(along) <= h / 2.0 + margin) \
            & (np.sum(d * d, axis=1) - along * along <= (r + margin) ** 2)
    else:
        raise ValueError(f"Unknown primitive type: {prim_type}")
    return set(map(tuple, grid[inside]))


def _select_best_primitive(mesh, candidates):
    """Return the candidate params with the highest solid voxel IoU, and score.

    The mesh is voxel-filled once; each candidate is scored by analytic
    containment on the same lattice (no per-candidate voxelization), which is
    where the bulk of the cost used to be.
    """
    if not candidates:
        raise ValueError("No candidate primitives to select from")
    if len(candidates) == 1:
        return candidates[0], 0.0

    max_dim = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    if max_dim <= 0:
        return candidates[0], 0.0
    pitch = max_dim / 32.0

    mesh_set = _voxel_coord_set(mesh, pitch, fill=True)
    if not mesh_set:
        return candidates[0], 0.0

    best_params = None
    best_score = -1.0
    for params in candidates:
        try:
            prim_set = _primitive_voxel_set(params, pitch)
        except Exception as e:  # noqa: BLE001
            logger.debug("scoring %s failed: %s", params.get('type'), e)
            continue
        union = len(mesh_set | prim_set)
        score = len(mesh_set & prim_set) / union if union else 0.0
        logger.debug("candidate %s IoU=%.4f", params['type'], score)
        if score > best_score:
            best_score = score
            best_params = params

    if best_params is None:
        return candidates[0], 0.0
    return best_params, best_score


def estimate_best_primitive(mesh, oriented=None):
    """Estimate which primitive shape best fits a mesh using volume IoU.

    Fits box, cylinder, and sphere candidates and returns the type of the one
    with the highest solid voxel intersection-over-union.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to analyze.
    oriented : bool or None, optional
        Orientation policy for box/cylinder candidates, see
        :func:`fit_primitive_to_mesh`. ``None`` (default) considers both
        axis-aligned and oriented fits.

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

    candidates = (_fit_type_variants(combined_mesh, 'box', oriented)
                  + _fit_type_variants(combined_mesh, 'cylinder', oriented)
                  + _fit_type_variants(combined_mesh, 'sphere', oriented))
    best_params, best_score = _select_best_primitive(combined_mesh, candidates)

    logger.info("Best primitive: %s (IoU: %.4f)",
                best_params['type'], best_score)

    return best_params['type'], best_score


def fit_primitive_to_mesh(mesh, primitive_type=None, oriented=None):
    """Fit a primitive shape to a mesh.

    This is the primary public entry point for per-mesh primitive fitting.
    It operates purely on a :class:`trimesh.Trimesh` (or a list of them) and
    returns a plain parameter dictionary, with no file or URDF I/O, so it can
    be composed into other pipelines.

    Parameters
    ----------
    mesh : trimesh.Trimesh or list of trimesh.Trimesh
        Input mesh to fit a primitive to. A list is concatenated first.
    primitive_type : str, optional
        Type of primitive to fit: 'box', 'cylinder', 'sphere', or 'capsule'.
        If None, automatically selects the best-fitting type across box,
        cylinder, and sphere (see :func:`estimate_best_primitive`).
    oriented : bool or None, optional
        Orientation policy for box and cylinder fits:

        - ``None`` (default): produce both an axis-aligned and an oriented
          (OBB / minimum-volume) candidate and keep whichever has the higher
          solid voxel IoU -- the tighter fit when it genuinely helps, the
          conservative axis-aligned fit otherwise, without the caller having
          to choose.
        - ``True``: force the oriented fit (real OBB rotation / arbitrary
          cylinder axis).
        - ``False``: force the axis-aligned fit (identity box rotation /
          axis-aligned cylinder axis).

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

    See Also
    --------
    primitive_params_to_origin : Convert the returned params into a URDF
        ``<origin>`` (xyz, rpy) pair.
    """
    trimesh = _lazy_trimesh()

    if isinstance(mesh, list):
        if len(mesh) == 0:
            raise ValueError("Empty mesh list")
        combined_mesh = trimesh.util.concatenate(mesh)
    else:
        combined_mesh = mesh

    if primitive_type is None:
        candidates = (_fit_type_variants(combined_mesh, 'box', oriented)
                      + _fit_type_variants(combined_mesh, 'cylinder', oriented)
                      + _fit_type_variants(combined_mesh, 'sphere', oriented))
    else:
        candidates = _fit_type_variants(combined_mesh, primitive_type, oriented)

    best_params, _ = _select_best_primitive(combined_mesh, candidates)
    logger.info("Fitted primitive: %s", best_params['type'])
    return best_params


def primitive_params_to_origin(primitive_params):
    """Compute the URDF ``<origin>`` for a fitted primitive.

    Given the parameter dictionary returned by :func:`fit_primitive_to_mesh`,
    return the ``(xyz, rpy)`` origin of the primitive expressed in the frame
    of the mesh that was fitted. This decouples the origin computation from
    any XML / indentation handling, so consumers can emit their own
    ``<geometry>`` element and only need the params plus this origin.

    Parameters
    ----------
    primitive_params : dict
        Primitive parameters from :func:`fit_primitive_to_mesh`.

    Returns
    -------
    xyz : np.ndarray
        Translation [x, y, z] of the primitive origin (the fitted center).
    rpy : np.ndarray
        Roll-pitch-yaw [roll, pitch, yaw] of the primitive origin, in the
        URDF fixed-axis convention.

    Notes
    -----
    - For 'box', the rotation is taken from ``params['rotation']`` (identity
      for an axis-aligned fit, a real rotation for an oriented fit).
    - For 'sphere', the rotation is identity (spheres are isotropic).
    - For 'cylinder' and 'capsule', the rotation aligns the primitive's
      local Z-axis with the fitted ``axis`` direction.
    """
    from skrobot.coordinates.math import matrix2rpy
    from skrobot.coordinates.math import rotation_matrix_z_to_axis

    prim_type = primitive_params['type']
    xyz = np.asarray(primitive_params['center'], dtype=float)

    if prim_type == 'box':
        rotation = np.asarray(
            primitive_params.get('rotation', np.eye(3)), dtype=float)
        rpy = matrix2rpy(rotation)
    elif prim_type == 'sphere':
        rpy = np.zeros(3)
    elif prim_type in ('cylinder', 'capsule'):
        axis = np.asarray(primitive_params['axis'], dtype=float)
        rotation = rotation_matrix_z_to_axis(axis)
        rpy = matrix2rpy(rotation)
    else:
        raise ValueError(f"Unknown primitive type: {prim_type}")

    return xyz, np.asarray(rpy, dtype=float)

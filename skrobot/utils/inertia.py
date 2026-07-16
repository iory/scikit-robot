"""Compute URDF link inertials (mass, centre of mass, inertia tensor) from
meshes, so a generated URDF carries real dynamics instead of placeholders.

The mesh is treated as a uniform-density solid.  Many meshes are NOT
watertight, which makes trimesh's volume integral unreliable; the mass
properties therefore fall back to the convex hull (always watertight, a mild
over-estimate) and, failing that, the oriented bounding box -- and the used
``method`` is reported instead of silently emitting wrong numbers.
"""

from logging import getLogger
import os

import numpy as np

from skrobot._lazy_imports import _lazy_trimesh
from skrobot.coordinates.math import rpy2matrix


logger = getLogger(__name__)


__all__ = [
    'DEFAULT_DENSITY',
    'mesh_mass_properties',
    'transform_inertial',
    'link_inertial_from_mesh',
    'rescale_inertial_to_mass',
    'validate_inertia',
]


DEFAULT_DENSITY = 1000.0  # kg/m^3 (water) -- generic light part


def _solid_properties(mesh, density):
    """(mass, com(3,), I 3x3 about the com) for ``mesh`` as a solid of
    ``density``, or None if the mesh has no usable volume."""
    if mesh is None or not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        return None
    volume = float(getattr(mesh, 'volume', 0.0) or 0.0)
    if not np.isfinite(volume) or volume <= 0:
        return None
    mesh.density = float(density)
    mass = float(mesh.mass)
    com = np.asarray(mesh.center_mass, dtype=float)
    inertia = np.asarray(mesh.moment_inertia, dtype=float)
    if not (np.isfinite(mass) and mass > 0 and np.all(np.isfinite(com))
            and np.all(np.isfinite(inertia))):
        return None
    return mass, com, inertia


_PROPS_CACHE = {}


def mesh_mass_properties(mesh_path, density=DEFAULT_DENSITY, scale=1.0):
    """Mass properties of a mesh file treated as a uniform-density solid.

    Non-watertight meshes fall back to the convex hull and then to the
    oriented bounding box.  Results are cached by
    ``(path, mtime, density, scale)``: loading plus the watertight checks
    dominate the cost and the result is pose independent, so repeated
    calls for an unchanged file are effectively free.

    Parameters
    ----------
    mesh_path : str
        Path to the mesh (any format trimesh can load).
    density : float
        Material density in kg/m^3.
    scale : float
        Mesh-unit to metre factor (e.g. 0.001 for millimetre meshes).

    Returns
    -------
    props : tuple or None
        ``(mass, com, inertia, method)`` where ``com`` has shape (3,),
        ``inertia`` is the 3x3 tensor about the centre of mass (both in
        scaled mesh coordinates) and ``method`` is ``'mesh'``, ``'hull'``
        or ``'bbox'``.  None if no usable geometry was found.
    """
    try:
        key = (os.path.abspath(mesh_path), os.path.getmtime(mesh_path),
               float(density), float(scale))
    except OSError:
        return None
    if key in _PROPS_CACHE:
        return _PROPS_CACHE[key]
    result = None
    try:
        trimesh = _lazy_trimesh()
        mesh = trimesh.load(mesh_path, force='mesh')
        if mesh is not None and hasattr(mesh, 'vertices') \
                and len(mesh.vertices):
            mesh = mesh.copy()
            mesh.apply_scale(scale)
            method = 'mesh'
            props = _solid_properties(mesh, density) \
                if mesh.is_watertight else None
            if props is None:
                method = 'hull'
                try:
                    props = _solid_properties(mesh.convex_hull, density)
                except Exception:
                    props = None
            if props is None:
                method = 'bbox'
                try:
                    props = _solid_properties(mesh.bounding_box_oriented,
                                              density)
                except Exception:
                    props = None
            if props is not None:
                mass, com, inertia = props
                result = (mass, com, inertia, method)
    except Exception as e:  # noqa: BLE001
        logger.debug('mass properties failed for %s (%s)', mesh_path, e)
        result = None
    _PROPS_CACHE[key] = result
    return result


def transform_inertial(mass, com, inertia, visual_xyz, visual_rpy,
                       method='given'):
    """Express mass properties given in a visual/mesh frame in the link frame.

    The rotation rotates the tensor (taken about the centre of mass, so the
    translation does not enter it) and the translation moves the centre of
    mass -- exactly the mapping a URDF ``<visual><origin>`` defines.

    Parameters
    ----------
    mass : float
        Mass in kg.
    com : sequence of 3 floats
        Centre of mass in the source frame.
    inertia : numpy.ndarray or sequence of 6 floats
        Either the 3x3 tensor about the com, or the 6 components
        ``(ixx, ixy, ixz, iyy, iyz, izz)``.
    visual_xyz, visual_rpy : sequence of 3 floats
        The URDF origin mapping the source frame into the link frame.
    method : str
        Provenance tag carried through to the result.

    Returns
    -------
    info : dict or None
        ``{'mass', 'com', 'inertia', 'method'}`` with ``inertia`` as the
        6 components in the link frame, or None if the inputs are missing
        or non-finite.
    """
    if mass is None or com is None or inertia is None:
        return None
    try:
        mass = float(mass)
        com = np.asarray(com, dtype=float)
        inertia = np.asarray(inertia, dtype=float)
        if inertia.shape == (6,):
            ixx, ixy, ixz, iyy, iyz, izz = inertia
            inertia = np.array([[ixx, ixy, ixz],
                                [ixy, iyy, iyz],
                                [ixz, iyz, izz]], dtype=float)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(mass) and mass > 0 and com.shape == (3,)
            and np.all(np.isfinite(com)) and inertia.shape == (3, 3)
            and np.all(np.isfinite(inertia))):
        return None
    roll, pitch, yaw = visual_rpy
    rot = rpy2matrix(float(roll), float(pitch), float(yaw))
    com_link = rot @ com + np.asarray(visual_xyz, dtype=float)
    tensor = rot @ inertia @ rot.T
    return {
        'mass': mass,
        'com': [float(c) for c in com_link],
        'inertia': (float(tensor[0, 0]), float(tensor[0, 1]),
                    float(tensor[0, 2]), float(tensor[1, 1]),
                    float(tensor[1, 2]), float(tensor[2, 2])),
        'method': method,
    }


def link_inertial_from_mesh(mesh_path, visual_xyz, visual_rpy,
                            density=DEFAULT_DENSITY, scale=1.0):
    """URDF link inertial computed from a mesh, expressed in the link frame.

    Parameters
    ----------
    mesh_path : str or None
        Path to the link's mesh (any format trimesh can load).
    visual_xyz, visual_rpy : sequence of 3 floats
        The link's visual origin (mesh to link), in metres / radians.
    density : float
        Material density in kg/m^3.
    scale : float
        Mesh-unit to metre factor (e.g. 0.001 for millimetre meshes).

    Returns
    -------
    info : dict or None
        ``{'mass', 'com', 'inertia', 'method'}`` where ``inertia`` is
        ``(ixx, ixy, ixz, iyy, iyz, izz)`` and ``method`` is ``'mesh'``,
        ``'hull'`` or ``'bbox'``.  None if no usable geometry was found
        (the caller should keep a placeholder).
    """
    if not mesh_path:
        return None
    props = mesh_mass_properties(mesh_path, density=density, scale=scale)
    if props is None:
        return None
    mass, com, inertia, method = props
    return transform_inertial(mass, com, inertia, visual_xyz, visual_rpy,
                              method=method)


def rescale_inertial_to_mass(info, target_mass):
    """Rescale a computed inertial dict to an exact target mass (kg).

    For a rigid body of fixed geometry, changing the (uniform) density scales
    the mass and the full inertia tensor by the SAME factor while the centre
    of mass is unchanged, so ``mass`` and all six inertia components are
    multiplied by ``target_mass / mass`` and ``com`` is kept.  ``method`` is
    annotated with ``'->mass'`` so provenance reporting shows the rescale.

    Returns a new dict (does not mutate ``info``).  If ``info`` is falsy or
    its mass is non-positive or non-finite, returns ``info`` unchanged.
    """
    if not info:
        return info
    try:
        mass = float(info['mass'])
        target = float(target_mass)
    except (TypeError, ValueError, KeyError):
        return info
    if not (np.isfinite(mass) and mass > 0
            and np.isfinite(target) and target > 0):
        return info
    factor = target / mass
    return {
        'mass': target,
        'com': list(info['com']),
        'inertia': tuple(float(x) * factor for x in info['inertia']),
        'method': '{}->mass'.format(info.get('method', '?')),
    }


def validate_inertia(mass, inertia6, rel_tol=1e-6):
    """Physics sanity-check one link's inertial; return a list of problem
    strings (empty list means OK).

    A real rigid body's inertia tensor (taken about the centre of mass) must
    be symmetric positive definite -- all principal moments (eigenvalues
    ``I1 <= I2 <= I3``) strictly positive -- and those moments must satisfy
    the triangle inequality ``I1 + I2 >= I3`` (the geometric constraint every
    physical mass distribution obeys; the other two combinations follow once
    all are positive).  Violating either makes a simulator's integrator
    diverge, and usually signals a units / frame / transform bug upstream.

    ``rel_tol`` is applied relative to the largest principal moment so
    ordinary floating-point / tessellation noise does not trip the checks.
    """
    problems = []
    if mass is None or not np.isfinite(mass) or mass <= 0:
        problems.append('mass is not positive (= {})'.format(mass))
    try:
        ixx, ixy, ixz, iyy, iyz, izz = (float(x) for x in inertia6)
    except (TypeError, ValueError):
        problems.append('inertia is not a 6-tuple (ixx,ixy,ixz,iyy,iyz,izz)')
        return problems
    tensor = np.array([[ixx, ixy, ixz],
                       [ixy, iyy, iyz],
                       [ixz, iyz, izz]], dtype=float)
    if not np.all(np.isfinite(tensor)):
        problems.append('inertia tensor has non-finite entries')
        return problems
    e = np.linalg.eigvalsh(tensor)            # ascending, real (symmetric)
    atol = rel_tol * max(abs(float(e[-1])), 1e-12)
    if e[0] <= atol:
        problems.append(
            'inertia tensor is not positive definite '
            '(smallest principal moment {:.4g} <= 0)'.format(e[0]))
    elif e[0] + e[1] < e[2] - atol:           # triangle inequality
        problems.append(
            'principal moments violate the triangle inequality '
            '(I1+I2 < I3: {:.4g} + {:.4g} < {:.4g})'.format(e[0], e[1], e[2]))
    return problems

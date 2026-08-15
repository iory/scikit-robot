"""The file side of a URDF: resolving, loading and writing its meshes.

A URDF names its geometry indirectly -- a relative path or a
``package://`` URI -- and every one of those names has to be turned into a
file on disk, read, and (when converting a model) written back out in
another format. That is what this module does, leaving
:mod:`skrobot.utils.urdf` to the XML types themselves.

It is organised in the order a mesh travels:

* the export settings, which are module-global because they have to reach
  the ``_to_xml`` methods of types that cannot take extra arguments, and
  the context managers that scope them;
* resolving a URDF's file references, including ROS package lookup;
* loading meshes, with the caches and the vertex-normal handling;
* deciding what an element's mesh is written to, and writing it.

The one rule the last part exists to keep is that **an output mesh file
only ever holds geometry that matches it**: an element whose vertices were
changed -- by baking its origin in, or by simplification -- may not be
written under the name of the file it was read from, or under a name
another element with different geometry is using.
"""

import contextlib
import hashlib
from logging import getLogger
import os
import pickle
import re
import sys

from lxml import etree as ET
import numpy as np

from skrobot._lazy_imports import _lazy_trimesh
from skrobot.data import get_cache_dir
from skrobot.pycompat import lru_cache
from skrobot.utils.checksum import checksum_md5


try:
    # for python3
    from urllib.parse import urlparse
except ImportError:
    # for python2
    from urlparse import urlparse

try:
    import rospkg
except ImportError:
    rospkg = None


logger = getLogger(__name__)

# Whether the one-time "install DracoPy" hint has already been emitted.
# A scene may reference many Draco-compressed meshes; we warn per file but
# only show the verbose installation instruction once per process.
_DRACO_MISSING_HINT_SHOWN = False

_CONFIGURABLE_VALUES = {"mesh_simplify_factor": np.inf,
                        'no_mesh_load_mode': False,
                        'export_mesh_format': None,
                        'collision_mesh_format': None,
                        'decimation_area_ratio_threshold': None,
                        'simplify_vertex_clustering_voxel_size': None,
                        'target_triangles': None,
                        'enable_mesh_cache': False,
                        'force_visual_mesh_origin_to_zero': False,
                        'overwrite_mesh': False,
                        'scale_factor': 1.0,
                        'blender_remesh': False,
                        'blender_voxel_size': 0.002,
                        'blender_decimate': False,
                        'blender_decimate_ratio': 0.1,
                        'blender_executable': None,
                        '_current_geometry_context': None,  # 'collision' or 'visual'
                        '_source_urdf_path': None,  # Original URDF file path for mesh resolution
                        }
_MESH_CACHE = {}
_REMESHED_FILES_CACHE = {}  # Cache to track which files have been remeshed
# Mesh files written by the save currently in progress, mapping the
# normalized output path to the geometry context ('visual' or 'collision')
# that wrote it. Scoped by :func:`_export_session`.
_EXPORTED_MESH_FILES = {}


@contextlib.contextmanager
def _configured(**values):
    """Set ``_CONFIGURABLE_VALUES`` entries for the duration of a block.

    Every public context manager in this module is a thin wrapper around
    this one, so that all of them restore state the same way: on the way
    out, whatever the block did, and back to the value that was in force
    on the way in rather than to a hard-coded default.

    Both properties matter. A block that leaks its settings poisons the
    rest of the process -- a batch converter that fails on one URDF would
    convert the next one with the failed run's mesh format, scale and
    origin baking still switched on, and a viewer opened afterwards would
    load every model through them. Restoring the previous value (instead
    of the default) additionally makes the managers nest.

    Parameters
    ----------
    values : dict
        ``_CONFIGURABLE_VALUES`` keys to set for the duration of the
        block. A key that did not exist before is removed again.
    """
    missing = object()
    previous = {key: _CONFIGURABLE_VALUES.get(key, missing) for key in values}
    _CONFIGURABLE_VALUES.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                _CONFIGURABLE_VALUES.pop(key, None)
            else:
                _CONFIGURABLE_VALUES[key] = value


@contextlib.contextmanager
def mesh_simplify_factor(factor):
    with _configured(mesh_simplify_factor=factor):
        yield


@contextlib.contextmanager
def no_mesh_load_mode():
    with _configured(no_mesh_load_mode=True):
        yield


@contextlib.contextmanager
def export_mesh_format(
        mesh_format,
        decimation_area_ratio_threshold=None,
        simplify_vertex_clustering_voxel_size=None,
        target_triangles=None,
        overwrite_mesh=False,
        collision_mesh_format=None,
        blender_remesh=False,
        blender_voxel_size=0.002,
        blender_decimate=False,
        blender_decimate_ratio=0.1,
        blender_executable=None,
        remeshed_suffix='_remeshed',
        draco_compression=False):
    with _configured(
            export_mesh_format=mesh_format,
            collision_mesh_format=collision_mesh_format,
            decimation_area_ratio_threshold=decimation_area_ratio_threshold,
            simplify_vertex_clustering_voxel_size=(
                simplify_vertex_clustering_voxel_size),
            target_triangles=target_triangles,
            overwrite_mesh=overwrite_mesh,
            blender_remesh=blender_remesh,
            blender_voxel_size=blender_voxel_size,
            blender_decimate=blender_decimate,
            blender_decimate_ratio=blender_decimate_ratio,
            blender_executable=blender_executable,
            remeshed_suffix=remeshed_suffix,
            draco_compression=draco_compression):
        yield


@contextlib.contextmanager
def _export_session():
    """Scope the record of what one :meth:`URDF.save` has already written.

    Both caches say "this file is already the geometry we were about to
    write", which is only true of the save that wrote it. Letting either
    outlive its save makes the next one skip meshes that are not there
    yet -- and a module-global cache that nothing resets is how a
    second ``URDF.load`` in one process used to come out collapsed.
    """
    _REMESHED_FILES_CACHE.clear()
    _EXPORTED_MESH_FILES.clear()
    try:
        yield
    finally:
        _REMESHED_FILES_CACHE.clear()
        _EXPORTED_MESH_FILES.clear()


@contextlib.contextmanager
def enable_mesh_cache():
    with _configured(enable_mesh_cache=True):
        yield


@contextlib.contextmanager
def force_visual_mesh_origin_to_zero():
    with _configured(force_visual_mesh_origin_to_zero=True):
        yield


def bake_origin_into_meshes(geometry, origin):
    """Bake ``origin`` into the vertices of a geometry's meshes.

    Used by :func:`force_visual_mesh_origin_to_zero`, which zeroes an
    element's ``<origin>`` and moves that offset into the geometry itself.

    The meshes are copied before being transformed, because one
    :class:`~trimesh.base.Trimesh` may be shared by several elements: the
    same file referenced by a ``<visual>`` and a ``<collision>``, or by two
    links, and under :func:`enable_mesh_cache` every such load returns the
    one cached list. Transforming in place would bake one element's origin
    into another element's geometry, or bake both origins into a single
    mesh.

    Parameters
    ----------
    geometry : :class:`.Geometry`
        Geometry whose meshes are baked. Only a ``<mesh>`` geometry is
        touched; a primitive carries no vertices to bake into and is left
        alone. Its ``mesh.meshes`` is replaced with the transformed copies,
        and ``mesh.baked_origin`` records what was baked -- the geometry no
        longer matches the file it was loaded from, so saving the URDF has
        to give it a file of its own (see
        :func:`resolve_mesh_output_path`).
    origin : (4, 4) float
        Pose to bake into the vertices. Treated as identity within
        ``1e-8``, in which case nothing is copied or transformed.
    """
    if geometry.mesh is None or np.allclose(origin, np.eye(4)):
        # No vertices to bake into, or an offset small enough that baking it
        # would only cost a copy of possibly shared meshes.
        return
    baked_meshes = []
    for mesh in geometry.mesh.meshes:
        baked_mesh = mesh.copy()
        baked_mesh.apply_transform(origin)
        # copy() deep-copies the material, including its texture image; the
        # material is not what the transform changes, so hand the copy back
        # the original rather than a per-element duplicate of the image
        visual, original = baked_mesh.visual, mesh.visual
        if hasattr(visual, 'material') and hasattr(original, 'material'):
            visual.material = original.material
        baked_meshes.append(baked_mesh)
    geometry.mesh.meshes = baked_meshes
    geometry.mesh.baked_origin = np.array(origin, dtype=np.float64)


@contextlib.contextmanager
def apply_scale(scale_factor):
    with _configured(scale_factor=scale_factor):
        yield


@contextlib.contextmanager
def source_urdf_path(path):
    """Resolve mesh filenames against ``path`` while saving a URDF.

    ``URDF.save`` normally resolves relative mesh paths against the
    output location; inside this context it resolves them against the
    directory the URDF was loaded from instead.

    The previous value is restored on exit. Leaving it set leaks into
    every later export in the same process -- meshes then resolve
    against a stale directory.

    Parameters
    ----------
    path : str
        Directory the URDF was loaded from.
    """
    with _configured(_source_urdf_path=path):
        yield


def _try_ament(ros_package):
    """Resolve via ament_index_python. Returns path or None if unavailable / not found."""
    try:
        from ament_index_python.packages import get_package_share_directory
        from ament_index_python.packages import PackageNotFoundError
    except ImportError:
        return None
    try:
        return get_package_share_directory(ros_package)
    except PackageNotFoundError:
        return None
    except Exception as e:
        logger.warning(
            "ament_index lookup for ROS package '%s' failed: %s",
            ros_package, e)
        return None


def _try_rospkg(ros_package):
    """Resolve via rospkg. Returns path or None if unavailable / not found."""
    if rospkg is None:
        return None
    try:
        return rospkg.RosPack().get_path(ros_package)
    except rospkg.common.ResourceNotFound:
        return None


def _manifest_package_name(directory):
    """Return the ``<name>`` declared in ``<directory>/package.xml``, or None."""
    manifest = os.path.join(directory, 'package.xml')
    if not os.path.isfile(manifest):
        return None
    try:
        name = ET.parse(manifest).getroot().findtext('name')
    except (OSError, ET.XMLSyntaxError):
        return None
    return name.strip() if name else None


def _env_paths(variable):
    """Absolute, expanded, non-empty entries of an ``os.pathsep`` env variable."""
    return [os.path.abspath(os.path.expanduser(p))
            for p in os.environ.get(variable, '').split(os.pathsep) if p]


def _find_package_dir(root, ros_package):
    """Locate ``ros_package`` under one ``ROS_PACKAGE_PATH`` entry.

    An entry may itself be the package, a directory that contains it, or a
    workspace ``src`` to crawl (the several catkin layouts). The recursive walk
    is the last resort and prunes build/VCS trees.
    """
    if _manifest_package_name(root) == ros_package:
        return root
    direct = os.path.join(root, ros_package)
    if _manifest_package_name(direct) == ros_package:
        return direct
    if not os.path.isdir(root):
        return None
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs
                   if not d.startswith('.')
                   and d not in ('build', 'devel', 'install', 'log')]
        if 'package.xml' in files:
            if _manifest_package_name(current) == ros_package:
                return current
            # A catkin package cannot contain another; stop descending here
            # (also avoids crawling large mesh/resource trees).
            dirs[:] = []
    return None


def _try_env_prefixes(ros_package):
    """Resolve a ROS package from the sourced shell environment alone.

    Uses only the standard search-path variables, so it works when neither
    ``ament_index_python`` nor ``rospkg`` is importable -- e.g. inside a
    PyInstaller-frozen binary launched from a sourced ROS workspace. Returns
    the package's share/root directory, or ``None``.

    * ``AMENT_PREFIX_PATH`` / ``COLCON_PREFIX_PATH`` / ``CMAKE_PREFIX_PATH``:
      ``<prefix>/share/<pkg>`` (the ament / ROS 2 install layout).
    * ``ROS_PACKAGE_PATH``: each entry is the package, a directory of packages,
      or a workspace ``src`` to crawl (the catkin / ROS 1 layouts).
    """
    for variable in ('AMENT_PREFIX_PATH', 'COLCON_PREFIX_PATH',
                     'CMAKE_PREFIX_PATH'):
        for prefix in _env_paths(variable):
            share = os.path.join(prefix, 'share', ros_package)
            if _manifest_package_name(share) == ros_package:
                return share
    for root in _env_paths('ROS_PACKAGE_PATH'):
        found = _find_package_dir(root, ros_package)
        if found:
            return found
    return None


@lru_cache(maxsize=None)
def get_path_with_cache(ros_package):
    """Resolve a ROS package name to its share/install directory.

    Tries ``ament_index_python`` (ROS 2) and ``rospkg`` (ROS 1) when available,
    then falls back to :func:`_try_env_prefixes`, which resolves straight from
    the sourced shell environment and needs no ROS Python at all. The order of
    the first two respects the ``ROS_VERSION`` environment variable so that
    users with ROS 1 and ROS 2 coexisting on the same machine get the resolver
    matching their currently-sourced distro:

    * ``ROS_VERSION=1`` -> rospkg first, then ament (preserves the old
      behaviour where rospkg-only resolution was used).
    * ``ROS_VERSION=2`` or unset -> ament first, then rospkg. This is what
      makes the function work on plain ``ros-<distro>-desktop`` installs that
      ship ament but not rospkg (previously such environments hit a
      ``TypeError`` deeper in mesh loading).

    The environment fallback runs last in both orders, so it only changes the
    outcome when the ROS Python resolvers are absent or come up empty -- e.g. a
    PyInstaller-frozen binary launched from a sourced workspace.

    Results are cached for the process lifetime (``lru_cache``); a change to the
    ROS search-path environment variables after the first lookup is not picked
    up, matching the existing behaviour for the ament/rospkg resolvers.

    Raises
    ------
    ImportError
        No ROS Python resolver is installed and the package was not found under
        the ROS search-path environment variables either.
    LookupError
        The resolvers were tried but none of them found the package.
    """
    # ament_index / rospkg are the authoritative package indices when present;
    # _try_env_prefixes is a dependency-free fallback that resolves straight
    # from the sourced shell environment, so a frozen binary (no ROS Python)
    # still finds meshes after `source install/setup.bash`.
    if os.environ.get('ROS_VERSION') == '1':
        resolvers = (_try_rospkg, _try_ament, _try_env_prefixes)
    else:
        resolvers = (_try_ament, _try_rospkg, _try_env_prefixes)

    for resolver in resolvers:
        path = resolver(ros_package)
        if path is not None:
            return path

    # Distinguish "neither resolver installed" from "package missing".
    ament_available = True
    try:
        import ament_index_python  # noqa: F401
    except ImportError:
        ament_available = False
    if not ament_available and rospkg is None:
        raise ImportError(
            "Cannot resolve ROS package '{}': neither ament_index_python "
            "nor rospkg is installed, and it was not found under "
            "AMENT_PREFIX_PATH / ROS_PACKAGE_PATH / CMAKE_PREFIX_PATH. "
            "Source your ROS workspace (e.g. `source install/setup.bash`) "
            "or install one of the resolvers.".format(ros_package))
    raise LookupError(
        "ROS package '{}' was not found by ament_index_python or rospkg. "
        "Did you forget to source your ROS workspace "
        "(e.g. `source install/setup.bash` for ROS 2)?".format(ros_package))


def search_up(start_dir, relative_path):
    current_dir = start_dir
    while True:
        candidate = os.path.join(current_dir, relative_path)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            return None
        current_dir = parent


def resolve_filepath(base_path, file_path):
    if os.path.isabs(file_path):
        if os.path.exists(file_path):
            return os.path.normpath(file_path)
        return None

    parsed_url = urlparse(file_path)
    base_path = os.path.abspath(base_path)

    if parsed_url.scheme == 'package':
        try:
            ros_package = parsed_url.netloc
            package_path = get_path_with_cache(ros_package)
            pkg_relative_path = parsed_url.path.lstrip("/")
            resolved_filepath = os.path.join(package_path, pkg_relative_path)
            if os.path.exists(resolved_filepath):
                return os.path.normpath(resolved_filepath)
        except Exception:
            # Catches ament's PackageNotFoundError, rospkg's ResourceNotFound,
            # and the ImportError raised by get_path_with_cache when neither
            # resolver is installed. We fall through to the search_up()
            # heuristic below so behaviour stays graceful for pure-filesystem
            # URDFs that just happen to use package:// for documentation.
            pass

    rel_paths = [
        os.path.join(parsed_url.netloc, parsed_url.path.lstrip('/')),
        parsed_url.path.lstrip('/')
    ]
    for rel in rel_paths:
        found_path = search_up(base_path, rel)
        if found_path:
            return os.path.normpath(found_path)
    return None


def get_filename(base_path, file_path, makedirs=False):
    """Formats a file path correctly for URDF loading.

    Parameters
    ----------
    base_path : str
        The base path to the URDF's folder.
    file_path : str
        The path to the file.
    makedirs : bool, optional
        If ``True``, the directories leading to the file will be created
        if needed.

    Returns
    -------
    resolved : str
        The resolved filepath -- just the normal ``file_path`` if it was an
        absolute path, otherwise that path joined to ``base_path``.
    """
    resolved_file_path = resolve_filepath(base_path, file_path)
    if resolved_file_path is None:
        logger.error('could not find %s', file_path)
        return None
    if not os.path.isabs(resolved_file_path):
        resolved_file_path = os.path.join(base_path, resolved_file_path)
    if makedirs:
        d, _ = os.path.split(resolved_file_path)
        if not os.path.exists(d):
            os.makedirs(d)
    return resolved_file_path


def resolve_simplified_mesh_path(filename, simplify_factor):
    hash_value = checksum_md5(filename)
    cache_base_path = os.path.join(get_cache_dir(), "simplified_mesh")

    if not os.path.exists(cache_base_path):
        os.makedirs(cache_base_path)

    rounded_simplify_factor = round(simplify_factor, 3)
    cache_path = os.path.join(cache_base_path, "{}-{}.pkl".format(
        hash_value, rounded_simplify_factor))
    return cache_path


def load_meshes(filename):
    enable_mesh_cache = _CONFIGURABLE_VALUES['enable_mesh_cache']
    if enable_mesh_cache:
        if filename in _MESH_CACHE:
            return _MESH_CACHE[filename]
    use_simplified_meshs = _CONFIGURABLE_VALUES["mesh_simplify_factor"] < 1.0

    if use_simplified_meshs:
        cache_path = resolve_simplified_mesh_path(
            filename, _CONFIGURABLE_VALUES["mesh_simplify_factor"])

        if os.path.exists(cache_path):
            with open(cache_path, mode="rb") as f:
                return pickle.load(f)
        else:
            meshes = _load_meshes(filename)

            assert sys.version_info.major > 2, "supported only for python3.x"
            try:
                import open3d  # noqa
            except ImportError:
                message = "to simplify mesh, you need to install open3d\n"
                message += "run 'pip install open3d'"
                raise ImportError(message)

            mesh_simplified_list = []
            for mesh in meshes:
                n_face = len(mesh.faces)
                n_face_reduced = int(
                    n_face * _CONFIGURABLE_VALUES["mesh_simplify_factor"])
                mesh_simplified = mesh.simplify_quadratic_decimation(
                    n_face_reduced)
                mesh_simplified_list.append(mesh_simplified)

            with open(cache_path, mode="wb") as f:
                pickle.dump(mesh_simplified_list, f)
            return mesh_simplified_list
    else:
        meshes = _load_meshes(filename)
        if enable_mesh_cache:
            _MESH_CACHE[filename] = meshes
        return meshes


def _gltf_uses_draco(filename):
    """Return whether a glTF/GLB file uses Draco mesh compression.

    A glTF/GLB file that lists ``KHR_draco_mesh_compression`` in its
    ``extensionsUsed`` or ``extensionsRequired`` cannot be decoded without
    DracoPy.  trimesh silently returns degenerate (all-zero) geometry in
    that case rather than raising, so we detect it explicitly.

    Parameters
    ----------
    filename : str
        Path to a ``.glb`` or ``.gltf`` file.

    Returns
    -------
    uses_draco : bool
        ``True`` if the file references the Draco extension, ``False``
        otherwise (including when the file cannot be parsed).
    """
    import json
    import struct

    ext_name = 'KHR_draco_mesh_compression'
    try:
        _, ext = os.path.splitext(filename)
        if ext.lower() == '.glb':
            with open(filename, 'rb') as f:
                header = f.read(12)
                if len(header) < 12 or header[:4] != b'glTF':
                    return False
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    return False
                chunk_length, chunk_type = struct.unpack('<II', chunk_header)
                if chunk_type != 0x4E4F534A:  # 'JSON'
                    return False
                gltf = json.loads(f.read(chunk_length))
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                gltf = json.load(f)
    except Exception:
        return False

    extensions = set(gltf.get('extensionsUsed', []) or [])
    extensions.update(gltf.get('extensionsRequired', []) or [])
    return ext_name in extensions


def _load_meshes(filename):
    """Loads triangular meshes from a file.

    Parameters
    ----------
    filename : str
        Path to the mesh file.

    Returns
    -------
    meshes : list of :class:`~trimesh.base.Trimesh`
        The meshes loaded from the file.
    """
    if filename is None:
        raise FileNotFoundError(
            "Cannot load mesh: file path is None. This usually means a "
            "package:// URI in the URDF could not be resolved. Make sure the "
            "referenced ROS package is installed and your environment is "
            "sourced (`source install/setup.bash` for ROS 2, or set "
            "`ROS_PACKAGE_PATH` for ROS 1).")
    # Imported here, not at module level: skrobot.utils.mesh pulls in
    # trimesh, and importing this module must not (see _lazy_trimesh).
    from skrobot.utils.mesh import _authored_vertex_normals
    from skrobot.utils.mesh import _dump_scene
    from skrobot.utils.mesh import _restore_vertex_normals
    from skrobot.utils.mesh import get_transparency

    trimesh = _lazy_trimesh()
    _, ext = os.path.splitext(filename)
    is_glb_or_gltf = ext.lower() in ('.glb', '.gltf')
    dracopy_available = False

    # Register DracoPy handlers for Draco decompression of GLB/GLTF
    if is_glb_or_gltf:
        from skrobot.utils.draco import is_dracopy_available
        from skrobot.utils.draco import register_dracopy_handlers
        dracopy_available = is_dracopy_available()
        if dracopy_available:
            register_dracopy_handlers()

    # A Draco-compressed glTF/GLB cannot be decoded without DracoPy.  In
    # that case trimesh does not raise; it silently returns degenerate
    # (all-zero) geometry.  Detect the situation up front and skip the mesh
    # with a clear warning, instead of letting the broken geometry pass as a
    # successful load.  We deliberately do NOT raise here: a single Draco
    # mesh would otherwise abort the whole URDF load (and trigger a confusing
    # "load as URDF string" fallback).  Skipping lets the rest of the model
    # load while keeping the missing geometry visible in the logs.
    #
    # NOTE: PR #715 already added a "pip install DracoPy" hint, but it only
    # fired when trimesh.load() raised or returned an empty mesh list.  For
    # the urdfeus Draco models trimesh does neither -- it returns a single
    # all-zero mesh -- so the hint never triggered and nothing was reported.
    # Parsing the extension list here covers that silent-degenerate case
    # regardless of how trimesh behaves.
    if is_glb_or_gltf and not dracopy_available and _gltf_uses_draco(filename):
        global _DRACO_MISSING_HINT_SHOWN
        if not _DRACO_MISSING_HINT_SHOWN:
            logger.error(
                "DracoPy is not installed, so Draco-compressed glTF/GLB "
                "meshes (KHR_draco_mesh_compression) cannot be decoded and "
                "will be skipped; trimesh would otherwise return empty "
                "all-zero geometry. Install DracoPy with: pip install DracoPy")
            _DRACO_MISSING_HINT_SHOWN = True
        logger.warning(
            "Skipping Draco-compressed mesh (DracoPy not installed): %s",
            filename)
        return []

    load_error_reported = False
    try:
        # It seems that .3DXML files assume [mm] unit.
        # Convert the mesh unit from [mm] to [m].
        # To convert the mesh unit from millimeters to meters,
        # use the function meshes.convert_units('meter').
        meshes = trimesh.load(filename)
        # Keep whatever normals the file stored: they encode which edges the
        # author meant to be smooth, and nothing downstream can recover that
        # once they have been dropped.
        authored_normals = _authored_vertex_normals(meshes)
        if meshes.units is not None and meshes.units != 'meter':
            meshes = meshes.convert_units('meter')
            _restore_vertex_normals(meshes, authored_normals)
    except Exception as e:
        if is_glb_or_gltf and not dracopy_available:
            logger.error(
                "Failed to load mesh from %s: %s. "
                "This file may use Draco compression. "
                "Install DracoPy with: pip install DracoPy", filename, e)
        else:
            logger.error("Failed to load meshes from %s. Error: %s", filename, e)
        load_error_reported = True
        meshes = []

    # If we got a scene, dump the meshes
    if isinstance(meshes, trimesh.Scene):
        meshes = _dump_scene(meshes)

    if isinstance(meshes, (list, tuple, set)):
        meshes = list(meshes)
        if len(meshes) == 0:
            if not load_error_reported:
                if is_glb_or_gltf and not dracopy_available:
                    logger.error(
                        "Failed to load mesh from %s. "
                        "This file may use Draco compression. "
                        "Install DracoPy with: pip install DracoPy", filename)
                else:
                    logger.error('At least one mesh must be present in file.'
                                 ' Please check %s file', filename)
            meshes = []
        for r in meshes:
            if not isinstance(r, trimesh.Trimesh):
                raise TypeError('Could not load meshes from file {}'.
                                format(filename))
    elif isinstance(meshes, trimesh.Trimesh):
        meshes = [meshes]
    else:
        logger.error('Unable to load mesh from file %s', filename)
        meshes = []

    for mesh in meshes:
        transparency = get_transparency(mesh)
        if transparency is not None and transparency == 0.0:
            if isinstance(mesh.visual.material,
                          trimesh.visual.material.PBRMaterial):
                mesh.visual.material.baseColorFactor[3] = 255
    return meshes


def _replace_extension(filename, suffix, ext):
    """Insert ``suffix`` before, and replace, a filename's extension.

    Works on a plain path and on a ``package://`` URI alike, since both end
    in ``<stem>.<ext>``.

    Parameters
    ----------
    filename : str
        Path or URI to rewrite.
    suffix : str
        Inserted between the stem and the extension. May be empty.
    ext : str or None
        Replacement extension, leading dot included. ``None`` keeps the
        current one.

    Returns
    -------
    str
        The rewritten path or URI.
    """
    stem, current_ext = os.path.splitext(filename)
    return stem + suffix + (current_ext if ext is None else ext)


def geometry_key(baked_origin):
    """A stable identity for the geometry an element ends up holding.

    ``force_visual_mesh_origin_to_zero`` bakes each element's ``<origin>``
    into its own copy of the mesh, so elements sharing one mesh file end up
    with *different* geometry and can no longer share one output file. This
    is what "different" means: a hash of the baked pose, equal for two
    elements that baked the same origin and for the same element across
    runs.

    Parameters
    ----------
    baked_origin : (4, 4) float or None
        The pose baked into the vertices, or ``None`` if the geometry is
        still the file's own.

    Returns
    -------
    str
        ``''`` for unbaked geometry, otherwise eight hex digits.
    """
    if baked_origin is None:
        return ''
    origin = np.round(np.asarray(baked_origin, dtype=np.float64), 9)
    # -0.0 and 0.0 compare equal but have different bytes; without this two
    # elements baking the same origin would look different.
    origin[origin == 0] = 0.0
    return hashlib.md5(origin.tobytes()).hexdigest()[:8]


def element_output_name(link_name, context, index, count):
    """Name the mesh file an element of a link gets to itself.

    Only used once an element's geometry stops matching the file it was
    read from, which is when it needs a file of its own (see
    :func:`resolve_mesh_output_path`). Naming it after the element is what
    makes the result readable -- ``leg_rleg.dae`` and ``leg_lleg.dae``
    rather than two hashes -- so that whoever opens the output directory
    can tell which mesh belongs where.

    Parameters
    ----------
    link_name : str
        Name of the link the element belongs to.
    context : str
        ``'visual'`` or ``'collision'``.
    index : int
        Position of the element among its link's elements of that kind.
    count : int
        How many the link has. The index is left out of the name when
        there is only one, which is the common case.

    Returns
    -------
    str
        A name safe to put in a filename.
    """
    name = link_name
    if context == 'collision':
        name += '_collision'
    if count > 1:
        name += '_{}'.format(index)
    # A URDF link name is free-form text; a mesh filename is not.
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)


def mesh_variant_suffix(baked_origin, element_name):
    """Filename suffix telling one element's geometry from another's.

    Named after the element when the caller knows which one it is, and
    after the geometry itself when it does not -- a :class:`.Mesh` built by
    hand rather than parsed as part of a link, or two link names that a
    filename cannot tell apart.

    Parameters
    ----------
    baked_origin : (4, 4) float or None
        The pose baked into this element's vertices, if any.
    element_name : str or None
        :func:`element_output_name` for the element, when known.

    Returns
    -------
    str
        ``''`` when the geometry still matches its file, otherwise ``'_'``
        and a name.
    """
    if baked_origin is None:
        return ''
    if element_name:
        return '_' + element_name
    return '_' + geometry_key(baked_origin)


def mesh_processing_key():
    """Filename suffix identifying the configured mesh processing.

    Decimation, vertex clustering and the Blender modifiers change the
    geometry for the whole export, so unlike a baked origin they do not
    tell two elements apart. What they do tell apart is the processed mesh
    from the file it was read out of -- which matters when the target
    format equals the source format and the natural output path is the
    input file itself.

    The key is a hash of the settings, so re-running with the same
    settings reuses the file and re-running with different ones does not
    quietly inherit the previous result.

    Returns
    -------
    str
        ``''`` when nothing is configured, otherwise ``'_'`` and eight hex
        digits.
    """
    settings = [_CONFIGURABLE_VALUES.get(key) for key in (
        'target_triangles',
        'decimation_area_ratio_threshold',
        'simplify_vertex_clustering_voxel_size',
        'blender_decimate',
        'blender_decimate_ratio')]
    if not any(settings[:4]):
        return ''
    digest = hashlib.md5(repr(settings).encode('utf-8')).hexdigest()
    return '_' + digest[:8]


def resolve_mesh_output_path(source_file, urdf_filename, ext, variant_suffix,
                             processing_key=''):
    """Decide where an element's mesh is written, and how the URDF names it.

    The single place that maps an element to an output mesh file. Every
    bug in which two elements silently shared one file, or in which a
    processed mesh landed on top of its own source, was a disagreement
    about this mapping, so it holds the invariant those bugs broke:

        an output file only ever holds geometry that matches it.

    Two things can make an element's geometry differ from the file it was
    read from, and each contributes to the name. A baked origin differs per
    element, so it always does -- that is ``variant_suffix``, which
    :func:`mesh_variant_suffix` takes from the element's own name. The
    configured processing is the same for every element of an export, so it
    only has to contribute when the output would otherwise be the source
    file itself; passing an empty ``processing_key`` leaves it out to allow
    exactly that.

    Parameters
    ----------
    source_file : str
        Resolved path of the mesh file the element was loaded from.
    urdf_filename : str
        The element's ``filename`` attribute, as it appears in the URDF --
        a relative path or a ``package://`` URI.
    ext : str or None
        Target extension, leading dot included. ``None`` keeps the source
        format.
    variant_suffix : str
        :func:`mesh_variant_suffix` for this element -- ``''`` while its
        geometry still matches ``source_file``.
    processing_key : str
        :func:`mesh_processing_key`, or ``''`` to let a processed mesh
        replace its source.

    Returns
    -------
    output_file : str
        Path to write the mesh to.
    output_urdf_filename : str
        What the saved URDF should call it.
    """
    suffix = variant_suffix
    output_file = _replace_extension(source_file, suffix, ext)
    if processing_key and os.path.normpath(output_file) == os.path.normpath(
            source_file):
        suffix += processing_key
        output_file = _replace_extension(source_file, suffix, ext)
    return output_file, _replace_extension(urdf_filename, suffix, ext)


def claim_mesh_output_path(source_file, urdf_filename, ext, variant_suffix,
                           geometry, processing_key=''):
    """Resolve an element's output path and record that it owns it.

    :func:`resolve_mesh_output_path` names a file after the element, which
    reads far better than naming it after the geometry but no longer makes
    two different geometries impossible to land on one name: a link name
    goes through :func:`element_output_name` on the way into a filename,
    and two of them can come out the same. This is where that is caught --
    a path already claimed by *different* geometry falls back to a name
    derived from the geometry itself, which cannot collide.

    Parameters
    ----------
    source_file, urdf_filename, ext, variant_suffix, processing_key
        As for :func:`resolve_mesh_output_path`.
    geometry : str
        :func:`geometry_key` for this element.

    Returns
    -------
    output_file : str
        Path to write the mesh to.
    output_urdf_filename : str
        What the saved URDF should call it.
    """
    output_file, output_urdf_filename = resolve_mesh_output_path(
        source_file, urdf_filename, ext, variant_suffix, processing_key)
    claimed = _EXPORTED_MESH_FILES.get(os.path.normpath(output_file))
    if claimed is not None and claimed[1] != geometry and variant_suffix:
        logger.warning(
            'Two elements with different geometry both want %s; falling '
            'back to a name taken from the geometry itself.', output_file)
        output_file, output_urdf_filename = resolve_mesh_output_path(
            source_file, urdf_filename, ext, '_' + geometry, processing_key)
    return output_file, output_urdf_filename


def _should_skip_mesh_export(output_file, context, geometry):
    """Whether an element's mesh file is already what it has to be.

    * Another element of this save already wrote the file. One mesh shared
      by several links needs writing once. A ``<collision>`` additionally
      must not overwrite what a ``<visual>`` wrote, because the collision
      export concatenates the meshes and drops their materials -- it would
      strip the colors off the visual mesh. The reverse is an upgrade and
      is allowed.
    * The file already exists from an earlier run and ``overwrite_mesh``
      was not given. This one only holds for a mesh that still matches the
      file it was read from, since only then does an existing file's name
      say what is in it. Geometry that was baked lands in a file named
      after the element, which says nothing about which origin went into
      it, so it is always written: skipping would leave the output of an
      earlier run, with a different origin baked in, standing.

    Parameters
    ----------
    output_file : str
        Path the element would be written to.
    context : str or None
        ``'visual'`` or ``'collision'``.
    geometry : str
        :func:`geometry_key` for this element, ``''`` if it still matches
        its source file.

    Returns
    -------
    bool
        True if the write must be skipped.
    """
    key = os.path.normpath(output_file)
    claimed = _EXPORTED_MESH_FILES.get(key)
    if claimed is not None:
        return not (claimed[0] == 'collision' and context == 'visual')
    if geometry:
        return False
    return (not _CONFIGURABLE_VALUES['overwrite_mesh']
            and os.path.exists(key))


def _apply_mesh_processing(meshes, blender_remesh_applied):
    """Run the configured geometry-reducing steps over ``meshes``.

    Parameters
    ----------
    meshes : list of :class:`~trimesh.base.Trimesh`
        Meshes to process.
    blender_remesh_applied : bool
        Whether Blender already remeshed them. Its voxel remesher and the
        decimate modifier are alternatives, not a pipeline.

    Returns
    -------
    list of :class:`~trimesh.base.Trimesh`
        The processed meshes, or the originals if nothing is configured.
    """
    if _CONFIGURABLE_VALUES['blender_decimate'] and not blender_remesh_applied:
        from skrobot.utils.blender_mesh import decimate_with_blender
        meshes = decimate_with_blender(
            meshes,
            ratio=_CONFIGURABLE_VALUES['blender_decimate_ratio'],
            blender_executable=_CONFIGURABLE_VALUES['blender_executable'],
            verbose=True,
        )

    if _CONFIGURABLE_VALUES['target_triangles'] is not None:
        from skrobot.utils.mesh import auto_simplify_quadric_decimation_with_texture_preservation  # NOQA
        meshes = auto_simplify_quadric_decimation_with_texture_preservation(
            meshes,
            target_number_of_triangles=_CONFIGURABLE_VALUES[
                'target_triangles'],
            verbose=True)
    elif _CONFIGURABLE_VALUES['decimation_area_ratio_threshold'] is not None:
        from skrobot.utils.mesh import auto_simplify_quadric_decimation
        meshes = auto_simplify_quadric_decimation(
            meshes, area_ratio_threshold=_CONFIGURABLE_VALUES[
                'decimation_area_ratio_threshold'])

    if _CONFIGURABLE_VALUES['simplify_vertex_clustering_voxel_size']:
        from skrobot.utils.mesh import simplify_vertex_clustering
        meshes = simplify_vertex_clustering(
            meshes,
            _CONFIGURABLE_VALUES['simplify_vertex_clustering_voxel_size'],
        )
    return meshes


def _write_collada(meshes, output_file):
    """Write ``meshes`` as a Collada file, one geometry per face color."""
    from skrobot.utils.mesh import split_mesh_by_face_color

    trimesh = _lazy_trimesh()
    export_meshes = []
    for mesh in meshes:
        # STL files load with visual.defined=False even though they have
        # default face colors. Explicitly setting face_colors makes
        # visual.defined=True and visual.kind='face', which allows
        # trimesh's export_collada to use the colors.
        if not mesh.visual.defined and hasattr(mesh.visual, 'face_colors'):
            mesh.visual.face_colors = mesh.visual.face_colors
        export_meshes.extend(split_mesh_by_face_color(mesh))
    with open(output_file, 'wb') as f:
        f.write(trimesh.exchange.dae.export_collada(export_meshes))


def _write_gltf(meshes, output_file):
    """Write ``meshes`` as glTF/GLB, keeping colors that carry no image.

    Many DAE files store a per-material diffuse color as a texture visual
    with no image; without converting it to vertex colors the color is
    lost on the way into glTF.
    """
    from skrobot.utils.mesh import _material_color_to_vertex_colors

    trimesh = _lazy_trimesh()
    for mesh in meshes:
        if mesh.visual.kind != 'texture':
            continue
        material = getattr(mesh.visual, 'material', None)
        has_image = material is not None and (
            getattr(material, 'image', None) is not None
            or getattr(material, 'baseColorTexture', None) is not None)
        if has_image:
            continue
        try:
            color_visual = mesh.visual.to_color()
            vertex_colors = np.asarray(color_visual.vertex_colors)
            if len(vertex_colors) != len(mesh.vertices):
                color_visual = trimesh.visual.ColorVisuals(
                    mesh=mesh,
                    vertex_colors=_material_color_to_vertex_colors(
                        mesh, material))
            mesh.visual = color_visual
        except Exception:
            pass

    if _CONFIGURABLE_VALUES.get('draco_compression', False):
        from skrobot.utils.draco import export_glb_with_draco
        export_glb_with_draco(meshes, output_file)
        return
    scene = trimesh.Scene()
    for mesh in meshes:
        scene.add_geometry(mesh)
    scene.export(output_file)


def _write_meshes(meshes, output_file):
    """Write ``meshes`` to ``output_file``, in the format its name asks for.

    Parameters
    ----------
    meshes : list of :class:`~trimesh.base.Trimesh`
        Meshes making up one URDF geometry element.
    output_file : str
        Path to write. Whether it may be written at all is
        :func:`_should_skip_mesh_export`'s decision, not this one's.
    """
    trimesh = _lazy_trimesh()
    if output_file.endswith('.dae'):
        _write_collada(meshes, output_file)
    elif output_file.endswith('.stl'):
        trimesh.exchange.export.export_mesh(
            trimesh.util.concatenate(meshes), output_file)
    elif output_file.endswith('.glb') or output_file.endswith('.gltf'):
        _write_gltf(meshes, output_file)
    else:
        trimesh.exchange.export.export_mesh(meshes, output_file)

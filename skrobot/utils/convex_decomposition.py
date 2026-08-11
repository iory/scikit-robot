"""Approximate convex decomposition of meshes using CoACD."""

import hashlib
import json
from logging import getLogger
import os
import shutil
import tempfile
import threading
import zipfile

import numpy as np

from skrobot._lazy_imports import _lazy_trimesh


logger = getLogger(__name__)


__all__ = [
    'COACD_PRESETS',
    'is_coacd_available',
    'convex_decomposition',
    'convex_decomposition_cached',
    'clear_convex_decomposition_cache',
]


# CoACD parameters per quality preset.  CoACD's cost is dominated by its MCTS
# search, which runs until ``max_convex_hull`` cuts whenever ``threshold`` is
# too low to stop earlier -- so a low threshold + high part cap + many MCTS
# iterations makes every mesh (even a tiny one) pay the full search.
# 'balanced' keeps the search shallow (seconds to tens of seconds per mesh);
# 'fine' trades runtime for a tighter fit.
COACD_PRESETS = {
    'balanced': {'threshold': 0.2, 'max_convex_hull': 6,
                 'preprocess_resolution': 30, 'mcts_iterations': 30},
    'fine': {'threshold': 0.1, 'max_convex_hull': 8,
             'preprocess_resolution': 40, 'mcts_iterations': 60},
}


def is_coacd_available():
    """Return True if the optional ``coacd`` package is importable.

    CoACD is an optional dependency (install with ``pip install coacd`` or
    ``pip install scikit-robot[coacd]``); use this to check for it without
    paying the import cost.

    Returns
    -------
    available : bool
        True if ``import coacd`` would succeed.
    """
    import importlib.util
    return importlib.util.find_spec('coacd') is not None


def convex_decomposition(mesh, quality='balanced', **params):
    """Decompose a mesh into approximate convex parts using CoACD.

    Convex parts are what physics engines (Gazebo / Bullet / MuJoCo) actually
    want as collision geometry; reusing a concave visual mesh works but is
    slow and can produce wrong contacts.  Each returned part is passed
    through its convex hull, so the results are clean watertight convex
    meshes.

    Note that CoACD is slow (seconds to tens of seconds per mesh depending on
    complexity and preset); callers that decompose repeatedly should cache the
    result keyed by mesh content and parameters.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to decompose.
    quality : str, optional
        Preset name, one of ``COACD_PRESETS`` keys ('balanced' or 'fine').
        'balanced' (default) is a few times faster; 'fine' fits tighter.
    **params
        Explicit CoACD parameters (e.g. ``threshold``, ``max_convex_hull``,
        ``preprocess_resolution``, ``mcts_iterations``).  These override the
        preset values and are forwarded to ``coacd.run_coacd``.

    Returns
    -------
    parts : list of trimesh.Trimesh
        The convex parts covering the input mesh.

    Raises
    ------
    RuntimeError
        If the optional ``coacd`` package is not installed.
    ValueError
        If ``quality`` is not a known preset name.

    Examples
    --------
    >>> import trimesh
    >>> from skrobot.utils.convex_decomposition import convex_decomposition
    >>> mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.3)
    >>> parts = convex_decomposition(mesh)
    >>> all(part.is_watertight for part in parts)
    True
    """
    trimesh = _lazy_trimesh()
    if quality not in COACD_PRESETS:
        raise ValueError(
            'unsupported quality: {!r} (expected one of {})'.format(
                quality, sorted(COACD_PRESETS)))
    if not is_coacd_available():
        raise RuntimeError(
            "CoACD convex decomposition needs the optional 'coacd' package"
            ' -- install it with: pip install coacd')
    import coacd

    run_params = dict(COACD_PRESETS[quality])
    run_params.update(params)
    coacd.set_log_level('error')
    coacd_mesh = coacd.Mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64))
    raw_parts = coacd.run_coacd(coacd_mesh, merge=True, **run_params)

    parts = []
    for vertices, faces in raw_parts:
        # CoACD parts are convex by construction; taking the convex hull
        # drops any sliver faces and guarantees a watertight mesh.
        part = trimesh.Trimesh(
            vertices=np.asarray(vertices),
            faces=np.asarray(faces),
            process=False).convex_hull
        parts.append(part)
    logger.debug('CoACD decomposed mesh into %d convex part(s)', len(parts))
    return parts


# ------------------------------------------------------------------ disk cache
# CoACD costs seconds to tens of seconds per mesh, and a robot description
# typically decomposes the same part many times (several links sharing one
# servo mesh, a re-export of an unchanged model, a test suite).  The results
# are a pure function of the mesh content and the CoACD parameters, so they can
# be cached on disk and reused indefinitely -- a cache hit costs a hash and a
# few file reads, and never touches CoACD.

# Bump when the on-disk layout or the meaning of a cache entry changes, so that
# entries written by an older skrobot are ignored instead of misread.
_CACHE_VERSION = 1

# formats whose geometry lives in files the named one only references, so
# hashing that file alone cannot notice the geometry changing
_EXTERNAL_GEOMETRY_FORMATS = frozenset(('.gltf', '.obj', '.dae', '.mtl'))

# Name of the manifest written *last* inside a cache entry: its presence is what
# makes an entry valid, so a build interrupted halfway is simply a miss.
_MANIFEST_NAME = 'parts.json'

# Errors that mean "this cache entry is unusable" (missing, truncated, written
# by a crashed process, hand-edited, ...) rather than a programming mistake.
# EOFError is in here for a specific reason: a rename that was not preceded by
# an fsync can leave a zero-length file behind after a crash, and numpy raises
# EOFError -- not OSError -- on one.  Reading a cache must never be able to
# fail a call that would have succeeded without it.
_CACHE_READ_ERRORS = (OSError, EOFError, ValueError, KeyError, TypeError,
                      IndexError, zipfile.BadZipFile)

# One lock per cache key.  Several links commonly share a single source mesh;
# without this they would all miss, all run CoACD, and all write the same files
# at once (on Windows, a file-in-use error).  Locking per key -- rather than
# globally -- keeps decompositions of *different* meshes parallel.
_cache_locks = {}
_cache_locks_guard = threading.Lock()


def _cache_key_lock(key):
    """Return the process-wide :class:`threading.Lock` for a cache key.

    Parameters
    ----------
    key : str
        Cache key, as returned by ``_cache_key``.

    Returns
    -------
    lock : threading.Lock
        The lock guarding builds of that key, created on first use.
    """
    with _cache_locks_guard:
        lock = _cache_locks.get(key)
        if lock is None:
            lock = _cache_locks[key] = threading.Lock()
        return lock


def _default_cache_dir():
    """Return the default directory holding cached decompositions.

    Returns
    -------
    cache_dir : str
        ``<skrobot cache dir>/convex_decomposition``.  The skrobot cache dir
        honours the ``SKROBOT_CACHE_DIR`` environment variable, so this is
        resolved on every call rather than at import time.
    """
    from skrobot.data import get_cache_dir
    return os.path.join(get_cache_dir(), 'convex_decomposition')


def _file_digest(path):
    """Return the SHA-1 hex digest of a file's bytes.

    Parameters
    ----------
    path : str
        Path of the file to hash.

    Returns
    -------
    digest : str
        Hex digest of the file content.
    """
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    if os.path.splitext(path)[1].lower() in _EXTERNAL_GEOMETRY_FORMATS:
        # the named file does not hold the geometry: a .gltf points at sibling
        # .bin buffers, an .obj may point at other files.  Editing only what it
        # references would leave this digest unchanged, so fold in the sibling
        # directory's shape as well -- coarse, but it moves when they do.
        directory = os.path.dirname(os.path.abspath(path))
        try:
            for name in sorted(os.listdir(directory)):
                sibling = os.path.join(directory, name)
                if sibling == os.path.abspath(path) or not os.path.isfile(
                        sibling):
                    continue
                stat = os.stat(sibling)
                h.update('{}:{}:{}'.format(
                    name, stat.st_size, stat.st_mtime_ns).encode())
        except OSError:
            pass
    return h.hexdigest()


def _mesh_digest(mesh):
    """Return the SHA-1 hex digest of a mesh's vertices and faces.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to hash.

    Returns
    -------
    digest : str
        Hex digest of the exact vertex and face arrays.  This is a content
        hash, not a geometric one: a translated or rotated copy hashes
        differently, which is what the cache needs since the decomposition is
        expressed in the mesh frame.
    """
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    h = hashlib.sha1()
    h.update(json.dumps([list(vertices.shape), list(faces.shape)]).encode())
    h.update(vertices.tobytes())
    h.update(faces.tobytes())
    return h.hexdigest()


def _cache_key(source_kind, digest, run_params):
    """Return the cache key for one decomposition.

    Parameters
    ----------
    source_kind : str
        ``'path'`` or ``'mesh'``; keeps the two hashing domains separate.
    digest : str
        Content digest of the source mesh.
    run_params : dict
        The effective CoACD parameters (preset merged with overrides).

    Returns
    -------
    key : str
        16 hex characters -- short enough to keep the cache paths well clear
        of the Windows path length limit, wide enough (64 bits) that an
        accidental collision is not a practical concern.
    """
    h = hashlib.sha1(digest.encode())
    h.update(json.dumps([source_kind, run_params, _CACHE_VERSION],
                        sort_keys=True, default=repr).encode())
    return h.hexdigest()[:16]


def _atomic_write(path, write):
    """Write a file atomically by writing a temporary file and renaming it.

    Parameters
    ----------
    path : str
        Final path of the file.
    write : callable
        Called with the open binary file object of the temporary file.
    """
    tmp = '{}.{}.{}.tmp'.format(path, os.getpid(), threading.get_ident())
    try:
        with open(tmp, 'wb') as f:
            write(f)
            # flush to disk before the rename: a rename that overtakes its own
            # data leaves a zero-length file behind after a crash
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _read_cache_entry(part_dir):
    """Read the convex parts stored in a cache entry.

    Parameters
    ----------
    part_dir : str
        Directory of the cache entry.

    Returns
    -------
    parts : list of trimesh.Trimesh or None
        The cached parts, or None if the entry is absent, incomplete or
        unreadable -- in which case the caller rebuilds it.
    """
    trimesh = _lazy_trimesh()
    try:
        with open(os.path.join(part_dir, _MANIFEST_NAME)) as f:
            manifest = json.load(f)
        if manifest['version'] != _CACHE_VERSION:
            return None
        names = manifest['parts']
        parts = []
        for name in names:
            # a manifest is data, not a path list to trust: a name carrying a
            # separator would read a file outside the entry
            if os.path.basename(name) != name:
                return None
            with np.load(os.path.join(part_dir, name)) as data:
                vertices = data['vertices']
                faces = data['faces']
            parts.append(trimesh.Trimesh(vertices=vertices, faces=faces,
                                         process=False))
        return parts
    except _CACHE_READ_ERRORS:
        return None


def _write_cache_entry(part_dir, quality, run_params, parts):
    """Store convex parts in a cache entry.

    The manifest is written last, so an interrupted or partially written entry
    reads back as a miss rather than as a truncated result.

    Parameters
    ----------
    part_dir : str
        Directory of the cache entry; created if needed.
    quality : str
        Preset name the parts were computed with (recorded for inspection).
    run_params : dict
        Effective CoACD parameters (recorded for inspection).
    parts : list of trimesh.Trimesh
        The parts to store.
    """
    parent = os.path.dirname(part_dir)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=os.path.basename(part_dir) + '.',
                               suffix='.tmp', dir=parent)
    try:
        names = []
        for i, part in enumerate(parts):
            name = 'part_{}.npz'.format(i)
            vertices = np.ascontiguousarray(part.vertices, dtype=np.float64)
            faces = np.ascontiguousarray(part.faces, dtype=np.int64)
            _atomic_write(
                os.path.join(staging, name),
                lambda f, v=vertices, t=faces: np.savez(f, vertices=v,
                                                        faces=t))
            names.append(name)
        manifest = {'version': _CACHE_VERSION, 'quality': quality,
                    'params': run_params, 'parts': names}
        _atomic_write(
            os.path.join(staging, _MANIFEST_NAME),
            lambda f: f.write(json.dumps(manifest, sort_keys=True,
                                         default=repr).encode()))
        # An entry is published as one directory, never file by file: CoACD is
        # not deterministic -- two runs on the same mesh return the same parts
        # in a different order -- so interleaving two builds in a shared
        # directory yields an entry that reads back as valid while covering
        # less of the mesh than either run did.
        try:
            os.rename(staging, part_dir)
            return
        except OSError:
            pass
        if _read_cache_entry(part_dir) is not None:
            # somebody published a whole entry first; theirs is as good as ours
            shutil.rmtree(staging, ignore_errors=True)
            return
        # what is there is unreadable (corrupt, stale version, half-deleted):
        # move it aside and take its place
        discard = staging + '.discard'
        try:
            os.rename(part_dir, discard)
            os.rename(staging, part_dir)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
        finally:
            shutil.rmtree(discard, ignore_errors=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def convex_decomposition_cached(mesh, quality='balanced', cache_dir=None,
                                **params):
    """Decompose a mesh into convex parts, reusing a decomposition on disk.

    Same result as :func:`convex_decomposition`, but the parts are persisted
    under ``cache_dir`` and keyed by the mesh content and the effective CoACD
    parameters.  CoACD costs seconds to tens of seconds per mesh, so repeated
    decompositions -- several links sharing one part mesh, a re-export of an
    unchanged model, a test run -- become instant.  A cache hit only reads the
    stored parts back: the ``coacd`` package need not even be installed.

    Concurrent calls for the same key are serialised with a per-key lock, so a
    mesh shared by many links is decomposed once even from several threads.
    Distinct meshes still decompose in parallel.

    Parameters
    ----------
    mesh : trimesh.Trimesh or str or os.PathLike
        Input mesh, or the path of a mesh file.  A path is hashed by its bytes
        and only loaded on a cache miss, so a hit costs no mesh parsing at all.
    quality : str, optional
        Preset name, one of ``COACD_PRESETS`` keys ('balanced' or 'fine').
    cache_dir : str or os.PathLike or None, optional
        Directory holding the cache.  Defaults to ``convex_decomposition``
        inside the skrobot cache directory (``~/.skrobot``, or
        ``SKROBOT_CACHE_DIR`` if set).
    **params
        Explicit CoACD parameters overriding the preset, forwarded to
        :func:`convex_decomposition`.  The key is built from the *effective*
        parameters, so two calls that would run CoACD identically share one
        entry whatever preset name they name it by.

    Returns
    -------
    parts : list of trimesh.Trimesh
        The convex parts covering the input mesh, exactly as
        :func:`convex_decomposition` returns them.

    Raises
    ------
    RuntimeError
        If the result is not cached and the optional ``coacd`` package is not
        installed.
    ValueError
        If ``quality`` is not a known preset name.

    Notes
    -----
    A cache entry that is missing, incomplete or unreadable (an interrupted
    build, a hand-edited manifest, a stale format) is treated as a miss and
    rebuilt.  If the cache cannot be written -- a read-only directory, say --
    the freshly computed parts are still returned and a warning is logged.

    Examples
    --------
    >>> import trimesh
    >>> from skrobot.utils.convex_decomposition import (
    ...     convex_decomposition_cached)
    >>> mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.3)
    >>> parts = convex_decomposition_cached(mesh)   # runs CoACD
    >>> parts = convex_decomposition_cached(mesh)   # read back from disk
    """
    if quality not in COACD_PRESETS:
        raise ValueError(
            'unsupported quality: {!r} (expected one of {})'.format(
                quality, sorted(COACD_PRESETS)))
    run_params = dict(COACD_PRESETS[quality])
    run_params.update(params)

    if isinstance(mesh, (str, os.PathLike)):
        source_path = os.fspath(mesh)
        key = _cache_key('path', _file_digest(source_path), run_params)
    else:
        source_path = None
        key = _cache_key('mesh', _mesh_digest(mesh), run_params)

    if cache_dir is None:
        cache_dir = _default_cache_dir()
    part_dir = os.path.join(os.fspath(cache_dir), key)

    parts = _read_cache_entry(part_dir)
    if parts is not None:
        logger.debug('convex decomposition cache hit (%s)', key)
        return parts

    # Hold the per-key lock across the re-check and the build: another thread
    # may have finished this very key while we were reading the miss.
    with _cache_key_lock(key):
        parts = _read_cache_entry(part_dir)
        if parts is not None:
            logger.debug('convex decomposition cache hit (%s)', key)
            return parts
        if source_path is not None:
            mesh = _lazy_trimesh().load(source_path, force='mesh')
        parts = convex_decomposition(mesh, quality=quality, **params)
        try:
            _write_cache_entry(part_dir, quality, run_params, parts)
        except OSError as e:
            logger.warning('could not cache convex decomposition in %s: %s',
                           part_dir, e)
        return parts


def clear_convex_decomposition_cache(cache_dir=None):
    """Delete cached convex decompositions.

    Nothing evicts entries on its own: every re-export of a mesh and every
    change of parameters mints a fresh key, and the superseded entries stay on
    disk.  Call this to reclaim the space.

    Parameters
    ----------
    cache_dir : str or None, optional
        Directory to clear.  Defaults to the same location
        :func:`convex_decomposition_cached` writes to.

    Returns
    -------
    removed : int
        How many cache entries were deleted.
    """
    cache_dir = cache_dir or _default_cache_dir()
    removed = 0
    try:
        names = sorted(os.listdir(cache_dir))
    except OSError:
        return 0
    for name in names:
        entry = os.path.join(cache_dir, name)
        if not os.path.isdir(entry):
            continue
        shutil.rmtree(entry, ignore_errors=True)
        if not os.path.exists(entry):
            removed += 1
    return removed

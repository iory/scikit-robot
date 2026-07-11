"""Approximate convex decomposition of meshes using CoACD."""

from logging import getLogger

import numpy as np

from skrobot._lazy_imports import _lazy_trimesh


logger = getLogger(__name__)


__all__ = [
    'COACD_PRESETS',
    'is_coacd_available',
    'convex_decomposition',
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

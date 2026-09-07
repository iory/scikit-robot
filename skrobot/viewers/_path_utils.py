import warnings

import numpy as np


def polyline_segments(mesh):
    """Extract line segments from a trimesh ``Path3D``.

    ``LineString`` stores its geometry as a :class:`trimesh.path.Path3D`,
    which keeps the vertices in a flat array and the connectivity in its
    entities. This flattens that into explicit segment endpoints.

    Parameters
    ----------
    mesh : trimesh.path.Path3D
        Path whose entities are walked to build the segments.

    Returns
    -------
    segments : numpy.ndarray
        ``(n_segments, 2, 3)`` float array of segment endpoints. Empty
        with shape ``(0, 2, 3)`` when the path holds no usable segment.
    entity_indices : numpy.ndarray
        ``(n_segments,)`` int array giving, for each segment, the index
        of the entity it came from. Used to expand per-entity colors to
        per-segment colors. Segments recovered from ``vertex_nodes``
        (i.e. when the path exposes no entities) are all attributed to
        entity ``0``.
    """
    empty = (np.zeros((0, 2, 3), dtype=np.float64),
             np.zeros(0, dtype=np.int64))

    vertices = getattr(mesh, 'vertices', None)
    if vertices is None:
        return empty
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return empty
    if len(vertices) < 2:
        return empty

    segments = []
    entity_indices = []

    def append_segment(i, j, entity_index):
        if i < 0 or j < 0 or i >= len(vertices) or j >= len(vertices):
            return
        segments.append((vertices[i], vertices[j]))
        entity_indices.append(entity_index)

    entities = getattr(mesh, 'entities', None)
    if entities is not None:
        for entity_index, entity in enumerate(entities):
            points = np.asarray(
                getattr(entity, 'points', []), dtype=np.int64).reshape(-1)
            if len(points) < 2:
                continue
            for i in range(len(points) - 1):
                append_segment(int(points[i]), int(points[i + 1]),
                               entity_index)
            # ``closed`` entities usually already repeat the first index
            # as their last point, in which case the loop above has
            # drawn the closing segment.
            if (getattr(entity, 'closed', False) and len(points) > 2
                    and points[-1] != points[0]):
                append_segment(int(points[-1]), int(points[0]), entity_index)

    if not segments:
        vertex_nodes = getattr(mesh, 'vertex_nodes', None)
        if vertex_nodes is None:
            return empty
        vertex_nodes = np.asarray(vertex_nodes, dtype=np.int64)
        if vertex_nodes.ndim == 1:
            if len(vertex_nodes) % 2 != 0:
                return empty
            vertex_nodes = vertex_nodes.reshape(-1, 2)
        if vertex_nodes.ndim != 2 or vertex_nodes.shape[1] < 2:
            return empty
        for node in vertex_nodes:
            append_segment(int(node[0]), int(node[1]), 0)

    if not segments:
        return empty
    return (np.asarray(segments, dtype=np.float64),
            np.asarray(entity_indices, dtype=np.int64))


def polyline_segment_colors(mesh, entity_indices):
    """Resolve per-segment RGB colors for a trimesh ``Path3D``.

    trimesh stores path colors per entity, while renderers usually want
    one color per segment.

    Parameters
    ----------
    mesh : trimesh.path.Path3D
        Path holding the colors.
    entity_indices : numpy.ndarray
        ``(n_segments,)`` entity index of each segment, as returned by
        :func:`polyline_segments`.

    Returns
    -------
    numpy.ndarray or None
        ``(n_segments, 3)`` uint8 RGB array, or ``None`` when the path
        carries no color or the colors cannot be matched to the
        segments.
    """
    colors = getattr(mesh, 'colors', None)
    if colors is None:
        return None
    colors = np.asarray(colors)
    if colors.ndim == 1:
        colors = colors.reshape(1, -1)
    if colors.ndim != 2 or colors.shape[0] == 0 or colors.shape[1] < 3:
        return None

    entities = getattr(mesh, 'entities', None)
    n_entities = 0 if entities is None else len(entities)
    if colors.shape[0] == 1:
        per_entity = np.repeat(colors, max(n_entities, 1), axis=0)
    elif colors.shape[0] == n_entities:
        per_entity = colors
    else:
        warnings.warn(
            ('Dropping Path3D colors: expected 1 or {} rows (per-entity), '
             'got {}.').format(n_entities, colors.shape[0]),
            UserWarning)
        return None

    if entity_indices.size and entity_indices.max() >= per_entity.shape[0]:
        warnings.warn(
            'Dropping Path3D colors: segment entity index out of range.',
            UserWarning)
        return None

    rgb = per_entity[entity_indices][:, :3]
    if np.issubdtype(rgb.dtype, np.floating) and rgb.size and rgb.max() <= 1.0:
        rgb = rgb * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)

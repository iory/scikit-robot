import math
import os
import tempfile
import time
import warnings

import numpy as np

from skrobot.coordinates import Coordinates
import skrobot.model as model_module


def _load_mitsuba(variant):
    """Import mitsuba lazily and select a headless-capable variant.

    The variant may be given explicitly, via the ``SKROBOT_MITSUBA_VARIANT``
    environment variable (e.g. ``SKROBOT_MITSUBA_VARIANT=llvm_ad_rgb`` to force
    the CPU), or left to auto-select. Auto-select prefers the Apple-GPU (Metal)
    variant when it is compiled in and otherwise uses the CPU (llvm) variant.
    """
    import mitsuba as mi
    available = mi.variants()
    if variant is None:
        variant = os.environ.get('SKROBOT_MITSUBA_VARIANT') or None
    if variant is not None and variant not in available:
        raise ValueError(
            "mitsuba variant '{}' is not available (compiled variants: {}). "
            'On Apple Silicon use metal_ad_rgb, on NVIDIA cuda_ad_rgb, '
            'otherwise llvm_ad_rgb.'.format(variant, list(available)))
    if variant is None:
        # Prefer the Apple-GPU (Metal) variant when present: it only ships in
        # the macOS wheels, so this transparently uses the GPU on a Mac while
        # falling back to the CPU (llvm) variant everywhere else. CUDA is not
        # auto-selected -- a compiled-in cuda variant does not guarantee a
        # usable GPU/driver at runtime.
        for cand in ('metal_ad_rgb', 'llvm_ad_rgb', 'scalar_rgb', 'cuda_ad_rgb'):
            if cand in available:
                variant = cand
                break
        else:
            variant = available[0]
    if mi.variant() != variant:
        mi.set_variant(variant)
    return mi


def _rgba01(color, default_alpha=1.0):
    """Return ``color`` as clipped RGBA floats in [0, 1]."""
    rgba = np.asarray(color, dtype=np.float64).reshape(-1)
    if rgba.size < 3:
        raise ValueError('color must have at least 3 channels')
    if rgba.size == 3:
        rgba = np.hstack([rgba[:3], [float(default_alpha)]])
    else:
        rgba = rgba[:4]
    if rgba.max() > 1.0:
        rgba = rgba / 255.0
    return np.clip(rgba, 0.0, 1.0)


def _rgba_rows01(colors, default_alpha=1.0):
    """Return an ``(N, 4)`` RGBA array in [0, 1] from N color rows."""
    arr = np.asarray(colors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError('colors must have at least 3 channels')
    if arr.shape[1] == 3:
        alpha = np.full((len(arr), 1), float(default_alpha), dtype=np.float64)
        arr = np.hstack([arr[:, :3], alpha])
    else:
        arr = arr[:, :4]
    if arr.size and arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _face_colors_with_alpha(face_colors):
    """Return face colors with an explicit alpha channel, preserving scale."""
    arr = np.asarray(face_colors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError('face colors must have at least 3 channels')
    if arr.shape[1] >= 4:
        return arr[:, :4]
    default_alpha = 255.0 if arr.max() > 1.0 else 1.0
    alpha = np.full((len(arr), 1), default_alpha, dtype=np.float64)
    return np.hstack([arr[:, :3], alpha])


def _opacity_to_alpha(opacity):
    """Best-effort scalar alpha from Mitsuba mask opacity entries."""
    if isinstance(opacity, dict):
        value = opacity.get('value', 1.0)
    else:
        value = opacity
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 1.0
    if arr.max() > 1.0:
        arr = arr / 255.0
    return float(np.clip(arr[0], 0.0, 1.0))


def _mesh_color(mesh):
    """Best-effort average base color of a trimesh mesh, in [0, 1]."""
    default = np.array([0.75, 0.76, 0.78, 1.0])
    try:
        colors = getattr(mesh, 'colors', None)
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float64)
            if colors.size:
                return _rgba_rows01(colors).mean(axis=0)
    except Exception:
        pass
    visual = getattr(mesh, 'visual', None)
    if visual is None:
        return default
    try:
        kind = getattr(visual, 'kind', None)
        if kind in ('vertex', 'face') or hasattr(visual, 'vertex_colors'):
            colors = np.asarray(visual.vertex_colors, dtype=np.float64)
            if colors.size:
                return _rgba_rows01(colors).mean(axis=0)
        material = getattr(visual, 'material', None)
        if material is not None:
            col = getattr(material, 'main_color', None)
            if col is None:
                col = getattr(material, 'baseColorFactor', None)
            if col is not None:
                return _rgba01(col)
    except Exception:
        pass
    return default


def _color_submeshes(mesh):
    """Split a mesh into ``(submesh, rgba01)`` pairs, one per face color.

    A mesh that mixes several colors in one object -- e.g. an :class:`Axis`
    whose X/Y/Z arms are red/green/blue -- would otherwise be flattened to a
    single averaged color by :func:`_mesh_color` (the path tracer shades each
    shape with one diffuse color). Splitting it into one submesh per color keeps
    every part its own color. Single-color meshes (most robot links) are
    returned unchanged, so this only adds shapes where it is needed.
    """
    visual = getattr(mesh, 'visual', None)
    face_colors = None
    if visual is not None and getattr(visual, 'kind', None) in ('vertex', 'face'):
        try:
            face_colors = _face_colors_with_alpha(mesh.visual.face_colors)
        except Exception:
            face_colors = None
    if face_colors is None or len(face_colors) != len(mesh.faces):
        return [(mesh, _mesh_color(mesh))]
    uniq = np.unique(face_colors, axis=0)
    if len(uniq) <= 1:
        # Keep historical RGB from _mesh_color() for single-colour meshes,
        # but preserve face-alpha updates from Link.set_alpha().
        rgba = np.asarray(_mesh_color(mesh), dtype=np.float64).copy()
        rgba[3] = _rgba01(uniq[0])[3]
        return [(mesh, rgba)]
    result = []
    for c in uniq:
        idx = np.where(np.all(face_colors == c, axis=1))[0]
        sub = mesh.submesh([idx], append=True, repair=False)
        result.append((sub, _rgba01(c)))
    return result


def _diffuse_bsdf(color):
    """A diffuse BSDF dict for ``color`` (rgba in [0, 1]).

    Mitsuba's ``diffuse`` BSDF is one-sided: a triangle whose normal points away
    from the viewer reflects nothing and renders pure black. Robot/collision
    meshes and primitives (boxes, etc.) frequently have some inward-facing
    normals, which is why their shadowed faces came out black. Wrapping the
    diffuse in ``twosided`` makes both faces reflect, removing the black facets.
    """
    rgba = _rgba01(color)
    opaque = {'type': 'twosided',
              'bsdf': {'type': 'diffuse',
                       'reflectance': {'type': 'rgb',
                                       'value': [float(v)
                                                 for v in rgba[:3]]}}}
    if rgba[3] >= 1.0:
        return opaque
    # Mitsuba's mask BSDF applies stochastic opacity while keeping the nested
    # material unchanged, so alpha behaves as expected for Link.set_alpha().
    return {'type': 'mask',
            'opacity': {'type': 'rgb', 'value': float(rgba[3])},
            'nested': opaque}


class MitsubaViewer(object):
    """Headless offscreen renderer backed by Mitsuba 3.

    Unlike :class:`~skrobot.viewers.PyrenderViewer`, this viewer opens no window
    and needs no display server / OpenGL context, so it works over SSH, in CI,
    and on macOS where offscreen OpenGL is unavailable. It renders the same
    ``Link`` / ``CascadedLink`` geometry as the other viewers (including
    meshes, line/path geometry and point clouds), except backend-specific
    runtime objects such as pyrender ``.scene`` nodes.
    It uses Mitsuba's path tracer and writes the result with
    :meth:`save_image` (or returns it from :meth:`render`).

    Parameters
    ----------
    resolution : tuple(int, int), optional
        Output image size ``(width, height)``. Default ``(640, 480)``.
    update_interval : float, optional
        Accepted for constructor compatibility with
        :class:`~skrobot.viewers.PyrenderViewer` and
        :class:`~skrobot.viewers.TrimeshSceneViewer`. Ignored: this backend has
        no background window-refresh timer because rendering happens on demand.
    title : str, optional
        Accepted for constructor compatibility with windowed viewers. Ignored:
        the matplotlib window title used by :meth:`show` is not part of this
        viewer's stable API contract.
    spp : int, optional
        Samples per pixel for the path tracer. Higher is less noisy but
        slower. When ``None`` (default) it is chosen from the variant: ``512``
        on a GPU (CUDA / Metal), where samples are cheap, and ``64`` on the CPU
        (llvm / scalar), where they are not. Pass an int to force a value.
    variant : str, optional
        Mitsuba variant to use. When ``None`` (default) it auto-selects the
        Apple-GPU ``'metal_ad_rgb'`` variant on macOS and the CPU
        ``'llvm_ad_rgb'`` variant elsewhere. Pass ``'metal_ad_rgb'`` (Apple
        GPU), ``'cuda_ad_rgb'`` (NVIDIA) or ``'llvm_ad_rgb'`` (CPU) to force one
        (or set the ``SKROBOT_MITSUBA_VARIANT`` environment variable).
    ground : bool, optional
        If ``True`` (default) add a neutral ground plane and a key light.
    fov : float or sequence(float, float), optional
        Camera field of view in degrees. Default ``(60.0, 45.0)`` so the same
        tuple convention as :class:`~skrobot.viewers.PyrenderViewer` is used by
        default. A scalar sets ``fov`` with ``fov_axis`` from ``fov_axis``.
        A 2-element ``(xfov, yfov)`` sequence uses ``yfov`` with
        ``fov_axis='y'``. Passing ``None`` preserves Mitsuba's
        ``fov=45``/implicit-x-axis default.
    fov_axis : str, optional
        Mitsuba perspective sensor FOV axis used when ``fov`` is a scalar.
        Default is ``'x'``.
    ground_height : float, optional
        Ground plane world ``z``. When ``None`` (default), it is auto-detected
        from the lowest point of added geometry (robot links and markers), with
        a ``0.0`` fallback when empty. This avoids placing the default ``z=0``
        plane in front of robots that do not stand on the origin plane (e.g. a
        gantry hanging below the origin).
    ground_size : float, optional
        Half-extent of the square ground plane in metres. Mitsuba's rectangle
        spans ``[-1, 1]``, so ``ground_size=6.0`` (default) yields a
        ``12m x 12m`` ground.
    light_intensity : float, optional
        Radiance scale of the key area light. Default ``5.0``.
    light_size : float, optional
        Key-light rectangle half-extent in metres. Mitsuba's rectangle spans
        ``[-1, 1]``, so ``light_size=1.5`` yields a ``3m x 3m`` area light.
        When ``None`` (default), it is derived from scene extent so very large
        scenes do not get under-lit, and clamped to never shrink below the
        historical single-robot size.
    line_radius : float, optional
        Radius in metres used when converting line/path geometry (for example a
        ``LineString`` ``Path3D``) into renderable tube meshes. Default
        ``0.002``.
    point_radius : float, optional
        Radius in metres used when converting ``PointCloud`` geometry into
        renderable sphere meshes. Default ``0.004``.
    ambient_light : float, optional
        Radiance scale of the constant ambient emitter. Default ``0.12``.
    """
    def __init__(self, resolution=(640, 480), update_interval=None,
                 title=None, spp=None, variant=None,
                 ground=True, fov=(60.0, 45.0), fov_axis='x',
                 ground_height=None,
                 ground_size=6.0, light_intensity=5.0, ambient_light=0.12,
                 light_size=None, line_radius=0.002, point_radius=0.004):
        # Keep constructor parity with interactive viewers so scripts can swap
        # class names directly. These options are window-only there.
        del update_interval
        del title
        self.mi = _load_mitsuba(variant)
        self.resolution = tuple(resolution)
        self.spp = int(spp) if spp is not None else self._default_spp()
        self.ground = ground
        self.fov_axis = fov_axis
        self.fov = None
        if fov is not None:
            self._set_fov(fov)
        self.ground_height = ground_height
        self.ground_size = float(ground_size)
        self.light_intensity = float(light_intensity)
        self.light_size = None if light_size is None else float(light_size)
        self.line_radius = float(line_radius)
        self.point_radius = float(point_radius)
        self.ambient_light = float(ambient_light)
        # Rendering each point as a mesh sphere is exact but expensive; warn
        # when the point count is large while still rendering every point.
        self._point_cloud_warn_budget = 20000
        self._warned_point_clouds = set()
        self._point_sphere_template = None
        self._links = {}                 # mesh_id -> (link, ply_path, rgba)
        self._extra = {}                 # name -> scene-dict entry
        self._link_local_corners = {}    # key -> local AABB corners (8, 3)
        self._extra_local_corners = {}   # name -> local AABB corners (8, 3)
        self._auto_ground_z = None
        self._auto_ground_sig = None
        self._auto_light_r = None
        self._auto_light_sig = None
        self._tmpdir = tempfile.mkdtemp(prefix='skrobot_mitsuba_')
        self._camera = None              # (eye, target, up) or None -> auto
        self._last_image = None
        # interactive display state (populated by show())
        self._fig = None
        self._ax = None
        self._im = None
        self._nb_handle = None
        self._drag_xy = None
        # cached compiled scene for fast incremental (transform-only) redraws
        self._scene = None
        self._params = None
        self._mesh_local = {}            # key -> local verts/normals + param ids
        self._untracked = {}             # key -> merged link/marker (see below)
        self._untracked_T = {}           # key -> world transform at build time
        self._link_sig = None
        self._joint_axis_map = {}        # joint_id -> marker names
        self._geom_version = 0           # bumped when geometry/sensor changes
        self._built_version = -1

    def _default_spp(self):
        """Pick a samples-per-pixel default from the active variant.

        Path tracing is Monte-Carlo, so too few samples per pixel leave visible
        grain. On the GPU (CUDA / Metal) a high sample count is nearly free -- an
        RTX 4090 renders 512 spp of these scenes in ~50 ms -- so default to a
        clean value there. On the CPU (llvm / scalar) high spp would be slow, so
        keep it modest. Pass ``spp=`` explicitly to override either way.
        """
        variant = self.mi.variant()
        if 'cuda' in variant or 'metal' in variant:
            return 512
        return 64

    @staticmethod
    def _aabb_corners(bounds):
        bounds = np.asarray(bounds, float)
        bmin = bounds[0]
        bmax = bounds[1]
        return np.array([
            [bmin[0], bmin[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmax[1], bmax[2]],
            [bmax[0], bmin[1], bmin[2]],
            [bmax[0], bmin[1], bmax[2]],
            [bmax[0], bmax[1], bmin[2]],
            [bmax[0], bmax[1], bmax[2]],
        ], dtype=float)

    @staticmethod
    def _transform_points(points, T):
        points = np.asarray(points, float)
        T = np.asarray(T, float)
        return np.dot(points, T[:3, :3].T) + T[:3, 3]

    @staticmethod
    def _coerce_fov(fov):
        if np.isscalar(fov):
            return float(fov)
        arr = np.asarray(fov, dtype=float).reshape(-1)
        if arr.size == 2:
            return (float(arr[0]), float(arr[1]))
        raise ValueError('fov must be a scalar or a 2-element sequence')

    def _effective_sensor_fov(self):
        if self.fov is None:
            return 45, None
        if np.isscalar(self.fov):
            return float(self.fov), self.fov_axis
        return float(self.fov[1]), 'y'

    @staticmethod
    def _sensor_fov_xy_radians(sensor_fov, sensor_axis, resolution):
        """Return effective horizontal/vertical sensor FOV in radians."""
        width, height = resolution
        aspect = float(width) / max(float(height), 1e-12)
        half_fov = np.radians(float(sensor_fov)) / 2.0
        axis = 'x' if sensor_axis is None else str(sensor_axis).lower()
        if axis == 'y':
            half_y = half_fov
            half_x = np.arctan(np.tan(half_y) * aspect)
        else:
            half_x = half_fov
            half_y = np.arctan(np.tan(half_x) / aspect)
        return 2.0 * half_x, 2.0 * half_y

    def _effective_sensor_fov_xy_radians(self):
        """Return effective horizontal/vertical FOV from current sensor config."""
        sensor_fov, sensor_axis = self._effective_sensor_fov()
        return self._sensor_fov_xy_radians(
            sensor_fov, sensor_axis, self.resolution)

    def _set_fov(self, fov):
        prev = self._effective_sensor_fov()
        self.fov = self._coerce_fov(fov)
        if not np.isscalar(self.fov):
            self.fov_axis = 'y'
        return self._effective_sensor_fov() != prev

    def _auto_ground_height(self):
        zmins = []
        for key, (link, _ply, _color) in self._links.items():
            corners = self._link_local_corners.get(key)
            if corners is None:
                continue
            coords = link.worldcoords()
            zmins.append(coords.transform_vector(corners)[:, 2].min())
        for name, entry in self._extra.items():
            corners = self._extra_local_corners.get(name)
            if corners is None or 'to_world' not in entry:
                continue
            T = np.asarray(entry['to_world'].matrix, float)
            zmins.append(
                self._transform_points(corners, T)[:, 2].min())
        if zmins:
            return float(np.min(zmins))
        return 0.0

    def _auto_ground_signature(self):
        return (frozenset(self._links.keys()),
                frozenset(self._extra.keys()))

    def _auto_light_radius(self):
        points = self._collect_world_points()
        if points is None:
            return 1.0
        # Keep the same radius floor used by _auto_camera so tiny scenes do not
        # collapse to an unreasonably small key light.
        return max(0.3, 0.5 * float(np.linalg.norm(np.ptp(points, axis=0))))

    def _collect_world_points(self):
        """Return world-frame points spanning current scene geometry.

        Each tracked link contributes transformed AABB corners when available;
        otherwise its origin. Marker geometry contributes its transformed local
        corners. Returns ``None`` when there is no geometry.
        """
        points = []
        for key, (link, _ply, _color) in self._links.items():
            corners = self._link_local_corners.get(key)
            try:
                coords = link.worldcoords()
            except Exception:
                continue
            if corners is not None:
                points.append(coords.transform_vector(corners))
            else:
                points.append(np.asarray(coords.worldpos(), float).reshape(1, 3))
        for name, entry in self._extra.items():
            if 'to_world' not in entry:
                continue
            T = np.asarray(entry['to_world'].matrix, float)
            corners = self._extra_local_corners.get(name)
            if corners is not None:
                points.append(self._transform_points(corners, T))
            else:
                points.append(T[:3, 3].reshape(1, 3))
        if not points:
            return None
        return np.vstack(points)

    @staticmethod
    def _polyline_segments(mesh):
        vertices = getattr(mesh, 'vertices', None)
        if vertices is None:
            return []
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            return []
        if len(vertices) < 2:
            return []

        segments = []

        def append_segment(i, j):
            if i < 0 or j < 0 or i >= len(vertices) or j >= len(vertices):
                return
            segments.append((vertices[i], vertices[j]))

        entities = getattr(mesh, 'entities', None)
        if entities is not None:
            for entity in entities:
                points = np.asarray(
                    getattr(entity, 'points', []), dtype=np.int64).reshape(-1)
                if len(points) < 2:
                    continue
                for i in range(len(points) - 1):
                    append_segment(int(points[i]), int(points[i + 1]))
                if getattr(entity, 'closed', False) and len(points) > 2:
                    append_segment(int(points[-1]), int(points[0]))

        if not segments:
            vertex_nodes = getattr(mesh, 'vertex_nodes', None)
            if vertex_nodes is None:
                return segments
            vertex_nodes = np.asarray(vertex_nodes, dtype=np.int64)
            if vertex_nodes.ndim == 1:
                if len(vertex_nodes) % 2 != 0:
                    return segments
                vertex_nodes = vertex_nodes.reshape(-1, 2)
            if vertex_nodes.ndim != 2 or vertex_nodes.shape[1] < 2:
                return segments
            for node in vertex_nodes:
                append_segment(int(node[0]), int(node[1]))

        return segments

    def _path_to_tube_mesh(self, mesh):
        segments = self._polyline_segments(mesh)
        if not segments:
            return None

        import trimesh

        cylinders = []
        for p0, p1 in segments:
            if np.linalg.norm(p1 - p0) <= 1e-12:
                continue
            try:
                cylinders.append(
                    trimesh.creation.cylinder(
                        radius=self.line_radius, segment=[p0, p1]))
            except Exception:
                continue

        if not cylinders:
            return None
        if len(cylinders) == 1:
            return cylinders[0]
        return trimesh.util.concatenate(cylinders)

    def _point_cloud_to_sphere_mesh(self, point_cloud, link):
        points = np.asarray(getattr(point_cloud, 'vertices', None), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            return None

        n_points = len(points)
        if n_points > self._point_cloud_warn_budget:
            warn_key = (id(link), n_points)
            if warn_key not in self._warned_point_clouds:
                warnings.warn(
                    'PointCloud link "{}" has {} points (> {}); rendering all '
                    'points as spheres in MitsubaViewer may be slow.'.format(
                        getattr(link, 'name', '<unnamed>'),
                        n_points,
                        self._point_cloud_warn_budget))
                self._warned_point_clouds.add(warn_key)

        import trimesh

        if self._point_sphere_template is None:
            # Subdivisions=1 keeps each point light enough for practical clouds.
            self._point_sphere_template = trimesh.creation.icosphere(
                subdivisions=1, radius=1.0)

        unit_vertices = np.asarray(
            self._point_sphere_template.vertices, dtype=np.float64)
        unit_faces = np.asarray(self._point_sphere_template.faces, dtype=np.int64)
        n_unit_vertices = len(unit_vertices)

        vertices = (
            points[:, None, :]
            + self.point_radius * unit_vertices[None, :, :]).reshape(-1, 3)
        offsets = np.arange(n_points, dtype=np.int64)[:, None, None]
        faces = (unit_faces[None, :, :] + offsets * n_unit_vertices).reshape(-1, 3)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    @staticmethod
    def _mesh_sources(link):
        mesh = getattr(link, 'concatenated_visual_mesh', None)
        if mesh is not None:
            return [('concat', mesh)]
        # Prefer the concatenated mesh to match other backends and preserve
        # Link.set_color/reset_color/set_alpha updates. Fall back to the
        # original per-sub-mesh list only when concatenation is unavailable.
        mesh = getattr(link, 'visual_mesh', None)
        if isinstance(mesh, (list, tuple)):
            return [('visual_{}'.format(i), m) for i, m in enumerate(mesh)]
        return [('visual', mesh)]

    def _mesh_entries_for_rendering(self, link, mesh):
        if mesh is None:
            return []

        faces = getattr(mesh, 'faces', None)
        if faces is not None and len(faces) > 0:
            entries = []
            for sub, color in _color_submeshes(mesh):
                sub_faces = getattr(sub, 'faces', None)
                if sub is None or sub_faces is None or len(sub_faces) == 0:
                    continue
                entries.append((sub, color))
            return entries

        try:
            import trimesh
            if isinstance(mesh, trimesh.points.PointCloud):
                sphere_mesh = self._point_cloud_to_sphere_mesh(mesh, link)
                if sphere_mesh is not None:
                    return [(sphere_mesh, _mesh_color(mesh))]
        except Exception:
            pass

        tube_mesh = self._path_to_tube_mesh(mesh)
        if tube_mesh is not None:
            return [(tube_mesh, _mesh_color(mesh))]
        return []

    def _clear_link_entries(self, link):
        old_keys = [k for k, v in self._links.items() if v[0] is link]
        for key in old_keys:
            self._links.pop(key, None)
            self._link_local_corners.pop(key, None)

    def _export_ply_geometry_only(self, mesh, path):
        """Export a mesh as geometry-only PLY (positions + faces only).

        Vertex colour is carried by the BSDF in ``_scene_dict()``, not by PLY
        attributes. Keeping uint8 colour fields in PLY triggers Mitsuba warnings
        and those fields are ignored by this renderer.
        """
        import trimesh
        from trimesh.exchange import ply as ply_exchange

        clean = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            process=False)
        payload = ply_exchange.export_ply(
            clean, include_attributes=False, vertex_normal=False)
        with open(path, 'wb') as f:
            f.write(payload)

    def _register_link_entries(self, link):
        link_id = str(id(link))
        for source_name, mesh in self._mesh_sources(link):
            for sub_index, (sub, color) in enumerate(
                    self._mesh_entries_for_rendering(link, mesh)):
                key = '{}_{}_{}'.format(link_id, source_name, sub_index)
                if key in self._links:
                    continue
                ply = os.path.join(self._tmpdir, key + '.ply')
                self._export_ply_geometry_only(sub, ply)
                self._links[key] = (link, ply, color)
                self._link_local_corners[key] = self._aabb_corners(sub.bounds)

    # -- geometry management (matches the other viewers' add/delete API) --
    def _add_link(self, link):
        has_entries = any(v[0] is link for v in self._links.values())
        if not has_entries:
            self._register_link_entries(link)
        for child in link._child_links:
            self._add_link(child)

    def add(self, geometry, **kwargs):
        """Add a ``Link`` or ``CascadedLink`` to the scene.

        Backend-specific options the other viewers accept are taken for
        signature compatibility but have no effect here. In particular
        ``always_on_top`` (used by ``examples/skeleton_visualization.py``) is a
        rasteriser depth-test trick: a path tracer has no depth buffer to
        override, so such geometry is rendered normally and can be occluded.
        """
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError('geometry must be Link or CascadedLink')
        for link in links:
            self._add_link(link)

    def delete(self, geometry):
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError('geometry must be Link or CascadedLink')
        ids = {str(id(link)) for link in links}
        for key in [k for k, v in self._links.items()
                    if str(id(v[0])) in ids]:
            self._links.pop(key)
            self._link_local_corners.pop(key, None)

    def _unit_sphere_ply(self):
        # A triangulated unit sphere.  Mitsuba's analytic 'sphere' shape is not
        # supported by every backend (e.g. the Metal variant silently drops it),
        # whereas a triangle mesh renders on all of them.
        if getattr(self, '_sphere_ply', None) is None:
            import trimesh
            path = os.path.join(self._tmpdir, 'unit_sphere.ply')
            sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
            self._export_ply_geometry_only(sphere, path)
            self._sphere_ply = path
        return self._sphere_ply

    @staticmethod
    def _marker_color(entry):
        try:
            bsdf = entry['bsdf']
            alpha = 1.0
            if bsdf.get('type') == 'mask':
                alpha = _opacity_to_alpha(bsdf.get('opacity', 1.0))
                bsdf = bsdf['nested']
            rgb = np.asarray(bsdf['bsdf']['reflectance']['value'],
                             dtype=np.float64).reshape(-1)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            return np.asarray(
                [rgb[0], rgb[1], rgb[2], alpha], dtype=np.float64)
        except Exception:
            return None

    def _set_marker(self, name, ply, to_world, color, local_bounds):
        color = _rgba01(color)
        prev = self._extra.get(name)
        pose_only_update = False
        if prev is not None:
            prev_color = self._marker_color(prev)
            pose_only_update = (
                prev.get('type') == 'ply'
                and prev.get('filename') == ply
                and prev_color is not None
                and np.allclose(prev_color, color, atol=1e-12, rtol=0.0))
        self._extra[name] = {
            'type': 'ply', 'filename': ply,
            'to_world': to_world,
            'bsdf': _diffuse_bsdf(color)}
        self._extra_local_corners[name] = self._aabb_corners(local_bounds)
        if not pose_only_update:
            self._geom_version += 1
        return name

    def add_sphere(self, center, radius, color=(0.85, 0.1, 0.1), name=None):
        """Add a colored sphere marker (e.g. an obstacle or a target)."""
        name = name or 'sphere_{}'.format(len(self._extra))
        to_world = self.mi.ScalarTransform4f().translate(
            [float(v) for v in center]).scale(float(radius))
        return self._set_marker(
            name=name, ply=self._unit_sphere_ply(), to_world=to_world,
            color=color,
            local_bounds=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

    def add_box(self, center, extents, rotation=None,
                color=(0.55, 0.35, 0.18), name=None):
        """Add a colored box marker (e.g. a carried tray or object).

        Parameters
        ----------
        center : (3,) array
            World position of the box centre.
        extents : (3,) array
            Full side lengths of the box.
        rotation : (3, 3) array, optional
            World orientation. Defaults to identity (axis-aligned).
        """
        import trimesh
        name = name or 'box_{}'.format(len(self._extra))
        ekey = 'box_{:.4f}_{:.4f}_{:.4f}'.format(*[float(e) for e in extents])
        ply = os.path.join(self._tmpdir, ekey + '.ply')
        if not os.path.exists(ply):
            box = trimesh.creation.box(extents=[float(e) for e in extents])
            self._export_ply_geometry_only(box, ply)
        m = np.eye(4)
        if rotation is not None:
            m[:3, :3] = np.asarray(rotation, float)
        m[:3, 3] = [float(v) for v in center]
        half = 0.5 * np.asarray(extents, float)
        return self._set_marker(
            name=name, ply=ply,
            to_world=self.mi.ScalarTransform4f(m.tolist()),
            color=color, local_bounds=np.vstack([-half, half]))

    def _delete_marker(self, name):
        removed = False
        if name in self._extra:
            self._extra.pop(name)
            removed = True
        if name in self._extra_local_corners:
            self._extra_local_corners.pop(name)
            removed = True
        if removed:
            self._geom_version += 1
        return removed

    @staticmethod
    def _axis_marker_names(joint):
        joint_id = str(id(joint))
        return ('joint_axis_sphere_{}'.format(joint_id),
                'joint_axis_cylinder_{}'.format(joint_id))

    def add_joint_axis(self, joint, sphere_radius=0.01, axis_length=0.1,
                       axis_radius=0.003, axis_color=None):
        """Add joint axis visualization to the scene.

        Parameters
        ----------
        joint : Joint
            Joint object to visualize.
        sphere_radius : float, optional
            Radius of the sphere representing the joint position.
        axis_length : float, optional
            Length of the cylinder representing the joint axis.
        axis_radius : float, optional
            Radius of the cylinder representing the joint axis.
        axis_color : array-like, optional
            RGBA color for the axis cylinder. Defaults to red.
        """
        import trimesh

        from skrobot.model import Joint

        if not isinstance(joint, Joint):
            raise TypeError('joint must be a Joint object')
        if axis_color is None:
            axis_color = [1.0, 0.0, 0.0, 1.0]

        sphere_name, axis_name = self._axis_marker_names(joint)
        axis_length = abs(float(axis_length))
        position = np.asarray(joint.world_position, dtype=np.float64).reshape(3)
        self.add_sphere(
            center=position,
            radius=float(sphere_radius),
            color=(100.0 / 255.0, 100.0 / 255.0, 1.0, 1.0),
            name=sphere_name)
        if axis_length <= 1e-12:
            self._delete_marker(axis_name)
            self._joint_axis_map[str(id(joint))] = (sphere_name, None)
            return

        axis = getattr(joint, 'world_axis', None)
        if axis is None:
            self._delete_marker(axis_name)
            self._joint_axis_map[str(id(joint))] = (sphere_name, None)
            return
        axis = np.asarray(axis, dtype=np.float64).reshape(-1)
        if axis.size != 3:
            self._delete_marker(axis_name)
            self._joint_axis_map[str(id(joint))] = (sphere_name, None)
            return
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            self._delete_marker(axis_name)
            self._joint_axis_map[str(id(joint))] = (sphere_name, None)
            return
        direction = axis / axis_norm
        key = 'joint_axis_{:.6f}_{:.6f}'.format(
            axis_length, float(axis_radius))
        ply = os.path.join(self._tmpdir, key + '.ply')
        if not os.path.exists(ply):
            cylinder = trimesh.creation.cylinder(
                radius=float(axis_radius),
                segment=[[0.0, 0.0, -0.5 * axis_length],
                         [0.0, 0.0, 0.5 * axis_length]],
                sections=16)
            self._export_ply_geometry_only(cylinder, ply)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        rotation = np.eye(3)
        cosine = float(np.dot(z_axis, direction))
        if cosine < 1.0 - 1e-12:
            if cosine <= -1.0 + 1e-12:
                rotation = np.array([[1.0, 0.0, 0.0],
                                     [0.0, -1.0, 0.0],
                                     [0.0, 0.0, -1.0]], dtype=np.float64)
            else:
                cross = np.cross(z_axis, direction)
                cross_norm = float(np.linalg.norm(cross))
                if cross_norm > 1e-12:
                    axis_unit = cross / cross_norm
                    angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
                    K = np.array(
                        [[0.0, -axis_unit[2], axis_unit[1]],
                         [axis_unit[2], 0.0, -axis_unit[0]],
                         [-axis_unit[1], axis_unit[0], 0.0]],
                        dtype=np.float64)
                    rotation = (
                        np.eye(3)
                        + np.sin(angle) * K
                        + (1.0 - np.cos(angle)) * np.dot(K, K))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        self._set_marker(
            name=axis_name,
            ply=ply,
            to_world=self.mi.ScalarTransform4f(transform.tolist()),
            color=axis_color,
            local_bounds=[[-float(axis_radius), -float(axis_radius),
                           -0.5 * axis_length],
                          [float(axis_radius), float(axis_radius),
                           0.5 * axis_length]])
        self._joint_axis_map[str(id(joint))] = (sphere_name, axis_name)

    def delete_joint_axis(self, joint):
        joint_id = str(id(joint))
        sphere_name, axis_name = self._joint_axis_map.get(
            joint_id, self._axis_marker_names(joint))
        removed = self._delete_marker(sphere_name)
        if axis_name is not None:
            removed = self._delete_marker(axis_name) or removed
        self._joint_axis_map.pop(joint_id, None)
        return removed

    def _resolve_camera_view(self, angles, distance, center, fov,
                             coords_or_transform):
        """Resolve camera arguments to an ``(eye, target, up)`` tuple."""
        if coords_or_transform is not None:
            if isinstance(coords_or_transform, Coordinates):
                transform = coords_or_transform.worldcoords().T()
            else:
                transform = np.asarray(coords_or_transform, dtype=np.float64)
            eye = transform[:3, 3]
            forward = -transform[:3, 2]
            up = transform[:3, 1]
            return eye, eye + forward, up

        if angles is None:
            return None

        import trimesh
        rotation = trimesh.transformations.euler_matrix(*angles)
        points = self._collect_world_points()
        if points is None:
            bounds = np.zeros((2, 3), dtype=np.float64)
            # Keep the legacy fallback for empty scenes: avoid a degenerate
            # look-at target one unit away from the same point when no bounds
            # exist and no explicit distance is supplied.
            if distance is None:
                distance = 1.0
        else:
            bounds = np.vstack([points.min(axis=0), points.max(axis=0)])
        x_fov, y_fov = self._effective_sensor_fov_xy_radians()
        pose = trimesh.scene.cameras.look_at(
            points=bounds,
            fov=np.degrees([x_fov, y_fov]),
            rotation=rotation,
            distance=distance,
            center=center)
        eye = pose[:3, 3]
        if center is not None:
            target = np.asarray(center, dtype=np.float64).reshape(3)
        else:
            target = bounds.mean(axis=0)
        forward = -pose[:3, 2]
        up = pose[:3, 1]
        forward_norm = float(np.linalg.norm(forward))
        target_vec = target - eye
        target_norm = float(np.linalg.norm(target_vec))
        if forward_norm <= 1e-12 or target_norm <= 1e-12:
            raise RuntimeError(
                'Degenerate camera direction from trimesh.look_at '
                '(forward_norm={}, target_norm={}).'.format(
                    forward_norm, target_norm))
        forward_dir = forward / forward_norm
        target_dir = target_vec / target_norm
        if not np.allclose(forward_dir, target_dir, atol=1e-7, rtol=1e-6):
            raise RuntimeError(
                'Inconsistent camera direction from trimesh.look_at '
                '(forward_dir={}, target_dir={}).'.format(
                    forward_dir.tolist(), target_dir.tolist()))
        return eye, target, up

    def set_camera(self, angles=None, distance=None, center=None,
                   resolution=None, fov=None, coords_or_transform=None,
                   eye=None, target=None, up=(0, 0, 1)):
        """Set the camera pose.

        The signature mirrors :meth:`PyrenderViewer.set_camera` and
        :meth:`ViserViewer.set_camera` so the same call site works for any
        viewer backend, and adds the ``eye`` / ``target`` / ``up`` look-at form
        this renderer is usually driven with. When nothing usable is given the
        camera is left alone and auto-fits to the scene.

        Parameters
        ----------
        angles : array-like, optional
            Camera orientation as XYZ euler angles in radians, using the
            ``trimesh.transformations.euler_matrix`` convention shared with the
            trimesh / pyrender / viser viewers. First positional parameter, so
            ``set_camera([0, 0, np.pi / 2])`` means angles here too.
        distance : float, optional
            Distance from the camera to the look-at center. Estimated from the
            scene bounds and ``fov`` when omitted.
        center : array-like, optional
            World point to look at. Defaults to the center of the bounding box
            of the geometry currently in the scene.
        resolution : array-like, optional
            New output size ``(width, height)``. Takes effect on the next
            render.
        fov : float or sequence(float, float), optional
            Field of view in degrees, as in the constructor. It also feeds the
            ``distance`` estimate above when ``distance`` is omitted.
        coords_or_transform : Coordinates or (4, 4) array-like, optional
            Explicit camera world pose. Overrides ``angles`` / ``distance`` /
            ``center``.
        eye : array-like, optional
            Camera position. Used together with ``target``, and takes
            precedence over every argument above.
        target : array-like, optional
            World point the camera looks at, used together with ``eye``.
        up : array-like, optional
            Up vector for the ``eye`` / ``target`` form. Default ``(0, 0, 1)``.
        """
        if resolution is not None:
            new_resolution = tuple(int(v) for v in resolution)
            if new_resolution != self.resolution:
                self.resolution = new_resolution
                # Film size is compiled into the scene, so resize rebuilds.
                self._geom_version += 1
        if fov is not None and self._set_fov(fov):
            self._geom_version += 1
        if eye is not None and target is not None:
            self._camera = (np.asarray(eye, float), np.asarray(target, float),
                            np.asarray(up, float))
            return
        view = self._resolve_camera_view(
            angles, distance, center, fov, coords_or_transform)
        if view is None:
            return
        self._camera = (np.asarray(view[0], float), np.asarray(view[1], float),
                        np.asarray(view[2], float))

    # -- rendering --
    def _auto_camera(self):
        points = self._collect_world_points()
        if points is None:
            return np.array([1.5, -1.2, 1.0]), np.zeros(3), np.array([0, 0, 1.])
        center = points.min(axis=0) + 0.5 * np.ptp(points, axis=0)
        radius = max(0.3, 0.5 * float(np.linalg.norm(np.ptp(points, axis=0))))
        eye = center + radius * np.array([2.6, -2.2, 1.9])
        return eye, center, np.array([0, 0, 1.])

    def _scene_dict(self):
        mi = self.mi
        w, h = self.resolution
        eye, target, up = self._camera if self._camera is not None \
            else self._auto_camera()
        look = mi.ScalarTransform4f().look_at(
            origin=[float(v) for v in eye],
            target=[float(v) for v in target],
            up=[float(v) for v in up])
        sensor_fov, sensor_axis = self._effective_sensor_fov()
        sensor = {
            'type': 'perspective',
            'fov': sensor_fov,
            'to_world': look,
            'film': {'type': 'hdrfilm', 'width': w, 'height': h,
                     'rfilter': {'type': 'gaussian'}},
            'sampler': {'type': 'independent', 'sample_count': self.spp},
        }
        if sensor_axis is not None:
            sensor['fov_axis'] = sensor_axis
        d = {
            'type': 'scene',
            'integrator': {'type': 'path', 'max_depth': 8},
            'sensor': sensor,
            'ambient': {'type': 'constant',
                        'radiance': {'type': 'rgb',
                                     'value': self.ambient_light}},
        }
        if self.ground:
            center = target
            if self.light_size is None:
                sig = self._auto_ground_signature()
                if self._auto_light_r is None or self._auto_light_sig != sig:
                    self._auto_light_r = self._auto_light_radius()
                    self._auto_light_sig = sig
                # One-sided clamp: the ground plane has fixed world size, so a
                # smaller/lower key light darkens the frame across that fixed
                # floor. Large scenes needed scaling up; no measurement
                # justified scaling down.
                light_r = max(1.0, self._auto_light_r)
                light_size = 1.5 * light_r
            else:
                light_size = float(self.light_size)
                light_r = light_size / 1.5
            if self.ground_height is None:
                sig = self._auto_ground_signature()
                if self._auto_ground_z is None or self._auto_ground_sig != sig:
                    self._auto_ground_z = self._auto_ground_height()
                    self._auto_ground_sig = sig
                ground_z = self._auto_ground_z
            else:
                ground_z = float(self.ground_height)
            d['key'] = {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().look_at(
                    origin=[center[0] + 0.4 * light_r,
                            center[1] + 0.8 * light_r,
                            center[2] + 2.0 * light_r],
                    target=[float(v) for v in center],
                    up=[1, 0, 0]).scale(light_size),
                'emitter': {'type': 'area',
                            'radiance': {'type': 'rgb',
                                         'value': self.light_intensity}},
                # Do not rescale radiance when size/height scale together: for
                # area lights, irradiance at the subject is ~ area / distance^2,
                # so linear size scaling keeps exposure nearly unchanged.
                # The key light is a rectangle hovering above the subject, and
                # Mitsuba's area emitter is one-sided: from behind it emits
                # nothing while still being an opaque occluder. A
                # camera that ends up on its back side -- which any user-chosen
                # view can, e.g. examples/pr2_inverse_kinematics.py's
                # set_camera([45deg, 0, 135deg], distance=2.5) -- therefore saw
                # the black underside filling the whole frame. A ``null`` BSDF
                # lets camera rays pass straight through the plate, so it lights
                # the scene without ever occluding it.
                'bsdf': {'type': 'null'}}
            d['ground'] = {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().translate(
                    [center[0], center[1], ground_z]).scale(self.ground_size),
                'bsdf': _diffuse_bsdf([0.45, 0.47, 0.5])}
        for key, (link, ply, color) in self._links.items():
            d['m_' + key] = {
                'type': 'ply', 'filename': ply,
                'to_world': mi.ScalarTransform4f(
                    np.asarray(link.worldcoords().T(), np.float64).tolist()),
                'bsdf': _diffuse_bsdf(color)}
        d.update(self._extra)
        return d

    def _load_scene_dict(self):
        """Compile the scene dict into a Mitsuba scene.

        ``optimize=False`` disables Mitsuba's shape-merging optimization, which
        would otherwise collapse geometry that shares the same mesh *and* colour
        (e.g. a robot's symmetric left/right parts, or a marker box that happens
        to match one) into a single shape. A merged shape exposes no per-object
        ``vertex_positions`` in ``traverse``, so those objects could not be moved
        independently and would freeze at their initial pose during redraws.
        Keeping every object as its own shape is what makes the incremental
        transform update below correct for links and markers.
        """
        try:
            return self.mi.load_dict(self._scene_dict(), optimize=False)
        except TypeError:
            # Older Mitsuba builds lack the ``optimize`` keyword. Fall back to
            # the default loader; any shapes it merges are still moved correctly
            # (via a full rebuild) by the untracked-object guard in
            # _render_scene, just less efficiently.
            return self.mi.load_dict(self._scene_dict())

    def _build_scene(self):
        """Compile the scene once and remember each mesh's local geometry
        so later frames update only vertex transforms (no reload)."""
        mi = self.mi
        self._scene = self._load_scene_dict()
        self._params = mi.traverse(self._scene)
        self._mesh_local = {}
        self._untracked = {}
        for key, (link, _ply, _color) in self._links.items():
            vpk = 'm_' + key + '.vertex_positions'
            if vpk not in self._params:
                # Mitsuba merged this link's shape with an identical one, so it
                # has no independently updatable vertices. Remember it and its
                # current pose so _render_scene rebuilds when it actually moves,
                # instead of silently leaving it frozen at the wrong place.
                self._untracked['link:' + key] = {'kind': 'link', 'link': link}
                continue
            v0 = np.array(self._params[vpk]).reshape(-1, 3)
            coords = link.worldcoords()
            entry = {'kind': 'link', 'link': link, 'vpk': vpk,
                     'v': coords.inverse_transform_vector(v0)}
            nk = 'm_' + key + '.vertex_normals'
            if nk in self._params:
                n0 = np.array(self._params[nk]).reshape(-1, 3)
                entry['nk'] = nk
                entry['n'] = coords.inverse_rotate_vector(n0)
            self._mesh_local['link:' + key] = entry
        for name, marker in self._extra.items():
            if marker.get('type') != 'ply' or 'to_world' not in marker:
                continue
            vpk = name + '.vertex_positions'
            ukey = 'marker:' + name
            if vpk not in self._params:
                # Same merged-shape guard as links: if a marker was merged and
                # lost per-shape vertices, rebuild when it moves so it never
                # freezes at its initial pose.
                self._untracked[ukey] = {'kind': 'marker', 'name': name}
                continue
            v0 = np.array(self._params[vpk]).reshape(-1, 3)
            T = np.asarray(marker['to_world'].matrix, float)
            T_inv = np.linalg.inv(T)
            # Marker transforms are not always rigid (sphere uses uniform
            # scaling), so recover local points with the full 4x4 inverse.
            entry = {'kind': 'marker', 'name': name, 'vpk': vpk,
                     'v': self._transform_points(v0, T_inv)}
            nk = name + '.vertex_normals'
            if nk in self._params:
                n0 = np.array(self._params[nk]).reshape(-1, 3)
                entry['nk'] = nk
                entry['n'] = np.dot(n0, T[:3, :3])
            self._mesh_local[ukey] = entry
        self._untracked_T = {}
        for key, entry in self._untracked.items():
            if entry['kind'] == 'link':
                T = np.asarray(entry['link'].worldcoords().T(), float)
            else:
                marker = self._extra.get(entry['name'])
                T = np.asarray(marker['to_world'].matrix, float)
            self._untracked_T[key] = T
        self._link_sig = tuple(sorted(self._links.keys()))
        self._built_version = self._geom_version

    def _untracked_moved(self):
        """Whether any merged link/marker moved since the scene was built.

        Merged shapes have no per-shape vertex parameters in ``traverse`` and
        cannot be updated in place, so moving one requires a full rebuild.
        """
        for key, entry in self._untracked.items():
            if entry['kind'] == 'link':
                T = np.asarray(entry['link'].worldcoords().T(), float)
            else:
                marker = self._extra.get(entry['name'])
                if marker is None or 'to_world' not in marker:
                    return True
                T = np.asarray(marker['to_world'].matrix, float)
            if not np.allclose(T, self._untracked_T[key], atol=1e-9):
                return True
        return False

    def _update_transforms(self):
        mi = self.mi
        for entry in self._mesh_local.values():
            if entry['kind'] == 'link':
                coords = entry['link'].worldcoords()
                self._params[entry['vpk']] = mi.Float(
                    coords.transform_vector(entry['v']).ravel())
                if 'nk' in entry:
                    self._params[entry['nk']] = mi.Float(
                        coords.rotate_vector(entry['n']).ravel())
                continue
            marker = self._extra.get(entry['name'])
            if marker is None or 'to_world' not in marker:
                continue
            T = np.asarray(marker['to_world'].matrix, float)
            self._params[entry['vpk']] = mi.Float(
                self._transform_points(entry['v'], T).ravel())
            if 'nk' in entry:
                # A scaled marker changes normal length; renormalise after
                # applying the inverse-linear normal transform.
                n = np.dot(entry['n'], np.linalg.inv(T[:3, :3]))
                n_norm = np.linalg.norm(n, axis=1, keepdims=True)
                n_norm[n_norm == 0.0] = 1.0
                self._params[entry['nk']] = mi.Float((n / n_norm).ravel())
        if 'sensor.to_world' in self._params:
            eye, target, up = self._effective_camera()
            self._params['sensor.to_world'] = mi.Transform4f(
                mi.ScalarTransform4f().look_at(
                    origin=[float(v) for v in eye],
                    target=[float(v) for v in target],
                    up=[float(v) for v in up]))
        self._params.update()

    def _refresh_changed_link_meshes(self):
        """Refresh tracked link geometry for links marked visual-mesh-changed.

        ``Link.set_color`` / ``reset_color`` mutate ``concatenated_visual_mesh``
        and raise the ``visual_mesh_changed`` flag. Re-exporting only those
        links keeps incremental redraw fast while making post-add colour updates
        visible in the next render.
        """
        changed = False
        seen_links = set()
        for key, (link, _ply, _color) in list(self._links.items()):
            link_id = id(link)
            if link_id in seen_links:
                continue
            seen_links.add(link_id)
            if not getattr(link, 'visual_mesh_changed', False):
                continue
            self._clear_link_entries(link)
            self._register_link_entries(link)
            changed = True
            link._visual_mesh_changed = False
        if changed:
            self._geom_version += 1

    def _render_scene(self, spp):
        mi = self.mi
        self._refresh_changed_link_meshes()
        sig = tuple(sorted(self._links.keys()))
        if (self._scene is None or sig != self._link_sig
                or self._geom_version != self._built_version
                or self._untracked_moved()):
            self._build_scene()
        else:
            self._update_transforms()
        img = mi.render(self._scene, spp=spp)
        rgb = np.array(mi.util.convert_to_bitmap(img))[..., :3]
        self._last_image = rgb
        return rgb

    def render(self):
        """Render the current scene and return an ``(H, W, 3)`` uint8 array."""
        return self._render_scene(self.spp)

    def save_image(self, file_obj):
        """Render and write the image to ``file_obj`` (path or file handle)."""
        from PIL import Image
        Image.fromarray(self.render()).save(file_obj)

    # -- interactive-ish display (re-render on redraw; NOT a real-time GL
    #    viewer -- Mitsuba is a path tracer, so updates take a moment) --
    def _effective_camera(self):
        if self._camera is not None:
            return (np.asarray(self._camera[0], float),
                    np.asarray(self._camera[1], float),
                    np.asarray(self._camera[2], float))
        return self._auto_camera()

    @staticmethod
    def _in_notebook():
        try:
            from IPython import get_ipython
            return type(get_ipython()).__name__ == 'ZMQInteractiveShell'
        except Exception:
            return False

    def show(self, block=False):
        """Display the render and keep it updatable with :meth:`redraw`.

        In a Jupyter notebook the image is shown inline; otherwise a
        matplotlib window opens. Drag to orbit the camera (a fast low-quality
        preview is shown while dragging, then a full-quality frame on release)
        and scroll to zoom. Because Mitsuba path-traces every frame this is a
        quality previewer, not a real-time viewer.
        """
        img = self.render()
        if self._in_notebook():
            from IPython.display import display
            from PIL import Image
            self._nb_handle = display(Image.fromarray(img), display_id=True)
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                'MitsubaViewer.show() needs matplotlib for the interactive '
                "window. Install it with 'pip install matplotlib' (it is "
                'included in the scikit-robot[all] extra). render() and '
                'save_image() work without it.')
        plt.ion()
        w, h = self.resolution
        self._fig, self._ax = plt.subplots(figsize=(w / 100.0, h / 100.0))
        self._ax.set_axis_off()
        self._im = self._ax.imshow(img)
        self._fig.tight_layout(pad=0)
        self._init_orbit()
        cv = self._fig.canvas
        cv.mpl_connect('button_press_event', self._on_press)
        cv.mpl_connect('motion_notify_event', self._on_motion)
        cv.mpl_connect('button_release_event', self._on_release)
        cv.mpl_connect('scroll_event', self._on_scroll)
        self._fig.show()
        cv.draw_idle()
        cv.flush_events()
        if block:
            self.wait_until_close()

    def redraw(self):
        """Re-render at full quality and refresh the displayed image."""
        self._update_image(self.render())

    def _update_image(self, img):
        if getattr(self, '_nb_handle', None) is not None:
            from PIL import Image
            self._nb_handle.update(Image.fromarray(img))
        elif getattr(self, '_im', None) is not None:
            self._im.set_data(img)
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    # -- orbit controls (matplotlib window only) --
    def _init_orbit(self):
        eye, target, up = self._effective_camera()
        offset = eye - target
        self._orbit_target = target
        self._orbit_up = up
        self._orbit_dist = float(np.linalg.norm(offset))
        self._orbit_az = float(np.arctan2(offset[1], offset[0]))
        self._orbit_el = float(np.arcsin(
            np.clip(offset[2] / max(self._orbit_dist, 1e-9), -1.0, 1.0)))
        self._drag_xy = None

    def _apply_orbit(self):
        el = float(np.clip(self._orbit_el, -1.4, 1.4))
        r = self._orbit_dist
        offset = np.array([r * np.cos(el) * np.cos(self._orbit_az),
                           r * np.cos(el) * np.sin(self._orbit_az),
                           r * np.sin(el)])
        self._camera = (self._orbit_target + offset, self._orbit_target,
                        self._orbit_up)

    def _on_press(self, event):
        if event.inaxes is self._ax:
            self._drag_xy = (event.x, event.y)

    def _on_motion(self, event):
        if self._drag_xy is None or event.x is None:
            return
        dx = event.x - self._drag_xy[0]
        dy = event.y - self._drag_xy[1]
        self._drag_xy = (event.x, event.y)
        self._orbit_az -= dx * 0.01
        self._orbit_el += dy * 0.01
        self._apply_orbit()
        # fast, low-quality preview while dragging
        self._update_image(self._render_scene(min(self.spp, 4)))

    def _on_release(self, event):
        if self._drag_xy is None:
            return
        self._drag_xy = None
        self._update_image(self.render())          # full quality on release

    def _on_scroll(self, event):
        step = 0.9 if event.button == 'up' else 1.1
        self._orbit_dist = max(0.1, self._orbit_dist * step)
        self._apply_orbit()
        self._update_image(self.render())

    @property
    def is_active(self):
        """Whether a display is open (so animation loops know when to stop)."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            return plt.fignum_exists(self._fig.number)
        return self._nb_handle is not None

    @property
    def has_exit(self):
        """Whether the display has been closed by the user."""
        return not self.is_active

    def _preview_redraw(self):
        self._update_image(self._render_scene(min(self.spp, 16)))

    def pause(self, duration=0.001, fps=30.0):
        """Re-render the (possibly moved) scene and keep the window responsive.

        This mirrors the interactive viewers' ``pause`` so animation loops work
        unchanged. Rendering uses a capped sample count for a live-preview feel;
        it is re-render based, so it is smooth only on the GPU (Metal / CUDA)
        variants.
        """
        if not (math.isfinite(fps) and fps > 0):
            raise ValueError(
                'fps must be a positive finite number, got {}.'.format(fps))
        self._preview_redraw()
        if not (math.isfinite(duration) and duration > 0):
            return
        interval = 1.0 / float(fps)
        end = time.monotonic() + float(duration)
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            sleep_time = min(interval, remaining)
            if self._fig is not None:
                import matplotlib.pyplot as plt
                plt.pause(max(sleep_time, 1e-3))
            else:
                time.sleep(sleep_time)
            self._preview_redraw()

    def wait_until_close(self, redraw=True, interval=0.1,
                         message='==> Press [q] to close window'):
        """Block until the window is closed by the user.

        The signature matches
        :meth:`skrobot.viewers._base._InteractiveViewerMixin.wait_until_close`,
        which the trimesh and pyrender viewers use, so a call site written for
        those backends works here too -- ``examples/skeleton_visualization.py``
        passes ``message=`` and used to raise ``TypeError`` on this viewer.

        Parameters
        ----------
        redraw : bool, optional
            Unused. The mixin re-renders on a timer to keep a GL window
            responsive; the matplotlib window this viewer opens pumps its own
            event loop, and path-tracing a frame every ``interval`` seconds
            would be far too slow to do on a timer anyway.
        interval : float, optional
            Unused, for the same reason.
        message : str or None, optional
            Message printed once before waiting starts. Pass ``None`` to
            suppress it.
        """
        if message:
            print(message)
        if getattr(self, '_fig', None) is not None:
            import matplotlib.pyplot as plt
            plt.show(block=True)

    def close(self):
        if getattr(self, '_fig', None) is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = self._ax = self._im = None
        self._nb_handle = None

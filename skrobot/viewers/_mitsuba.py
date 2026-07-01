import os
import tempfile

import numpy as np

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


def _mesh_color(mesh):
    """Best-effort average base color of a trimesh mesh, in [0, 1]."""
    default = np.array([0.75, 0.76, 0.78])
    visual = getattr(mesh, 'visual', None)
    if visual is None:
        return default
    try:
        kind = getattr(visual, 'kind', None)
        if kind in ('vertex', 'face') or hasattr(visual, 'vertex_colors'):
            colors = np.asarray(visual.vertex_colors, dtype=np.float64)
            if colors.size:
                c = colors[:, :3].mean(axis=0)
                return np.clip(c / 255.0 if c.max() > 1.0 else c, 0.0, 1.0)
        material = getattr(visual, 'material', None)
        if material is not None:
            col = getattr(material, 'main_color', None)
            if col is None:
                col = getattr(material, 'baseColorFactor', None)
            if col is not None:
                c = np.asarray(col, dtype=np.float64)[:3]
                return np.clip(c / 255.0 if c.max() > 1.0 else c, 0.0, 1.0)
    except Exception:
        pass
    return default


class MitsubaViewer(object):
    """Headless offscreen renderer backed by Mitsuba 3.

    Unlike :class:`~skrobot.viewers.PyrenderViewer`, this viewer opens no window
    and needs no display server / OpenGL context, so it works over SSH, in CI,
    and on macOS where offscreen OpenGL is unavailable.  It renders the same
    ``Link`` / ``CascadedLink`` geometry the other viewers accept, using
    Mitsuba's path tracer, and writes the result with :meth:`save_image` (or
    returns it from :meth:`render`).

    Parameters
    ----------
    resolution : tuple(int, int), optional
        Output image size ``(width, height)``. Default ``(640, 480)``.
    spp : int, optional
        Samples per pixel for the path tracer. Higher is less noisy but
        slower. Default ``64``.
    variant : str, optional
        Mitsuba variant to use. When ``None`` (default) it auto-selects the
        Apple-GPU ``'metal_ad_rgb'`` variant on macOS and the CPU
        ``'llvm_ad_rgb'`` variant elsewhere. Pass ``'metal_ad_rgb'`` (Apple
        GPU), ``'cuda_ad_rgb'`` (NVIDIA) or ``'llvm_ad_rgb'`` (CPU) to force one
        (or set the ``SKROBOT_MITSUBA_VARIANT`` environment variable).
    ground : bool, optional
        If ``True`` (default) add a neutral ground plane and a key light.
    """

    def __init__(self, resolution=(640, 480), spp=64, variant=None,
                 ground=True):
        self.mi = _load_mitsuba(variant)
        self.resolution = tuple(resolution)
        self.spp = int(spp)
        self.ground = ground
        self._links = {}                 # mesh_id -> (link, ply_path, color)
        self._extra = {}                 # name -> scene-dict entry
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
        self._link_sig = None
        self._geom_version = 0           # bumped when markers change
        self._built_version = -1

    # -- geometry management (matches the other viewers' add/delete API) --
    def _add_link(self, link):
        link_id = str(id(link))
        # Use the per-sub-mesh visual meshes rather than the concatenated one:
        # each sub-mesh is (essentially) a single material colour, so a per-mesh
        # colour keeps distinct parts (e.g. white body vs. black gripper)
        # instead of averaging them into one grey.
        mesh = link.visual_mesh
        meshes = mesh if isinstance(mesh, (list, tuple)) else [mesh]
        for m in meshes:
            if m is None or len(m.faces) == 0:
                continue
            key = link_id + str(id(m))
            if key in self._links:
                continue
            ply = os.path.join(self._tmpdir, key + '.ply')
            m.copy().export(ply)          # local frame, exported once
            self._links[key] = (link, ply, _mesh_color(m))
        for child in link._child_links:
            self._add_link(child)

    def add(self, geometry, **kwargs):
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

    def _unit_sphere_ply(self):
        # A triangulated unit sphere.  Mitsuba's analytic 'sphere' shape is not
        # supported by every backend (e.g. the Metal variant silently drops it),
        # whereas a triangle mesh renders on all of them.
        if getattr(self, '_sphere_ply', None) is None:
            import trimesh
            path = os.path.join(self._tmpdir, 'unit_sphere.ply')
            trimesh.creation.icosphere(subdivisions=3, radius=1.0).export(path)
            self._sphere_ply = path
        return self._sphere_ply

    def add_sphere(self, center, radius, color=(0.85, 0.1, 0.1), name=None):
        """Add a colored sphere marker (e.g. an obstacle or a target)."""
        name = name or 'sphere_{}'.format(len(self._extra))
        to_world = self.mi.ScalarTransform4f().translate(
            [float(v) for v in center]).scale(float(radius))
        self._extra[name] = {
            'type': 'ply', 'filename': self._unit_sphere_ply(),
            'to_world': to_world,
            'bsdf': {'type': 'diffuse',
                     'reflectance': {'type': 'rgb', 'value': list(color)}}}
        self._geom_version += 1
        return name

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
            trimesh.creation.box(extents=[float(e) for e in extents]).export(ply)
        m = np.eye(4)
        if rotation is not None:
            m[:3, :3] = np.asarray(rotation, float)
        m[:3, 3] = [float(v) for v in center]
        self._extra[name] = {
            'type': 'ply', 'filename': ply,
            'to_world': self.mi.ScalarTransform4f(m.tolist()),
            'bsdf': {'type': 'diffuse',
                     'reflectance': {'type': 'rgb', 'value': list(color)}}}
        self._geom_version += 1
        return name

    def set_camera(self, eye=None, target=None, up=(0, 0, 1), **kwargs):
        """Set an explicit look-at camera. If unset, the camera auto-fits."""
        if eye is not None and target is not None:
            self._camera = (np.asarray(eye, float), np.asarray(target, float),
                            np.asarray(up, float))

    # -- rendering --
    def _auto_camera(self):
        pts = []
        for link, _, _ in self._links.values():
            pts.append(link.worldcoords().worldpos())
        if not pts:
            return np.array([1.5, -1.2, 1.0]), np.zeros(3), np.array([0, 0, 1.])
        pts = np.array(pts)
        center = pts.mean(axis=0)
        radius = max(0.3, np.linalg.norm(pts - center, axis=1).max())
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
        d = {
            'type': 'scene',
            'integrator': {'type': 'path', 'max_depth': 8},
            'sensor': {
                'type': 'perspective', 'fov': 45, 'to_world': look,
                'film': {'type': 'hdrfilm', 'width': w, 'height': h,
                         'rfilter': {'type': 'gaussian'}},
                'sampler': {'type': 'independent', 'sample_count': self.spp}},
            'ambient': {'type': 'constant',
                        'radiance': {'type': 'rgb', 'value': 0.12}},
        }
        if self.ground:
            center = target
            d['key'] = {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().look_at(
                    origin=[center[0] + 0.4, center[1] + 0.8, center[2] + 2.0],
                    target=[float(v) for v in center],
                    up=[1, 0, 0]).scale(1.5),
                'emitter': {'type': 'area',
                            'radiance': {'type': 'rgb', 'value': 5.0}}}
            d['ground'] = {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().translate(
                    [center[0], center[1], 0.0]).scale(6),
                'bsdf': {'type': 'diffuse',
                         'reflectance': {'type': 'rgb', 'value': [0.45, 0.47, 0.5]}}}
        for key, (link, ply, color) in self._links.items():
            d['m_' + key] = {
                'type': 'ply', 'filename': ply,
                'to_world': mi.ScalarTransform4f(
                    np.asarray(link.worldcoords().T(), np.float64).tolist()),
                'bsdf': {'type': 'diffuse',
                         'reflectance': {'type': 'rgb', 'value': list(color)}}}
        d.update(self._extra)
        return d

    def _build_scene(self):
        """Compile the scene once and remember each link mesh's local geometry
        so later frames update only vertex transforms (no reload)."""
        mi = self.mi
        self._scene = mi.load_dict(self._scene_dict())
        self._params = mi.traverse(self._scene)
        self._mesh_local = {}
        for key, (link, _ply, _color) in self._links.items():
            vpk = 'm_' + key + '.vertex_positions'
            if vpk not in self._params:
                continue
            v0 = np.array(self._params[vpk]).reshape(-1, 3)
            T = np.asarray(link.worldcoords().T(), float)
            R, t = T[:3, :3], T[:3, 3]
            entry = {'link': link, 'vpk': vpk, 'v': (v0 - t) @ R}
            nk = 'm_' + key + '.vertex_normals'
            if nk in self._params:
                n0 = np.array(self._params[nk]).reshape(-1, 3)
                entry['nk'] = nk
                entry['n'] = n0 @ R
            self._mesh_local[key] = entry
        self._link_sig = tuple(sorted(self._links.keys()))
        self._built_version = self._geom_version

    def _update_transforms(self):
        mi = self.mi
        for entry in self._mesh_local.values():
            T = np.asarray(entry['link'].worldcoords().T(), float)
            R, t = T[:3, :3], T[:3, 3]
            self._params[entry['vpk']] = mi.Float(
                (entry['v'] @ R.T + t).ravel())
            if 'nk' in entry:
                self._params[entry['nk']] = mi.Float(
                    (entry['n'] @ R.T).ravel())
        if 'sensor.to_world' in self._params:
            eye, target, up = self._effective_camera()
            self._params['sensor.to_world'] = mi.Transform4f(
                mi.ScalarTransform4f().look_at(
                    origin=[float(v) for v in eye],
                    target=[float(v) for v in target],
                    up=[float(v) for v in up]))
        self._params.update()

    def _render_scene(self, spp):
        mi = self.mi
        sig = tuple(sorted(self._links.keys()))
        if (self._scene is None or sig != self._link_sig
                or self._geom_version != self._built_version):
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

    def pause(self, duration=0.001, fps=30.0):
        """Re-render the (possibly moved) scene and keep the window responsive.

        This mirrors the interactive viewers' ``pause`` so animation loops work
        unchanged. Rendering uses a capped sample count for a live-preview feel;
        it is re-render based, so it is smooth only on the GPU (Metal / CUDA)
        variants.
        """
        self._update_image(self._render_scene(min(self.spp, 16)))
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.pause(max(float(duration), 1e-3))

    def wait_until_close(self):
        """Block until the window is closed (matplotlib window only)."""
        if getattr(self, '_fig', None) is not None:
            import matplotlib.pyplot as plt
            plt.show(block=True)

    def close(self):
        if getattr(self, '_fig', None) is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = self._ax = self._im = None
        self._nb_handle = None

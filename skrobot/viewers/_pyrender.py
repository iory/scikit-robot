from __future__ import division

import collections
import threading

import numpy as np
import pyglet
from pyglet import compat_platform
import pyrender
from pyrender.trackball import Trackball
import trimesh
from trimesh import transformations
from trimesh.scene import cameras

from skrobot.coordinates import Coordinates

from .. import model as model_module


def _redraw_all_windows():
    try:
        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event('on_draw')
            window.flip()
            window._legacy_invalid = False
    except RuntimeError:
        pass


class PyrenderViewer(pyrender.Viewer):

    """PyrenderViewer class implemented as a Singleton.

    This ensures that only one instance of the viewer
    is created throughout the program. Any subsequent attempts to create a new
    instance will return the existing one.

    Parameters
    ----------
    resolution : tuple, optional
        The resolution of the viewer. Default is (640, 480).
    update_interval : float, optional
        The update interval (in seconds) for the viewer. Default is
        1.0 seconds.

    Notes
    -----
    Since this is a singleton, the __init__ method might be called
    multiple times, but only one instance is actually used.
    """

    # Class variable to hold the single instance of the class.
    _instance = None

    def __init__(self, resolution=None, update_interval=1.0,
                 render_flags=None):
        if getattr(self, '_initialized', False):
            return
        if resolution is None:
            resolution = (640, 480)

        self.thread = None
        self._visual_mesh_map = collections.OrderedDict()

        self._redraw = True

        refresh_rate = 1.0 / update_interval
        self._kwargs = dict(
            scene=pyrender.Scene(),
            viewport_size=resolution,
            run_in_thread=False,
            use_raymond_lighting=True,
            auto_start=False,
            render_flags=render_flags,
            refresh_rate=refresh_rate,
        )
        super(PyrenderViewer, self).__init__(**self._kwargs)
        self.viewer_flags['window_title'] = 'scikit-robot PyrenderViewer'
        self._initialized = True

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PyrenderViewer, cls).__new__(cls)
        return cls._instance

    def show(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.set_camera([np.deg2rad(45), -np.deg2rad(0), np.deg2rad(135)])
        if compat_platform == 'darwin':
            self._init_and_start_app()
            init_loop = 30
            for _ in range(init_loop):
                _redraw_all_windows()
        else:
            self.thread = threading.Thread(target=self._init_and_start_app)
            self.thread.daemon = True  # terminate when main thread exit
            self.thread.start()

    def _init_and_start_app(self):
        # Try multiple configs starting with target OpenGL version
        # and multisampling and removing these options if exception
        # Note: multisampling not available on all hardware
        from pyglet import clock
        from pyglet.gl import Config
        from pyrender.constants import MIN_OPEN_GL_MAJOR
        from pyrender.constants import MIN_OPEN_GL_MINOR
        from pyrender.constants import TARGET_OPEN_GL_MAJOR
        from pyrender.constants import TARGET_OPEN_GL_MINOR
        from pyrender.viewer import Viewer
        confs = [Config(sample_buffers=1, samples=4,
                        depth_size=24,
                        double_buffer=True,
                        major_version=TARGET_OPEN_GL_MAJOR,
                        minor_version=TARGET_OPEN_GL_MINOR),
                 Config(depth_size=24,
                        double_buffer=True,
                        major_version=TARGET_OPEN_GL_MAJOR,
                        minor_version=TARGET_OPEN_GL_MINOR),
                 Config(sample_buffers=1, samples=4,
                        depth_size=24,
                        double_buffer=True,
                        major_version=MIN_OPEN_GL_MAJOR,
                        minor_version=MIN_OPEN_GL_MINOR),
                 Config(depth_size=24,
                        double_buffer=True,
                        major_version=MIN_OPEN_GL_MAJOR,
                        minor_version=MIN_OPEN_GL_MINOR)]
        for conf in confs:
            try:
                super(Viewer, self).__init__(config=conf, resizable=True,
                                             width=self._viewport_size[0],
                                             height=self._viewport_size[1])
                break
            except pyglet.window.NoSuchConfigException:
                pass
            except pyglet.canvas.xlib.NoSuchDisplayException:
                print('No display found. Viewer is disabled.')
                self.has_exit = True
                return

        if not self.context:
            raise ValueError('Unable to initialize an OpenGL 3+ context')
        clock.schedule_interval(
            Viewer._time_event, 1.0 / self.viewer_flags['refresh_rate'], self
        )
        self.switch_to()
        self.set_caption(self.viewer_flags['window_title'])
        if compat_platform != 'darwin':
            pyglet.app.run()

    def redraw(self):
        self._redraw = True
        if compat_platform == 'darwin':
            _redraw_all_windows()

    def on_draw(self):
        if not self._redraw:
            with self._render_lock:
                super(PyrenderViewer, self).on_draw()
            return

        with self._render_lock:
            # apply latest angle-vector
            for link_id, (node, link) in self._visual_mesh_map.items():
                link.update(force=True)
                transform = link.worldcoords().T()
                if link.visual_mesh_changed:
                    mesh = link.concatenated_visual_mesh
                    pyrender_mesh = pyrender.Mesh.from_trimesh(
                        mesh, smooth=False)
                    self.scene.remove_node(node)
                    node = self.scene.add(pyrender_mesh, pose=transform)
                    self._visual_mesh_map[link_id] = (node, link)
                    link._visual_mesh_changed = False
                else:
                    node.matrix = transform
            super(PyrenderViewer, self).on_draw()

        self._redraw = False

    def on_mouse_press(self, *args, **kwargs):
        self._redraw = True
        return super(PyrenderViewer, self).on_mouse_press(*args, **kwargs)

    def on_mouse_drag(self, *args, **kwargs):
        self._redraw = True
        return super(PyrenderViewer, self).on_mouse_drag(*args, **kwargs)

    def on_mouse_scroll(self, *args, **kwargs):
        self._redraw = True
        return super(PyrenderViewer, self).on_mouse_scroll(*args, **kwargs)

    def on_key_press(self, *args, **kwargs):
        self._redraw = True
        return super(PyrenderViewer, self).on_key_press(*args, **kwargs)

    def on_resize(self, *args, **kwargs):
        self._redraw = True
        return super(PyrenderViewer, self).on_resize(*args, **kwargs)

    def _add_link(self, link):
        assert isinstance(link, model_module.Link)

        with self._render_lock:
            transform = link.worldcoords().T()
            link_id = str(id(link))
            mesh = link.concatenated_visual_mesh

            if link_id not in self._visual_mesh_map and mesh:
                node = None
                if isinstance(mesh, trimesh.path.Path3D):
                    pyrender_mesh = pyrender.Mesh(
                        primitives=[pyrender.Primitive(
                            mesh.vertices[mesh.vertex_nodes].reshape(-1, 3),
                            mode=pyrender.constants.GLTF.LINE_STRIP,
                            color_0=mesh.colors)])
                    node = self.scene.add(pyrender_mesh)
                elif isinstance(mesh, trimesh.PointCloud):
                    pyrender_mesh = pyrender.Mesh(
                        primitives=[pyrender.Primitive(
                            mesh.vertices,
                            mode=pyrender.constants.GLTF.POINTS,
                            color_0=mesh.colors)])
                    node = self.scene.add(pyrender_mesh)
                else:
                    pyrender_mesh = pyrender.Mesh.from_trimesh(
                        mesh, smooth=False)
                    # Check if the mesh has vertices
                    # before adding it to the scene
                    if len(mesh.vertices) != 0:
                        node = self.scene.add(pyrender_mesh, pose=transform)
                # Add the node and link to the
                # visual mesh map only if the node is successfully created
                if node is not None:
                    self._visual_mesh_map[link_id] = (node, link)

        for child_link in link._child_links:
            self._add_link(child_link)

    def add(self, geometry):
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError('geometry must be Link or CascadedLink')

        for link in links:
            self._add_link(link)

        self._redraw = True

    def delete(self, geometry):
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError('geometry must be Link or CascadedLink')

        with self._render_lock:
            all_links = links
            while all_links:
                link = all_links[0]
                link_id = str(id(link))
                if link_id in self._visual_mesh_map:
                    self.scene.remove_node(self._visual_mesh_map[link_id][0])
                    self._visual_mesh_map.pop(link_id)
                all_links = all_links[1:]
                all_links.extend(link.child_links)
        self._redraw = True

    def set_camera(self, angles=None, distance=None, center=None,
                   resolution=None, fov=None, coords_or_transform=None):
        if angles is None and coords_or_transform is None:
            return
        if angles is not None:
            if fov is None:
                fov = np.array([60, 45])
            rotation = transformations.euler_matrix(*angles)
            pose = cameras.look_at(
                self.scene.bounds, fov=fov, rotation=rotation,
                distance=distance, center=center)
        else:
            if isinstance(coords_or_transform, Coordinates):
                pose = coords_or_transform.worldcoords().T()
        self._camera_node.matrix = pose
        self._trackball = Trackball(
            pose=pose,
            size=self.viewport_size,
            scale=self.scene.scale,
            target=self.scene.centroid
        )

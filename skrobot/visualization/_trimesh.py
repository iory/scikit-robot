import threading
import warnings

import pyglet
from pyglet import gl
import trimesh.viewer

from .. import robot_model as robot_model_module


class SceneViewer(trimesh.viewer.SceneViewer):

    def __init__(self, *args, **kwargs):
        if len(args) > 6:
            args = list(args)
            if args[5] is False:
                warnings.warn('start_loop must be always True')
            args[5] = True  # start_loop
            args = tuple(args)
        else:
            if 'start_loop' in kwargs and kwargs['start_loop'] is False:
                warnings.warn('start_loop must be always True')
            kwargs['start_loop'] = True

        self._links = []

        self._redraw = True
        self._camera_transform = None
        pyglet.clock.schedule_interval(self.on_update, 1 / 30)

        self._args = args
        self._kwargs = kwargs

        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._init_and_start_app)
        # Terminate this thread when main thread exit.
        self.thread.daemon = True
        self.thread.start()

    def _init_and_start_app(self):
        super(SceneViewer, self).__init__(*self._args, **self._kwargs)

    def redraw(self):
        self._redraw = True

    def on_update(self, dt):
        self.on_draw()

    def reset_view(self, flags=None):
        with self.lock:
            super(SceneViewer, self).reset_view(flags=flags)

    def on_draw(self):
        if not self._redraw:
            return

        with self.lock:
            if self._camera_transform is not None:
                camera_transform = self._camera_transform
                self._camera_transform = None
                self.scene.camera.transform = camera_transform
                self.view['ball']._n_pose = camera_transform

            # apply latest angle-vector
            for link in self._links:
                if isinstance(link, robot_model_module.Link):
                    link_list = [link]
                else:
                    link_list = link.link_list
                for l in link_list:
                    transform = l.worldcoords().T()
                    self.scene.graph.update(l.name, matrix=transform)

            super(SceneViewer, self).on_draw()

        self._redraw = False

    def on_mouse_press(self, *args, **kwargs):
        self._redraw = True
        return super(SceneViewer, self).on_mouse_press(*args, **kwargs)

    def on_mouse_drag(self, *args, **kwargs):
        self._redraw = True
        return super(SceneViewer, self).on_mouse_drag(*args, **kwargs)

    def on_mouse_scroll(self, *args, **kwargs):
        self._redraw = True
        return super(SceneViewer, self).on_mouse_scroll(*args, **kwargs)

    def on_key_press(self, *args, **kwargs):
        self._redraw = True
        return super(SceneViewer, self).on_key_press(*args, **kwargs)

    def on_resize(self, *args, **kwargs):
        self._redraw = True
        return super(SceneViewer, self).on_resize(*args, **kwargs)

    def on_close(self):
        return super(SceneViewer, self).on_close()

    def _gl_update_perspective(self):
        try:
            # for high DPI screens viewport size
            # will be different then the passed size
            width, height = self.get_viewport_size()
        except BaseException:
            # older versions of pyglet may not have this
            pass

        # set the new viewport size
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        # get field of view from camera
        fovY = self.scene.camera.fov[1]
        gl.gluPerspective(
            fovY,
            width / float(height),
            .01,
            self.scene.scale * 5.0
        )
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def add(self, link):
        if isinstance(link, robot_model_module.Link):
            link_list = [link]
        elif isinstance(link, robot_model_module.CascadedLink):
            link_list = link.link_list
        else:
            raise TypeError('link must be Link or CascadedLink')

        with self.lock:
            for l in link_list:
                transform = l.worldcoords().T()
                self.scene.add_geometry(
                    geometry=l.visual_mesh,
                    node_name=l.name,
                    geom_name=l.name,
                    transform=transform,
                )
            self._links.append(link)

    def set_camera(self, *args, **kwargs):
        with self.lock:
            self.scene.set_camera(*args, **kwargs)
            self._camera_transform = self.scene.camera_transform

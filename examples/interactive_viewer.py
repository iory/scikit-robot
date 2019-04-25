import argparse
import threading
import time
import warnings

import numpy as np
import pyglet
from pyglet import gl
import trimesh
import trimesh.viewer

import skrobot


class SceneViewerInThread(trimesh.viewer.SceneViewer):

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

        self.lock = threading.RLock()
        self.thread = threading.Thread(target=self._init_and_start_app)
        self.thread.start()

    def _init_and_start_app(self):
        super().__init__(*self._args, **self._kwargs)

    def redraw(self):
        self._redraw = True

    def on_update(self, dt):
        self.on_draw()

    def reset_view(self, flags):
        with self.lock:
            super().reset_view(flags)

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
                if isinstance(link, skrobot.robot_model.Link):
                    link_list = [link]
                else:
                    link_list = link.link_list
                for l in link_list:
                    transform = l.worldcoords().T()
                    self.scene.graph.update(l.name, matrix=transform)

            super().on_draw()

        self._redraw = False

    def on_mouse_press(self, *args, **kwargs):
        self._redraw = True
        return super().on_mouse_press(*args, **kwargs)

    def on_mouse_drag(self, *args, **kwargs):
        self._redraw = True
        return super().on_mouse_drag(*args, **kwargs)

    def on_mouse_scroll(self, *args, **kwargs):
        self._redraw = True
        return super().on_mouse_scroll(*args, **kwargs)

    def on_key_press(self, *args, **kwargs):
        self._redraw = True
        return super().on_key_press(*args, **kwargs)

    def on_resize(self, *args, **kwargs):
        self._redraw = True
        return super().on_resize(*args, **kwargs)

    def on_close(self):
        self.thread.exit()
        return super().on_close()

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
        if isinstance(link, skrobot.robot_model.Link):
            link_list = [link]
        elif isinstance(link, skrobot.robot_model.CascadedLink):
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
            self._camera_transform = self.scene.camera.transform


class Box(skrobot.robot_model.Link):

    def __init__(self, extents, vertex_colors=None, face_colors=None,
                 *args, **kwargs):
        super(Box, self).__init__(*args, **kwargs)
        self._extents = extents
        self._visual_mesh = trimesh.creation.box(
            extents=extents,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='enter interactive shell'
    )
    args = parser.parse_args()

    robot = skrobot.robot_models.Kuka()

    scene = trimesh.Scene()

    # base plane
    geom = trimesh.creation.box((2, 2, 0.01))
    geom.visual.face_colors = (0.75, 0.75, 0.75)
    scene.add_geometry(geom)

    viewer = SceneViewerInThread(scene, resolution=(640, 480))

    viewer.add(robot)

    viewer.set_camera(angles=[np.deg2rad(45), 0, 0], distance=4)

    box = Box(extents=(0.05, 0.05, 0.05), face_colors=(1., 0, 0))
    box.translate((0.5, 0, 0.3))
    viewer.add(box)

    if args.interactive:
        print('''\
>>> # Usage

>>> robot.reset_manip_pose()
>>> viewer.redraw()
>>> robot.init_pose()
>>> robot.inverse_kinematics(box, rotation_axis='y')
''')

        import IPython

        IPython.embed()
    else:
        print('==> Waiting 3 seconds')
        time.sleep(3)

        print('==> Moving to reset_manip_pose')
        robot.reset_manip_pose()
        print(robot.angle_vector())
        time.sleep(1)
        viewer.redraw()

        print('==> Waiting 3 seconds')
        time.sleep(3)

        print('==> Moving to init_pose')
        robot.init_pose()
        print(robot.angle_vector())
        time.sleep(1)
        viewer.redraw()

        print('==> Waiting 3 seconds')
        time.sleep(3)

        print('==> IK to box')
        robot.reset_manip_pose()
        robot.inverse_kinematics(box, rotation_axis='y')
        print(robot.angle_vector())
        time.sleep(1)
        viewer.redraw()

        print('==> Press [q] to close window')


if __name__ == '__main__':
    main()

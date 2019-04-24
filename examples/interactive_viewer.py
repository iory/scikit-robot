import argparse
import threading
import time

import numpy as np
import pyglet
import trimesh
import trimesh.viewer

import skrobot


class SceneViewerInThread(trimesh.viewer.SceneViewer):

    def __init__(self, *args, **kwargs):
        if len(args) > 6:
            args = list(args)
            args[5] = False  # start_loop
            args = tuple(args)
        else:
            kwargs['start_loop'] = False

        super().__init__(*args, **kwargs)

        self._redraw = True
        pyglet.clock.schedule_interval(self.on_update, 1 / 30)

        self.lock = threading.RLock()
        self.thread = threading.Thread(target=pyglet.app.run)
        self.thread.start()

    def redraw(self):
        self._redraw = True

    def on_update(self, dt):
        self.on_draw()

    def on_draw(self):
        self.view['ball']._n_pose = self.scene.camera.transform

        if not self._redraw:
            return

        with self.lock:
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

    def _gl_update_perspective(self):
        from pyglet import gl
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

    def scene_add_geometry(self, *args, **kwargs):
        with self.lock:
            self.scene.add_geometry(*args, **kwargs)

            self._update_vertex_list()
            self._gl_update_perspective()

    def scene_set_camera(self, *args, **kwargs):
        with self.lock:
            self.scene.set_camera(*args, **kwargs)


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

    # links
    for link in robot.link_list:
        transform = link.worldcoords().T()
        scene.add_geometry(
            geometry=link.visual_mesh,
            node_name=link.name,
            geom_name=link.name,
            transform=transform,
        )

    scene.set_camera(angles=[np.deg2rad(45), 0, 0])

    viewer = SceneViewerInThread(scene, resolution=(640, 480))

    if args.interactive:
        print('''\
>>> # Usage

>>> robot.reset_manip_pose()
>>> with viewer.lock:
>>>     for link in robot.link_list:
>>>         transform = link.worldcoords().T()
>>>         scene.graph.update(link.name, matrix=transform)
>>> viewer.redraw()\n''')

        import IPython

        IPython.embed()
    else:
        print('==> Waiting 3 seconds')
        time.sleep(3)

        print('==> Moving to reset_manip_pose')
        robot.reset_manip_pose()
        with viewer.lock:
            for link in robot.link_list:
                transform = link.worldcoords().T()
                scene.graph.update(link.name, matrix=transform)

        print('==> Waiting 3 seconds')
        time.sleep(3)

        print('==> Redrawing')
        viewer.redraw()

        print('==> Press [q] to close window')


if __name__ == '__main__':
    main()

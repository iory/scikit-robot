from __future__ import division

import collections

import pyglet
import pyrender
from pyrender.trackball import Trackball

from skrobot.coordinates import Coordinates

from .. import model as model_module


class PyrenderViewer(pyrender.Viewer):

    def __init__(self, resolution=None, update_interval=1.0):
        if resolution is None:
            resolution = (640, 480)

        self._visual_mesh_map = collections.OrderedDict()

        self._redraw = True
        pyglet.clock.schedule_interval(self.on_update, update_interval)

        self._kwargs = dict(
            scene=pyrender.Scene(),
            viewport_size=resolution,
            run_in_thread=True,
            use_raymond_lighting=True,
        )
        super(PyrenderViewer, self).__init__(**self._kwargs)

    def show(self):
        pass

    def redraw(self):
        self._redraw = True

    def on_update(self, dt):
        self.on_draw()

    def on_draw(self):
        if not self._redraw:
            with self._render_lock:
                super(PyrenderViewer, self).on_draw()
            return

        with self._render_lock:
            # apply latest angle-vector
            for _, (node, link) in self._visual_mesh_map.items():
                link.update(force=True)
                transform = link.worldcoords().T()
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
            mesh = link.visual_mesh

            if not (isinstance(mesh, list) or isinstance(mesh, tuple)):
                mesh = [mesh]
            for m in mesh:
                mesh_id = str(id(m))
                if mesh_id in self._visual_mesh_map:
                    continue
                pyrender_mesh = pyrender.Mesh.from_trimesh(
                    m, smooth=False)
                node = self.scene.add(pyrender_mesh, pose=transform)
                self._visual_mesh_map[mesh_id] = (node, link)

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
                mesh = link.visual_mesh
                if not (isinstance(mesh, list) or isinstance(mesh, tuple)):
                    mesh = [mesh]
                for m in mesh:
                    mesh_id = str(id(m))
                    if mesh_id not in self._visual_mesh_map:
                        continue
                    self.scene.remove_node(self._visual_mesh_map[mesh_id][0])
                    self._visual_mesh_map.pop(mesh_id)
                all_links = all_links[1:]
                all_links.extend(link.child_links)
        self._redraw = True

    def set_camera(self, angles=None, distance=None, coords_or_transform=None):
        if angles is None and coords_or_transform is None:
            return
        if angles is not None:
            return
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

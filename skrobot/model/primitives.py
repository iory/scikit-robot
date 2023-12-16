import uuid

import numpy as np
import trimesh
from trimesh import PointCloud

from skrobot.coordinates.base import CascadedCoords
from skrobot.coordinates.base import Coordinates
from skrobot.model import Link
from skrobot.sdf import BoxSDF
from skrobot.sdf import CylinderSDF
from skrobot.sdf import SphereSDF
from skrobot.sdf import trimesh2sdf


class Axis(Link):

    def __init__(self,
                 axis_radius=0.01,
                 axis_length=0.1,
                 pos=(0, 0, 0), rot=np.eye(3), name=None):
        if name is None:
            name = 'axis_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        super(Axis, self).__init__(pos=pos, rot=rot, name=name)
        self._visual_mesh = trimesh.creation.axis(
            origin_size=0.00000001,
            axis_radius=axis_radius,
            axis_length=axis_length,
        )

    @classmethod
    def from_coords(cls, coords, **kwargs):
        assert isinstance(coords, Coordinates)
        link = cls(**kwargs)
        link.parent = coords
        return link

    @classmethod
    def from_cascoords(cls, cascoords, **kwargs):
        assert isinstance(cascoords, CascadedCoords)
        link = cls(**kwargs)
        link.parent = cascoords.worldcoords()
        for cc in cascoords.descendants:
            child_link = cls.from_cascoords(cc, **kwargs)
            link.add_child_link(child_link)
        return link


class Box(Link):

    def __init__(self, extents, vertex_colors=None, face_colors=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None, with_sdf=False):
        if name is None:
            name = 'box_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        mesh = trimesh.creation.box(
            extents=extents,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )
        super(Box, self).__init__(pos=pos, rot=rot, name=name,
                                  collision_mesh=mesh,
                                  visual_mesh=mesh)
        self._extents = extents
        if with_sdf:
            sdf = BoxSDF(extents)
            self.assoc(sdf, relative_coords="local")
            self.sdf = sdf


class CameraMarker(Link):

    def __init__(self, focal=None, fov=(70, 40), z_near=0.01, z_far=1000.0,
                 marker_height=0.4, pos=(0, 0, 0), rot=np.eye(3),
                 without_axis=False, name=None):
        if name is None:
            name = 'camera_marker_{}'.format(
                str(uuid.uuid1()).replace('-', '_'))

        super(CameraMarker, self).__init__(
            pos=pos, rot=rot, name=name)
        camera = trimesh.scene.Camera(name=name,
                                      focal=focal,
                                      fov=fov,
                                      z_near=z_near,
                                      z_far=z_far)

        origin_size = None
        if without_axis is True:
            origin_size = 0.0
        self._visual_mesh = trimesh.creation.camera_marker(
            camera,
            marker_height=marker_height,
            origin_size=origin_size)


class Cone(Link):

    def __init__(self, radius, height,
                 sections=32,
                 vertex_colors=None, face_colors=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None):
        if name is None:
            name = 'cone_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        mesh = trimesh.creation.cone(
            radius=radius,
            height=height,
            sections=sections,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )
        super(Cone, self).__init__(pos=pos, rot=rot, name=name,
                                   collision_mesh=mesh,
                                   visual_mesh=mesh)


class Cylinder(Link):

    def __init__(self, radius, height,
                 sections=32,
                 vertex_colors=None, face_colors=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None, with_sdf=False):
        if name is None:
            name = 'cylinder_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        mesh = trimesh.creation.cylinder(
            radius=radius,
            height=height,
            sections=sections,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )
        super(Cylinder, self).__init__(pos=pos, rot=rot, name=name,
                                       collision_mesh=mesh,
                                       visual_mesh=mesh)
        if with_sdf:
            sdf = CylinderSDF(height, radius)
            self.assoc(sdf, relative_coords="local")
            self.sdf = sdf


class Sphere(Link):

    def __init__(self, radius, subdivisions=3, color=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None, with_sdf=False):
        if name is None:
            name = 'sphere_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        mesh = trimesh.creation.icosphere(
            radius=radius,
            subdivisions=subdivisions,
            color=color,
        )
        super(Sphere, self).__init__(pos=pos, rot=rot, name=name,
                                     collision_mesh=mesh,
                                     visual_mesh=mesh)

        if with_sdf:
            sdf = SphereSDF(radius)
            self.assoc(sdf, relative_coords="local")
            self.sdf = sdf


class Annulus(Link):

    def __init__(self, r_min, r_max, height,
                 vertex_colors=None, face_colors=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None):
        if name is None:
            name = 'annulus_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        mesh = trimesh.creation.annulus(
            r_min=r_min,
            r_max=r_max,
            height=height,
            vertex_colors=vertex_colors,
            face_colors=face_colors
        )
        super(Annulus, self).__init__(pos=pos, rot=rot, name=name,
                                      collision_mesh=mesh,
                                      visual_mesh=mesh)


class LineString(Link):

    def __init__(self,
                 points,
                 color=None,
                 pos=(0, 0, 0),
                 rot=np.eye(3),
                 name=None):

        if not isinstance(points, np.ndarray):
            raise TypeError("points must be np.ndarray")

        assert points.ndim == 2
        assert points.shape[0] > 1, "points must be more than 1"
        assert points.shape[1] == 3, "each point must be 3 dim"

        if color is not None:
            assert len(color) in (3, 4), "color must be RGB or RGBA"
            colors = [color]
        else:
            colors = None

        if name is None:
            name = 'linestring_{}'.format(str(uuid.uuid1()).replace('-', '_'))
        mesh = trimesh.load_path(points, colors=colors)
        super(LineString, self).__init__(
            pos=pos,
            rot=rot,
            name=name,
            collision_mesh=None,
            visual_mesh=mesh)


class MeshLink(Link):

    def __init__(self,
                 visual_mesh=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None, with_sdf=False,
                 dim_grid=100, padding_grid=5):
        if name is None:
            name = 'meshlink_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        super(MeshLink, self).__init__(pos=pos, rot=rot, name=name)
        self.visual_mesh = visual_mesh
        if self.visual_mesh is not None:
            if isinstance(self.visual_mesh, list):
                self._collision_mesh = \
                    self.visual_mesh[0] + self.visual_mesh[1:]
            else:
                self._collision_mesh = self.visual_mesh
            self._collision_mesh.metadata['origin'] = np.eye(4)

        if with_sdf:
            sdf = trimesh2sdf(self._collision_mesh, dim_grid=dim_grid,
                              padding_grid=padding_grid)
            self.assoc(sdf, relative_coords="local")
            self.sdf = sdf


class PointCloudLink(Link):

    def __init__(self,
                 point_cloud_like=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None):

        accep_types = (type(None), np.ndarray, PointCloud)
        if not isinstance(point_cloud_like, accep_types):
            message = "point cloud must be either of {}".format(accep_types)
            raise TypeError(message)

        if isinstance(point_cloud_like, np.ndarray):
            assert point_cloud_like.ndim == 2
            assert point_cloud_like.shape[1] == 3
            pcloud_mesh = PointCloud(point_cloud_like)
        else:
            pcloud_mesh = point_cloud_like

        if name is None:
            name = 'pointcloudlink_{}'.format(
                str(uuid.uuid1()).replace('-', '_'))

        super(PointCloudLink, self).__init__(pos=pos, rot=rot, name=name)
        self.visual_mesh = pcloud_mesh

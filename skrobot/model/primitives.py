import os
import uuid

import numpy as np
import trimesh

from skrobot.coordinates.base import CascadedCoords
from skrobot.coordinates.base import Coordinates
from skrobot.model import Link
from skrobot.sdf import BoxSDF
from skrobot.sdf import CylinderSDF
from skrobot.sdf import GridSDF
from skrobot.sdf import SphereSDF


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
            child_link = cls.from_cascoords(cc)
            link.add_child_link(child_link)
        return link


class Box(Link):

    def __init__(self, extents, vertex_colors=None, face_colors=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None, with_sdf=False):
        if name is None:
            name = 'box_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        super(Box, self).__init__(pos=pos, rot=rot, name=name)
        self._extents = extents
        self._visual_mesh = trimesh.creation.box(
            extents=extents,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )

        if with_sdf:
            sdf = BoxSDF(np.zeros(3), extents)
            self.assoc(sdf.coords)
            self.sdf = sdf
        else:
            self.sdf = None


class CameraMarker(Link):

    def __init__(self, focal=None, fov=(70, 40), z_near=0.01, z_far=1000.0,
                 marker_height=0.4, pos=(0, 0, 0), rot=np.eye(3), name=None):
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

        self._visual_mesh = trimesh.creation.camera_marker(
            camera,
            marker_height=marker_height)


class Cone(Link):

    def __init__(self, radius, height,
                 sections=32,
                 vertex_colors=None, face_colors=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None):
        if name is None:
            name = 'cone_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        super(Cone, self).__init__(pos=pos, rot=rot, name=name)
        self._visual_mesh = trimesh.creation.cone(
            radius=radius,
            height=height,
            sections=sections,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )


class Cylinder(Link):

    def __init__(self, radius, height,
                 sections=32,
                 vertex_colors=None, face_colors=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None, with_sdf=False):
        if name is None:
            name = 'cylinder_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        super(Cylinder, self).__init__(pos=pos, rot=rot, name=name)
        self._visual_mesh = trimesh.creation.cylinder(
            radius=radius,
            height=height,
            sections=sections,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )

        if with_sdf:
            sdf = CylinderSDF(np.zeros(3), height, radius)
            self.assoc(sdf.coords)
            self.sdf = sdf
        else:
            self.sdf = None


class Sphere(Link):

    def __init__(self, radius, subdivisions=3, color=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None, with_sdf=False):
        if name is None:
            name = 'sphere_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        super(Sphere, self).__init__(pos=pos, rot=rot, name=name)
        self._visual_mesh = trimesh.creation.icosphere(
            radius=radius,
            subdivisions=subdivisions,
            color=color,
        )

        if with_sdf:
            sdf = SphereSDF(np.zeros(3), radius)
            self.assoc(sdf.coords)
            self.sdf = sdf
        else:
            self.sdf = None


class Annulus(Link):

    def __init__(self, r_min, r_max, height,
                 vertex_colors=None, face_colors=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None):
        if name is None:
            name = 'annulus_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        super(Annulus, self).__init__(pos=pos, rot=rot, name=name)
        self._visual_mesh = trimesh.creation.annulus(
            r_min=r_min,
            r_max=r_max,
            height=height,
            vertex_colors=vertex_colors,
            face_colors=face_colors
        )


class MeshLink(Link):

    def __init__(self,
                 visual_mesh=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None, with_sdf=False,
                 dim_grid=100, padding_grid=5):
        if name is None:
            name = 'meshlink_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        super(MeshLink, self).__init__(pos=pos, rot=rot, name=name)
        self.visual_mesh = visual_mesh

        if with_sdf:
            assert os.path.isfile(visual_mesh),\
                "with_sdf is valid only with a mesh file"
            sdf = GridSDF.from_objfile(
                visual_mesh,
                dim_grid=dim_grid,
                padding_grid=padding_grid)
            self.assoc(sdf.coords)
            self.sdf = sdf
        else:
            self.sdf = None


class PointCloudLink(Link):

    def __init__(self,
                 visual_mesh=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None):
        if name is None:
            name = 'pointcloudlink_{}'.format(
                str(uuid.uuid1()).replace('-', '_'))

        super(PointCloudLink, self).__init__(pos=pos, rot=rot, name=name)
        self.visual_mesh = visual_mesh

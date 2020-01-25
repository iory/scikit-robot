import uuid

import numpy as np
import trimesh

from skrobot import model as model_module


class Axis(model_module.Link):

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


class Box(model_module.Link):

    def __init__(self, extents, vertex_colors=None, face_colors=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None):
        if name is None:
            name = 'box_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        super(Box, self).__init__(pos=pos, rot=rot, name=name)
        self._extents = extents
        self._visual_mesh = trimesh.creation.box(
            extents=extents,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )


class Cylinder(model_module.Link):

    def __init__(self, radius, height,
                 vertex_colors=None, face_colors=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None):
        if name is None:
            name = 'cylinder_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        super(Cylinder, self).__init__(pos=pos, rot=rot, name=name)
        self._visual_mesh = trimesh.creation.cylinder(
            radius=radius,
            height=height,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
            )


class Sphere(model_module.Link):

    def __init__(self, radius, subdivisions=3, color=None,
                 pos=(0, 0, 0), rot=np.eye(3), name=None):
        if name is None:
            name = 'sphere_{}'.format(str(uuid.uuid1()).replace('-', '_'))

        super(Sphere, self).__init__(pos=pos, rot=rot, name=name)
        self._visual_mesh = trimesh.creation.icosphere(
            radius=radius,
            subdivisions=subdivisions,
            color=color,
            )

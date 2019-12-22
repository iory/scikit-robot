import uuid

import numpy as np
import trimesh

from skrobot import model as model_module


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

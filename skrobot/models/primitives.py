import trimesh

from skrobot import robot_model as robot_model_module


class Box(robot_model_module.Link):

    def __init__(self, extents, vertex_colors=None, face_colors=None,
                 *args, **kwargs):
        super(Box, self).__init__(*args, **kwargs)
        self._extents = extents
        self._visual_mesh = trimesh.creation.box(
            extents=extents,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )

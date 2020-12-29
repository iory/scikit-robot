import warnings

import trimesh

from skrobot.utils import urdf


def _meshes_from_urdf_visuals(visuals):
    meshes = []
    for visual in visuals:
        meshes.extend(_meshes_from_urdf_visual(visual))
    return meshes


def _meshes_from_urdf_visual(visual):
    if not isinstance(visual, urdf.Visual):
        raise TypeError('visual must be urdf.Visual, but got: {}'
                        .format(type(visual)))

    meshes = []
    for mesh in visual.geometry.meshes:
        mesh = mesh.copy()

        # rescale
        if visual.geometry.mesh is not None:
            if visual.geometry.mesh.scale is not None:
                mesh.vertices = mesh.vertices * visual.geometry.mesh.scale

        # TextureVisuals is usually slow to render
        if not isinstance(mesh.visual, trimesh.visual.ColorVisuals):
            mesh.visual = mesh.visual.to_color()
            if mesh.visual.vertex_colors.ndim == 1:
                mesh.visual.vertex_colors = \
                    mesh.visual.vertex_colors[None].repeat(
                        mesh.vertices.shape[0], axis=0
                    )

        # If color or texture is not specified in mesh file,
        # use information specified in URDF.
        if (
            (mesh.visual.face_colors
             == trimesh.visual.DEFAULT_COLOR).all()
            and visual.material
        ):
            if visual.material.texture is not None:
                warnings.warn(
                    'texture specified in URDF is not supported'
                )
            elif visual.material.color is not None:
                mesh.visual.face_colors = visual.material.color

        mesh.apply_transform(visual.origin)
        meshes.append(mesh)
    return meshes

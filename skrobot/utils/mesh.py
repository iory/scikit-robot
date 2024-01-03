import numpy as np


def split_mesh_by_face_color(mesh):
    """Split a trimesh mesh based on face colors.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to be split.

    Returns
    -------
    List[trimesh.Trimesh]
        List of meshes, each corresponding to a unique face color.

    Notes
    -----
    This function uses the face colors of the provided mesh to
    generate a list of submeshes, where each submesh contains
    faces of the same color.
    """
    face_colors = mesh.visual.face_colors
    unique_colors = np.unique(face_colors, axis=0)
    submeshes = []
    for color in unique_colors:
        mask = np.all(face_colors == color, axis=1)
        submesh = mesh.submesh([mask])[0]
        submeshes.append(submesh)

    return submeshes

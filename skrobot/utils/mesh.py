import numpy as np
import trimesh


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
    if mesh.visual.kind == 'texture':
        return [mesh]
    face_colors = mesh.visual.face_colors
    unique_colors = np.unique(face_colors, axis=0)
    submeshes = []
    for color in unique_colors:
        mask = np.all(face_colors == color, axis=1)
        submesh = mesh.submesh([mask])[0]
        submeshes.append(submesh)

    return submeshes


def to_open3d(mesh):
    import open3d
    o3d_mesh = open3d.geometry.TriangleMesh()
    o3d_mesh.vertices = open3d.utility.Vector3dVector(np.array(mesh.vertices))
    o3d_mesh.triangles = open3d.utility.Vector3iVector(np.array(mesh.faces))

    # Convert vertex colors from RGBA to RGB and normalize the color values
    # Use NumPy slicing and broadcasting for efficient conversion
    vertex_colors = np.array(mesh.visual.vertex_colors)[:, :3] / 255.0
    o3d_mesh.vertex_colors = open3d.utility.Vector3dVector(vertex_colors)
    return o3d_mesh


def simplify_vertex_clustering(
        meshes, simplify_vertex_clustering_voxel_size=0.001):
    if not isinstance(meshes, list):
        meshes = [meshes]
    simplify_meshes = []
    for mesh in meshes:
        if mesh.visual.kind == 'texture':
            simplify_meshes.append(mesh)
            continue
        simple = to_open3d(mesh).simplify_vertex_clustering(
            simplify_vertex_clustering_voxel_size)
        mesh = trimesh.Trimesh(
            vertices=simple.vertices,
            faces=simple.triangles,
            vertex_colors=simple.vertex_colors)
        simplify_meshes.append(mesh)
    return simplify_meshes

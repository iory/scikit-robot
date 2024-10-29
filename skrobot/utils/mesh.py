import numpy as np
import trimesh
from trimesh.proximity import nearby_faces


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


def auto_simplify_quadric_decimation(meshes, area_ratio_threshold=0.98):
    if not isinstance(meshes, list):
        meshes = [meshes]

    mesh_simplified_list = []
    for mesh in meshes:
        n_face = len(mesh.faces)
        for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            n_face_reduced = int(n_face * f)
            mesh_simplified = mesh.simplify_quadric_decimation(
                face_count=n_face_reduced)
            ratio = mesh_simplified.area / mesh.area
            if ratio > area_ratio_threshold:
                break
        if len(mesh_simplified.faces) == 0 or ratio <= area_ratio_threshold:
            mesh_simplified_list.append(mesh)
            continue

        simplified_face_vertices = mesh_simplified.vertices[
            mesh_simplified.faces].reshape(-1, 3)
        org_indices_list = nearby_faces(mesh, simplified_face_vertices)
        org_indices_list = [np.concatenate(org_indices_list[i:i + 3])
                            for i in range(0, len(org_indices_list), 3)]
        vertex_colors = None
        face_colors = []
        for i in range(len(org_indices_list)):
            org_normals = mesh.face_normals[org_indices_list[i]]
            simplified_normal = mesh_simplified.face_normals[i]
            # Calculate cosine distance to get similar face.
            indices = np.argsort(1.0 - np.dot(org_normals, simplified_normal))
            face_colors.append(mesh.visual.face_colors[
                org_indices_list[i][indices[0]]])

        visual = mesh.visual
        if visual.kind == 'texture':
            visual = visual.to_color()
            visual.mesh = mesh
        mesh_simplified._visual = trimesh.visual.ColorVisuals(
            mesh=mesh_simplified,
            vertex_colors=vertex_colors,
            face_colors=face_colors,
        )
        mesh_simplified_list.append(mesh_simplified)
    return mesh_simplified_list

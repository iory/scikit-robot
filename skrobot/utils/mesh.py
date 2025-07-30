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


def create_vertex_colors_from_texture(mesh, verbose=False):
    """Create vertex colors from texture information.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh with texture information.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    numpy.ndarray
        Vertex colors as (n_vertices, 3) array.
    """
    vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.float64)

    if verbose:
        print("Creating vertex colors, please wait...")

    # Check if mesh has texture information
    if not (hasattr(mesh, 'visual') and mesh.visual.kind == 'texture'):
        if verbose:
            print("No texture information found, using default colors")
        return np.ones((len(mesh.vertices), 3)) * 0.5

    # Get texture image and UV coordinates
    texture_image = None
    uv_coords = None

    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        if hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
            texture_image = mesh.visual.material.image
        elif (hasattr(mesh.visual.material, 'baseColorTexture') and
              mesh.visual.material.baseColorTexture is not None and
              hasattr(mesh.visual.material.baseColorTexture, 'image')):
            texture_image = mesh.visual.material.baseColorTexture.image

    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uv_coords = mesh.visual.uv

    if texture_image is None:
        if verbose:
            print("No texture image found, using material colors")
        # Try to use material main color if available
        if (hasattr(mesh.visual, 'material') and
            mesh.visual.material is not None and
            hasattr(mesh.visual.material, 'main_color')):
            main_color = np.array(mesh.visual.material.main_color[:3]) / 255.0
            return np.tile(main_color, (len(mesh.vertices), 1))
        else:
            return np.ones((len(mesh.vertices), 3)) * 0.5

    if uv_coords is None:
        if verbose:
            print("No UV coordinates found, using default colors")
        return np.ones((len(mesh.vertices), 3)) * 0.5

    # Convert texture image to numpy array
    texture_np = np.array(texture_image)
    if len(texture_np.shape) == 2:
        # Grayscale image, convert to RGB
        texture_np = np.stack([texture_np] * 3, axis=-1)
    height, width = texture_np.shape[:2]

    # UV coordinates are per triangle vertex, so we have 3 * num_faces UV coordinates
    num_triangles = len(mesh.faces)
    progress_step = max(1, num_triangles // 100)

    for triangle_index in range(num_triangles):
        # Process each vertex of the triangle
        for local_vertex in range(3):
            uv_index = triangle_index * 3 + local_vertex
            if uv_index >= len(uv_coords):
                continue

            u, v = uv_coords[uv_index]
            # Clamp UV coordinates to [0, 1] range
            u = max(0, min(1, u))
            v = max(0, min(1, v))

            # Convert UV to pixel coordinates
            x = int(u * (width - 1))
            y = int(v * (height - 1))

            # Get the global vertex index
            global_vertex_index = mesh.faces[triangle_index][local_vertex]

            # Sample the texture at this UV coordinate
            vertex_colors[global_vertex_index] = texture_np[y, x][:3] / 255.0

        if verbose and (triangle_index % progress_step) == 0:
            print('#', end='', flush=True)

    if verbose:
        print()

    return vertex_colors


def create_texture_from_vertex_colors(mesh):
    """Create texture from vertex colors.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh with vertex colors.

    Returns
    -------
    trimesh.Trimesh
        Mesh with texture created from vertex colors.
    """
    # Get vertex colors
    if hasattr(mesh.visual, 'vertex_colors'):
        vertex_colors = np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0
    else:
        # Use default colors if no vertex colors
        vertex_colors = np.ones((len(mesh.vertices), 3)) * 0.5

    # Calculate texture dimension to hold all triangle vertex colors
    texture_dimension = int(np.ceil(np.sqrt(len(mesh.faces) * 3)))

    # Create texture array
    texture = np.full((texture_dimension, texture_dimension, 3), 0, dtype=np.uint8)
    triangle_uvs = np.full((len(mesh.faces) * 3, 2), 0, dtype=np.float32)

    # Fill texture with vertex colors
    for triangle_index, triangle in enumerate(mesh.faces):
        for vertex_sub_index in range(3):
            UV_index = 3 * triangle_index + vertex_sub_index
            U = UV_index % texture_dimension
            V = UV_index // texture_dimension

            # Store vertex color in texture
            color = vertex_colors[triangle[vertex_sub_index]]
            texture[V, U] = (color * 255).astype(np.uint8)

            # Set UV coordinates
            triangle_uvs[UV_index] = [
                (U + 0.5) / texture_dimension,
                (V + 0.5) / texture_dimension
            ]

    # Create new mesh with same geometry
    result_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    # Create PIL Image from texture
    from PIL import Image
    texture_img = Image.fromarray(texture, mode='RGB')

    # Create material with texture
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture_img
    )

    # Create texture visual
    result_mesh.visual = trimesh.visual.TextureVisuals(
        uv=triangle_uvs,
        material=material
    )

    return result_mesh


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
            # Check if mesh has actual texture images (not just material colors)
            has_actual_texture = False
            if (hasattr(mesh.visual, 'material') and
                mesh.visual.material is not None):
                if (hasattr(mesh.visual.material, 'baseColorTexture') and
                    mesh.visual.material.baseColorTexture is not None):
                    has_actual_texture = True
                elif (hasattr(mesh.visual.material, 'image') and
                      mesh.visual.material.image is not None):
                    has_actual_texture = True

            # For meshes with only material colors, use Open3D directly
            if not has_actual_texture:
                import open3d as o3d

                # Get material color directly
                material_color = mesh.visual.material.main_color[:3]  # RGB only

                # Convert to Open3D directly
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

                # Set vertex colors directly in Open3D (normalized)
                vertex_colors_normalized = np.tile(material_color / 255.0, (len(mesh.vertices), 1))
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_normalized)

                # Use Open3D clustering
                simple = o3d_mesh.simplify_vertex_clustering(simplify_vertex_clustering_voxel_size)

                # Convert back to trimesh with vertex colors in constructor
                if simple.has_vertex_colors():
                    vertex_colors_back = np.asarray(simple.vertex_colors)
                    vertex_colors_rgba = np.column_stack([
                        (vertex_colors_back * 255).astype(np.uint8),
                        np.full(len(vertex_colors_back), 255, dtype=np.uint8)
                    ])

                    # Use constructor with vertex_colors to preserve exact colors
                    simplified_mesh = trimesh.Trimesh(
                        vertices=np.asarray(simple.vertices),
                        faces=np.asarray(simple.triangles),
                        vertex_colors=vertex_colors_rgba
                    )
                else:
                    simplified_mesh = trimesh.Trimesh(
                        vertices=np.asarray(simple.vertices),
                        faces=np.asarray(simple.triangles)
                    )

                simplify_meshes.append(simplified_mesh)
                continue

            # For actual texture meshes, use texture preservation
            import open3d as o3d

            # Convert texture to vertex colors
            vertex_colors = create_vertex_colors_from_texture(mesh, verbose=False)

            # Convert to open3d
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            # Perform vertex clustering
            simplified_o3d_mesh = o3d_mesh.simplify_vertex_clustering(
                simplify_vertex_clustering_voxel_size)

            # Convert back to trimesh
            vertices = np.asarray(simplified_o3d_mesh.vertices)
            faces = np.asarray(simplified_o3d_mesh.triangles)
            simplified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            # Set vertex colors
            if simplified_o3d_mesh.has_vertex_colors():
                vertex_colors_simplified = np.asarray(simplified_o3d_mesh.vertex_colors)
                vertex_colors_rgba = np.column_stack([
                    (vertex_colors_simplified * 255).astype(np.uint8),
                    np.full(len(vertex_colors_simplified), 255, dtype=np.uint8)
                ])
                simplified_mesh.visual.vertex_colors = vertex_colors_rgba

            # Convert vertex colors back to texture
            simplified_mesh = create_texture_from_vertex_colors(simplified_mesh)
            simplify_meshes.append(simplified_mesh)
            continue

        simple = to_open3d(mesh).simplify_vertex_clustering(
            simplify_vertex_clustering_voxel_size)
        mesh = trimesh.Trimesh(
            vertices=simple.vertices,
            faces=simple.triangles,
            vertex_colors=simple.vertex_colors)
        simplify_meshes.append(mesh)
    return simplify_meshes


def auto_simplify_quadric_decimation_with_texture_preservation(
        meshes, target_number_of_triangles=50000, verbose=False):
    """Simplify meshes using quadric decimation while preserving texture colors.

    This function converts texture information to vertex colors, performs
    decimation, and then converts the vertex colors back to texture format.
    Only creates textures when the original mesh has actual texture images.

    Parameters
    ----------
    meshes : list or trimesh.Trimesh
        Meshes to be simplified.
    target_number_of_triangles : int, optional
        Target number of triangles after decimation. Default is 50000.
    verbose : bool, optional
        Whether to print progress information. Default is False.

    Returns
    -------
    list
        List of simplified meshes with preserved color information.
    """
    import open3d as o3d

    if not isinstance(meshes, list):
        meshes = [meshes]

    mesh_simplified_list = []
    for mesh in meshes:
        if verbose:
            print(f"Processing mesh with {len(mesh.faces)} faces...")

        # Check if mesh has actual texture images (not just material colors)
        has_actual_texture = False
        if mesh.visual.kind == 'texture':
            if (hasattr(mesh.visual, 'material') and
                mesh.visual.material is not None):
                if (hasattr(mesh.visual.material, 'baseColorTexture') and
                    mesh.visual.material.baseColorTexture is not None):
                    has_actual_texture = True
                elif (hasattr(mesh.visual.material, 'image') and
                      mesh.visual.material.image is not None):
                    has_actual_texture = True

        # For meshes with only material colors (no texture images),
        # use Open3D directly to avoid trimesh ColorVisuals issues
        if mesh.visual.kind == 'texture' and not has_actual_texture:
            if verbose:
                print("Processing material-only mesh with Open3D...")

            # Get material color directly
            material_color = mesh.visual.material.main_color[:3]  # RGB only

            # Convert to Open3D directly
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

            # Set vertex colors directly in Open3D (normalized)
            vertex_colors_normalized = np.tile(material_color / 255.0, (len(mesh.vertices), 1))
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_normalized)

            # Perform Open3D decimation
            simplified_o3d_mesh = o3d_mesh.simplify_quadric_decimation(
                target_number_of_triangles=min(target_number_of_triangles, len(mesh.faces)))

            # Convert back to trimesh with vertex colors in constructor
            vertices = np.asarray(simplified_o3d_mesh.vertices)
            faces = np.asarray(simplified_o3d_mesh.triangles)

            if simplified_o3d_mesh.has_vertex_colors():
                vertex_colors_back = np.asarray(simplified_o3d_mesh.vertex_colors)
                vertex_colors_rgba = np.column_stack([
                    (vertex_colors_back * 255).astype(np.uint8),
                    np.full(len(vertex_colors_back), 255, dtype=np.uint8)
                ])

                # Use constructor with vertex_colors to preserve exact colors
                simplified_mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_colors=vertex_colors_rgba
                )
            else:
                simplified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            mesh_simplified_list.append(simplified_mesh)

            if verbose:
                print(f"Simplified mesh has {len(simplified_mesh.faces)} faces")
            continue

        # For meshes with actual textures, use texture preservation
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

        # Handle texture meshes by converting to vertex colors
        if mesh.visual.kind == 'texture':
            if verbose:
                print("Converting texture to vertex colors...")
            vertex_colors = create_vertex_colors_from_texture(mesh, verbose=verbose)
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        else:
            # Use existing vertex/face colors
            if hasattr(mesh.visual, 'vertex_colors'):
                vertex_colors = np.array(mesh.visual.vertex_colors)[:, :3] / 255.0
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        # Perform decimation
        if verbose:
            print(f"Performing decimation to {target_number_of_triangles} triangles...")

        simplified_o3d_mesh = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_number_of_triangles)

        # Convert back to trimesh
        vertices = np.asarray(simplified_o3d_mesh.vertices)
        faces = np.asarray(simplified_o3d_mesh.triangles)
        simplified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Set vertex colors on the simplified mesh
        if simplified_o3d_mesh.has_vertex_colors():
            vertex_colors = np.asarray(simplified_o3d_mesh.vertex_colors)
            # Convert to RGBA format (add alpha channel)
            vertex_colors_rgba = np.column_stack([
                (vertex_colors * 255).astype(np.uint8),
                np.full(len(vertex_colors), 255, dtype=np.uint8)  # full alpha
            ])
            simplified_mesh.visual.vertex_colors = vertex_colors_rgba

        # Convert vertex colors back to texture only if original had actual textures
        if mesh.visual.kind == 'texture' and has_actual_texture:
            if verbose:
                print("Converting vertex colors back to texture...")
            simplified_mesh = create_texture_from_vertex_colors(simplified_mesh)

        mesh_simplified_list.append(simplified_mesh)

        if verbose:
            print(f"Simplified mesh has {len(simplified_mesh.faces)} faces")

    return mesh_simplified_list


def auto_simplify_quadric_decimation(meshes, area_ratio_threshold=0.98):
    if not isinstance(meshes, list):
        meshes = [meshes]

    mesh_simplified_list = []
    for mesh in meshes:
        # For texture meshes, use the texture-preserving decimation method
        if mesh.visual.kind == 'texture':
            simplified_meshes = auto_simplify_quadric_decimation_with_texture_preservation(
                [mesh], target_number_of_triangles=int(len(mesh.faces) * area_ratio_threshold))
            mesh_simplified_list.extend(simplified_meshes)
            continue

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

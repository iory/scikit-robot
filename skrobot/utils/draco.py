"""Draco compression utilities using DracoPy with color support.

This module provides Draco mesh compression with vertex color preservation,
"""

import numpy as np


def _draco_encode_with_colors(ctx):
    """
    Handle KHR_draco_mesh_compression extension for encoding a mesh primitive.

    Parameters
    ----------
    ctx : dict
        PrimitiveExportContext with:
        - mesh: trimesh.Trimesh being exported
        - name: Mesh name
        - tree: glTF tree being built (mutable)
        - buffer_items: Buffer data being built (mutable)
        - primitive: Primitive dict being built (mutable)
        - include_normals: Whether to include normals

    Returns
    -------
    result : dict or None
        Dict with extension data for KHR_draco_mesh_compression, or None on failure.
    """
    try:
        import DracoPy
    except ImportError:
        return None

    mesh = ctx["mesh"]
    buffer_items = ctx["buffer_items"]
    primitive = ctx["primitive"]
    tree = ctx["accessors"]
    include_normals = ctx["include_normals"]

    # Get mesh data
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.uint32)

    # Get optional normals
    normals = None
    if include_normals and hasattr(mesh, "vertex_normals"):
        normals = mesh.vertex_normals.astype(np.float32)

    # Get vertex colors if available
    colors = None
    if hasattr(mesh, 'visual') and mesh.visual is not None:
        if mesh.visual.kind == 'vertex':
            vc = mesh.visual.vertex_colors
            if vc is not None and len(vc) == len(vertices):
                colors = np.asarray(vc)[:, :3].astype(np.uint8)
        elif mesh.visual.kind == 'face':
            # Convert face colors to vertex colors by expanding vertices
            face_colors = np.asarray(mesh.visual.face_colors)[:, :3]
            # Create per-face vertices
            vertices = vertices[faces].reshape(-1, 3).astype(np.float32)
            if normals is not None:
                normals = normals[faces].reshape(-1, 3).astype(np.float32)
            colors = np.repeat(face_colors, 3, axis=0).astype(np.uint8)
            faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
        elif not mesh.visual.defined:
            # Use default color if visual not defined but we want to preserve it
            if hasattr(mesh.visual, 'vertex_colors'):
                vc = mesh.visual.vertex_colors
                if vc is not None and len(vc) == len(vertices):
                    colors = np.asarray(vc)[:, :3].astype(np.uint8)

        # Handle case where visual.defined is False but main_color exists
        # This is common for STL files
        if colors is None and hasattr(mesh.visual, 'main_color'):
            main_color = mesh.visual.main_color
            if main_color is not None:
                # Create uniform vertex colors from main_color
                colors = np.tile(
                    np.asarray(main_color[:3], dtype=np.uint8),
                    (len(vertices), 1)
                )

    # Encode using DracoPy
    encode_kwargs = {
        'vertices': vertices.flatten(),
        'faces': faces.flatten(),
    }
    if colors is not None:
        encode_kwargs['colors'] = colors.flatten()

    result = DracoPy.encode(**encode_kwargs)

    # Pad buffer to 4-byte alignment (glTF requirement)
    compressed = result
    padding = (4 - len(compressed) % 4) % 4
    if padding > 0:
        compressed = compressed + b'\x00' * padding

    # Add compressed buffer
    buffer_view_index = len(tree.get("bufferViews", []))
    if "bufferViews" not in tree:
        tree["bufferViews"] = []

    tree["bufferViews"].append({
        "buffer": 0,
        "byteOffset": sum(len(b) for b in buffer_items),
        "byteLength": len(compressed)
    })
    buffer_items.append(compressed)

    # Build Draco extension data
    draco_attributes = {"POSITION": 0}
    attr_id = 1
    if colors is not None:
        draco_attributes["COLOR_0"] = attr_id
        attr_id += 1

    extension_data = {
        "bufferView": buffer_view_index,
        "attributes": draco_attributes
    }

    # Update primitive with extension
    if "extensions" not in primitive:
        primitive["extensions"] = {}
    primitive["extensions"]["KHR_draco_mesh_compression"] = extension_data

    # Clear the standard attributes since data is in Draco buffer
    # Keep empty accessors for spec compliance
    primitive["attributes"] = {}

    return {"draco_encoded": True}


def _draco_decode_with_colors(ctx):
    """
    Handle KHR_draco_mesh_compression extension for decoding a glTF primitive.

    This decoder uses DracoPy which supports vertex colors.

    Parameters
    ----------
    ctx : dict
        PrimitivePreprocessContext with:
        - data: The KHR_draco_mesh_compression extension data
        - views: List of buffer views from the glTF
        - accessors: List of accessors (mutable, will be appended to)
        - primitive: The primitive dict (mutable, indices/attributes will be updated)

    Returns
    -------
    result : dict or None
        Dict with {"decompressed": True}, or None on failure.
    """
    try:
        import DracoPy
    except ImportError:
        return None

    ext_data = ctx["data"]
    views = ctx["views"]
    accessors = ctx["accessors"]
    primitive = ctx["primitive"]

    # Get the compressed data from the bufferView
    buffer_view_index = ext_data["bufferView"]
    compressed_data = bytes(views[buffer_view_index])

    # Decompress using DracoPy
    decoded = DracoPy.decode(compressed_data)

    # Get vertices - DracoPy returns a flat array, reshape to (N, 3)
    points_flat = np.array(decoded.points)
    vertices = points_flat.reshape(-1, 3).astype(np.float32)
    n_verts = len(vertices)

    # Add position accessor (just the array, not a dict)
    primitive["attributes"]["POSITION"] = len(accessors)
    accessors.append(vertices)

    # Get faces - DracoPy returns a flat array, reshape to (N, 3)
    faces_flat = np.array(decoded.faces)
    if len(faces_flat) > 0:
        faces = faces_flat.reshape(-1, 3).astype(np.uint32)
        primitive["indices"] = len(accessors)
        accessors.append(faces)

    # Get colors if available - DracoPy returns a flat array, reshape to (N, 3)
    if hasattr(decoded, 'colors') and decoded.colors is not None and len(decoded.colors) > 0:
        colors_flat = np.array(decoded.colors)
        colors = colors_flat.reshape(-1, 3).astype(np.uint8)
        if len(colors) == n_verts:
            # Add alpha channel and normalize to 0-1 range for trimesh
            colors_rgba = np.column_stack([
                colors.astype(np.float32) / 255.0,
                np.ones(n_verts, dtype=np.float32)
            ])
            primitive["attributes"]["COLOR_0"] = len(accessors)
            accessors.append(colors_rgba)

    return {"decompressed": True}


def register_dracopy_handlers():
    """Register DracoPy-based handlers with trimesh's glTF extension system.

    This registers DracoPy-based handlers that support vertex colors.
    Requires trimesh >= 4.11 with the extension registration API.
    """
    try:
        from trimesh.exchange.gltf.extensions import register_handler

        # Register decode handler for import
        register_handler("KHR_draco_mesh_compression", scope="primitive_preprocess")(
            _draco_decode_with_colors
        )

        # Register encode handler for export
        register_handler("KHR_draco_mesh_compression", scope="primitive_export")(
            _draco_encode_with_colors
        )

        return True
    except ImportError:
        return False


def is_dracopy_available():
    """Check if DracoPy is installed."""
    try:
        import DracoPy  # NOQA
        return True
    except ImportError:
        return False


def export_glb_with_draco(meshes, filename):
    """Export meshes to GLB file with Draco compression preserving colors.

    This function creates a GLB file with KHR_draco_mesh_compression extension,
    using DracoPy for encoding which supports vertex colors.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
        Meshes to export.
    filename : str
        Output filename (should end with .glb).
    """
    import json
    import struct

    import DracoPy

    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]

    # Build glTF structure
    gltf = {
        "asset": {"version": "2.0", "generator": "scikit-robot with DracoPy"},
        "extensionsUsed": ["KHR_draco_mesh_compression"],
        "extensionsRequired": ["KHR_draco_mesh_compression"],
        "bufferViews": [],
        "accessors": [],
        "meshes": [],
        "nodes": [],
        "scenes": [{"nodes": []}],
        "scene": 0
    }

    buffer_data = b""

    for mesh_idx, mesh in enumerate(meshes):
        # Get mesh data with colors
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.uint32)
        colors = None

        # Extract colors based on visual type
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            if mesh.visual.kind == 'face':
                # Convert face colors to vertex colors
                face_colors = np.asarray(mesh.visual.face_colors)[:, :3]
                vertices = vertices[faces].reshape(-1, 3).astype(np.float32)
                colors = np.repeat(face_colors, 3, axis=0).astype(np.uint8)
                faces = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
            elif mesh.visual.kind == 'vertex':
                vc = mesh.visual.vertex_colors
                if vc is not None and len(vc) == len(vertices):
                    colors = np.asarray(vc)[:, :3].astype(np.uint8)
            elif hasattr(mesh.visual, 'vertex_colors'):
                vc = mesh.visual.vertex_colors
                if vc is not None and len(vc) == len(vertices):
                    colors = np.asarray(vc)[:, :3].astype(np.uint8)

            # Handle case where visual.defined is False but main_color exists
            # This is common for STL files
            if colors is None and hasattr(mesh.visual, 'main_color'):
                main_color = mesh.visual.main_color
                if main_color is not None:
                    # Create uniform vertex colors from main_color
                    colors = np.tile(
                        np.asarray(main_color[:3], dtype=np.uint8),
                        (len(vertices), 1)
                    )

        # Encode with DracoPy
        # DracoPy.encode uses 'points' not 'vertices'
        # Colors should be shape (N, K) not flattened
        draco_buffer = DracoPy.encode(
            points=vertices,
            faces=faces.flatten(),
            colors=colors if colors is not None else None
        )

        # Pad to 4-byte alignment
        padding = (4 - len(draco_buffer) % 4) % 4
        if padding > 0:
            draco_buffer = draco_buffer + b'\x00' * padding

        # Add buffer view for Draco data
        buffer_view_index = len(gltf["bufferViews"])
        gltf["bufferViews"].append({
            "buffer": 0,
            "byteOffset": len(buffer_data),
            "byteLength": len(draco_buffer)
        })
        buffer_data += draco_buffer

        # Build Draco extension attributes
        draco_attributes = {"POSITION": 0}
        if colors is not None:
            draco_attributes["COLOR_0"] = 1

        # Create mesh primitive
        primitive = {
            "attributes": {},
            "extensions": {
                "KHR_draco_mesh_compression": {
                    "bufferView": buffer_view_index,
                    "attributes": draco_attributes
                }
            }
        }

        # Add mesh to glTF
        gltf["meshes"].append({
            "primitives": [primitive]
        })

        # Add node
        node_index = len(gltf["nodes"])
        gltf["nodes"].append({
            "mesh": mesh_idx
        })
        gltf["scenes"][0]["nodes"].append(node_index)

    # Add buffer
    gltf["buffers"] = [{
        "byteLength": len(buffer_data)
    }]

    # Create GLB file
    json_str = json.dumps(gltf, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')

    # Pad JSON to 4-byte alignment
    json_padding = (4 - len(json_bytes) % 4) % 4
    json_bytes += b' ' * json_padding

    # GLB header
    glb_header = struct.pack('<4sII', b'glTF', 2,
                             12 + 8 + len(json_bytes) + 8 + len(buffer_data))

    # JSON chunk
    json_chunk_header = struct.pack('<II', len(json_bytes), 0x4E4F534A)  # JSON

    # BIN chunk
    bin_chunk_header = struct.pack('<II', len(buffer_data), 0x004E4942)  # BIN

    # Write GLB file
    with open(filename, 'wb') as f:
        f.write(glb_header)
        f.write(json_chunk_header)
        f.write(json_bytes)
        f.write(bin_chunk_header)
        f.write(buffer_data)

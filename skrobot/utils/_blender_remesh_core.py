"""
Core Blender remeshing functions.
This module is designed to be executed within Blender's Python environment.
"""
from pathlib import Path

import bpy


def clear_scene():
    """Clear all objects from the scene"""
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def remesh_and_bake_file(input_path, output_path, voxel_size=0.002, export_format='DAE'):
    """
    Imports a mesh, remeshes it, and applies materials based on nearest face colors.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to input mesh file.
    output_path : str or pathlib.Path
        Path to output mesh file.
    voxel_size : float, optional
        Voxel size for remeshing. Default is 0.002.
    export_format : str, optional
        Export format ('DAE' or 'STL'). Default is 'DAE'.
    """
    import math

    input_path = Path(input_path)
    output_path = Path(output_path)

    clear_scene()

    # Import with fix_orientation to maintain Blender's Z-up coordinate system
    bpy.ops.wm.collada_import(
        filepath=str(input_path),
        fix_orientation=True,
        auto_connect=False,
        import_units=True
    )

    # --- 1. COLLECT ORIGINAL COLORS ---
    source_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not source_objects:
        print("Warning: No mesh objects found")
        return

    # Build a map of face center positions to colors
    face_color_data = []

    for obj in source_objects:
        world_matrix = obj.matrix_world

        # Get vertex colors if available
        if obj.data.vertex_colors:
            vc = obj.data.vertex_colors.active.data

            for poly in obj.data.polygons:
                # Get average color for this face
                color_sum = [0.0, 0.0, 0.0, 0.0]
                for loop_index in poly.loop_indices:
                    for i in range(4):
                        color_sum[i] += vc[loop_index].color[i]

                avg_color = tuple(c / len(poly.loop_indices) for c in color_sum)

                # Get face center in world space
                face_center = world_matrix @ poly.center

                face_color_data.append({
                    'center': face_center.copy(),
                    'color': avg_color[:3]
                })

    # --- 2. JOIN ALL OBJECTS ---
    bpy.ops.object.select_all(action='DESELECT')
    for obj in source_objects:
        obj.select_set(True)
    if len(source_objects) > 1:
        bpy.context.view_layer.objects.active = source_objects[0]
        bpy.ops.object.join()

    joined_obj = bpy.context.active_object

    # --- 3. APPLY REMESH ---
    remesh_mod = joined_obj.modifiers.new(name="Remesh", type='REMESH')
    remesh_mod.mode = 'VOXEL'
    remesh_mod.voxel_size = voxel_size
    remesh_mod.use_remove_disconnected = True
    bpy.ops.object.modifier_apply(modifier=remesh_mod.name)

    # --- 4. ASSIGN MATERIALS BASED ON NEAREST ORIGINAL FACE ---
    if face_color_data:
        from mathutils.kdtree import KDTree

        # Build KD-tree for fast nearest neighbor search
        kd = KDTree(len(face_color_data))
        for i, data in enumerate(face_color_data):
            kd.insert(data['center'], i)
        kd.balance()

        # Clear existing materials
        joined_obj.data.materials.clear()

        # Map colors to material indices
        color_to_material = {}
        world_matrix = joined_obj.matrix_world

        for poly in joined_obj.data.polygons:
            # Get face center in world space
            face_center = world_matrix @ poly.center

            # Find nearest original face
            _nearest_co, nearest_index, _nearest_dist = kd.find(face_center)
            nearest_color = face_color_data[nearest_index]['color']

            # Round color to reduce number of materials
            rounded_color = tuple(round(c * 50) / 50 for c in nearest_color)

            # Create material if it doesn't exist
            if rounded_color not in color_to_material:
                mat = bpy.data.materials.new(name=f"Color_{len(color_to_material):03d}")
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes.get("Principled BSDF")
                if bsdf:
                    bsdf.inputs['Base Color'].default_value = (*rounded_color, 1.0)

                color_to_material[rounded_color] = len(joined_obj.data.materials)
                joined_obj.data.materials.append(mat)

            # Assign material to face
            poly.material_index = color_to_material[rounded_color]

    # Apply smooth shading
    if joined_obj.data.polygons:
        joined_obj.data.polygons.foreach_set('use_smooth', [True] * len(joined_obj.data.polygons))

    # --- 5. EXPORT ---
    # Convert from Blender Z-up to Collada Y-up by rotating X-axis -90 degrees
    joined_obj.rotation_euler[0] -= math.pi / 2

    # Apply transformation to vertices
    bpy.context.view_layer.objects.active = joined_obj
    joined_obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    if export_format.upper() == 'DAE':
        bpy.ops.wm.collada_export(
            filepath=str(output_path),
            selected=True,
            include_children=True,
            include_armatures=False,
            include_shapekeys=False,
            apply_modifiers=True,
            triangulate=True,
            use_object_instantiation=False,
            sort_by_name=False
        )
    elif export_format.upper() == 'STL':
        bpy.ops.export_mesh.stl(
            filepath=str(output_path),
            use_selection=True,
            ascii=False
        )

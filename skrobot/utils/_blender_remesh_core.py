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
    import xml.etree.ElementTree as ET

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Detect original coordinate system from DAE file
    original_up_axis = 'Y_UP'  # default
    if export_format.upper() == 'DAE' and input_path.suffix.lower() == '.dae':
        try:
            tree = ET.parse(input_path)
            root = tree.getroot()
            # Find up_axis element (handle namespace)
            ns = {'collada': 'http://www.collada.org/2005/11/COLLADASchema'}
            up_axis_elem = root.find('.//collada:up_axis', ns)
            if up_axis_elem is None:
                up_axis_elem = root.find('.//up_axis')
            if up_axis_elem is not None and up_axis_elem.text:
                original_up_axis = up_axis_elem.text.strip()
                print(f"Detected original coordinate system: {original_up_axis}")
        except Exception as e:
            print(f"Warning: Could not detect coordinate system: {e}")

    clear_scene()

    # Import with fix_orientation to ensure proper handling in Blender
    # Blender uses Z-up, so this converts Y-up to Z-up properly
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
    objects_with_vertex_colors = 0
    objects_with_materials = 0

    for obj in source_objects:
        world_matrix = obj.matrix_world

        # Try to get colors from vertex colors first, then from materials
        has_vertex_colors = obj.data.vertex_colors and len(obj.data.vertex_colors) > 0

        if has_vertex_colors:
            objects_with_vertex_colors += 1
            vc = obj.data.vertex_colors.active.data

            for poly in obj.data.polygons:
                # Get average color for this face from vertex colors
                color_sum = [0.0, 0.0, 0.0, 0.0]
                for loop_index in poly.loop_indices:
                    for i in range(4):
                        color_sum[i] += vc[loop_index].color[i]
                avg_color = tuple(c / len(poly.loop_indices) for c in color_sum)
                face_color = avg_color[:3]

                # Get face center in world space
                face_center = world_matrix @ poly.center

                face_color_data.append({
                    'center': face_center.copy(),
                    'color': face_color
                })
        elif obj.data.materials:
            objects_with_materials += 1

            for poly in obj.data.polygons:
                # Get color from material
                face_color = (0.5, 0.5, 0.5)  # default fallback

                if poly.material_index < len(obj.data.materials):
                    mat = obj.data.materials[poly.material_index]
                    if mat and mat.use_nodes:
                        # Get base color from Principled BSDF
                        bsdf = mat.node_tree.nodes.get("Principled BSDF")
                        if bsdf and 'Base Color' in bsdf.inputs:
                            base_color = bsdf.inputs['Base Color'].default_value
                            face_color = (base_color[0], base_color[1], base_color[2])

                # Get face center in world space
                face_center = world_matrix @ poly.center

                face_color_data.append({
                    'center': face_center.copy(),
                    'color': face_color
                })

    msg = f"Color collection: {objects_with_vertex_colors} objects with vertex colors"
    msg += f'{objects_with_materials} with materials'
    msg += f"Collected {len(face_color_data)} face color samples"
    print(msg)

    # Print color statistics
    if face_color_data:
        unique_colors = set(data['color'] for data in face_color_data)
        print(f"Unique colors in source: {len(unique_colors)}")
        # Sample some colors
        sample_colors = list(unique_colors)[:5]
        for i, c in enumerate(sample_colors):
            print(f"  Sample {i + 1}: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")

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
            # Use finer quantization (0.01 steps) to preserve more color detail
            rounded_color = tuple(round(c * 100) / 100 for c in nearest_color)

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

        print(f"Created {len(color_to_material)} materials from {len(face_color_data)} source colors")
        print("Material color samples:")
        for i, (color, mat_idx) in enumerate(list(color_to_material.items())[:5]):
            print(f"  Material {mat_idx}: ({color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f})")

    # Apply smooth shading
    if joined_obj.data.polygons:
        joined_obj.data.polygons.foreach_set('use_smooth', [True] * len(joined_obj.data.polygons))

    # --- 5. EXPORT ---
    # Convert back to original coordinate system if needed
    if export_format.upper() == 'DAE' and original_up_axis == 'Y_UP':
        # Convert from Blender Z-up back to Y-up
        # Rotate -90 degrees around X axis
        joined_obj.rotation_euler[0] -= math.pi / 2
        bpy.context.view_layer.objects.active = joined_obj
        joined_obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Ensure object is selected for export
    bpy.context.view_layer.objects.active = joined_obj
    joined_obj.select_set(True)

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

        # Fix coordinate system in exported DAE file to match original
        # Use text replacement to preserve all XML formatting and content
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace Z_UP with original coordinate system
            content = content.replace('<up_axis>Z_UP</up_axis>', f'<up_axis>{original_up_axis}</up_axis>')
            # Also handle namespace prefix variations
            content = content.replace(':up_axis>Z_UP</', f':up_axis>{original_up_axis}</')

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Restored coordinate system to: {original_up_axis}")
        except Exception as e:
            print(f"Warning: Could not fix coordinate system in output: {e}")

    elif export_format.upper() == 'STL':
        bpy.ops.export_mesh.stl(
            filepath=str(output_path),
            use_selection=True,
            ascii=False
        )

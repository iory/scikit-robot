"""
Core Blender decimation functions.
This module is designed to be executed within Blender's Python environment.

Unlike the voxel remesher, decimation (collapse) reduces the triangle count
while preserving the original shape, materials and vertex colors. The mesh I/O
is done in glTF/glb format because Blender 5.x no longer ships the Collada
(``.dae``) importer/exporter; glTF is supported natively on every Blender
version that includes ``bpy`` and round-trips materials and vertex colors.
"""
from pathlib import Path

import bpy


def clear_scene():
    """Clear all objects from the scene."""
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def decimate_glb_file(input_path, output_path, ratio=0.1):
    """Import a glb, apply the decimate (collapse) modifier and re-export glb.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to the input ``.glb`` file.
    output_path : str or pathlib.Path
        Path to the output ``.glb`` file.
    ratio : float, optional
        Collapse ratio passed to Blender's decimate modifier. The resulting
        triangle count is approximately ``ratio`` times the original. Must be
        in the range ``(0, 1]``. Default is 0.1.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    clear_scene()

    bpy.ops.import_scene.gltf(filepath=str(input_path))

    objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not objs:
        print("Warning: No mesh objects found in input")
        return
    before = sum(len(o.data.polygons) for o in objs)

    # Join all mesh objects so a single decimate modifier covers the whole link.
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    # Always set the active object explicitly; with a single imported mesh
    # the active object would otherwise be None and join() is skipped.
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object

    mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=mod.name)

    after = len(obj.data.polygons)
    print(f"Decimation complete. Faces: {before} -> {after} (ratio={ratio})")

    if obj.data.polygons:
        obj.data.polygons.foreach_set(
            'use_smooth', [True] * len(obj.data.polygons))

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        use_selection=True,
        export_format='GLB',
    )

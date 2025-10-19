"""Convert URDF meshes to primitive shapes."""

from logging import getLogger
from pathlib import Path

from lxml import etree as ET
import numpy as np

from skrobot.coordinates.math import matrix2ypr
from skrobot.coordinates.math import rotation_matrix_z_to_axis
from skrobot.coordinates.math import rpy_matrix
from skrobot.utils.primitive_fitting import fit_primitive_to_mesh
from skrobot.utils.urdf import get_filename
from skrobot.utils.urdf import load_meshes


logger = getLogger(__name__)


def extract_mesh_color(meshes):
    """Extract representative color from mesh(es).

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
        List of meshes to extract color from.

    Returns
    -------
    rgba : np.ndarray or None
        RGBA color array [R, G, B, A] in range [0, 1], or None if no color found.
    """
    if not meshes:
        return None

    from skrobot._lazy_imports import _lazy_trimesh
    _lazy_trimesh()

    colors = []

    for mesh in meshes:
        if hasattr(mesh.visual, 'material'):
            material = mesh.visual.material
            if hasattr(material, 'baseColorFactor') and material.baseColorFactor is not None:
                color = np.array(material.baseColorFactor)
                if color.max() > 1.0:
                    color = color / 255.0
                colors.append(color)
            elif hasattr(material, 'diffuse') and material.diffuse is not None:
                color = np.array(material.diffuse)
                if color.max() > 1.0:
                    color = color / 255.0
                colors.append(color)
            elif hasattr(material, 'main_color') and material.main_color is not None:
                color = np.array(material.main_color)
                if color.max() > 1.0:
                    color = color / 255.0
                colors.append(color)

        if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
            face_colors = mesh.visual.face_colors
            if len(face_colors) > 0:
                avg_color = np.mean(face_colors, axis=0)
                if avg_color.max() > 1.0:
                    avg_color = avg_color / 255.0
                colors.append(avg_color)

        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vertex_colors = mesh.visual.vertex_colors
            if len(vertex_colors) > 0:
                avg_color = np.mean(vertex_colors, axis=0)
                if avg_color.max() > 1.0:
                    avg_color = avg_color / 255.0
                colors.append(avg_color)

    if not colors:
        return None

    avg_rgba = np.mean(colors, axis=0)

    if len(avg_rgba) == 3:
        avg_rgba = np.append(avg_rgba, 1.0)

    return avg_rgba


def detect_indentation(root):
    """Detect indentation patterns from existing XML structure.

    Parameters
    ----------
    root : ET.Element
        Root element of the URDF XML tree.

    Returns
    -------
    indent_patterns : dict
        Dictionary containing indentation strings for various elements.
    """
    visual_examples = root.findall('.//visual')
    collision_examples = root.findall('.//collision')

    examples = visual_examples + collision_examples

    if examples:
        elem = examples[0]

        elem_indent = '    '
        if hasattr(elem, 'tail') and elem.tail and '\n' in elem.tail:
            lines = elem.tail.split('\n')
            if len(lines) > 1:
                parent_indent = lines[-1]
                elem_indent = parent_indent + '  '

        child_indent = '      '
        if elem.text and '\n' in elem.text:
            lines = elem.text.split('\n')
            if len(lines) > 1:
                child_indent = lines[-1]

        base_unit = '  '

        return {
            'elem_indent': f'\n{child_indent}',
            'elem_tail': f'\n{elem_indent.rstrip()}',
            'origin_tail': f'\n{child_indent}',
            'geometry_text': f'\n{child_indent}{base_unit}',
            'geometry_tail': f'\n{elem_indent}',
            'primitive_tail': f'\n{child_indent}'
        }

    return {
        'elem_indent': '\n      ',
        'elem_tail': '\n    ',
        'origin_tail': '\n      ',
        'geometry_text': '\n        ',
        'geometry_tail': '\n    ',
        'primitive_tail': '\n      '
    }


def create_primitive_geometry_element(primitive_params, indent_patterns):
    """Create XML geometry element for a primitive shape.

    Parameters
    ----------
    primitive_params : dict
        Dictionary with primitive parameters from fit_primitive_to_mesh.
    indent_patterns : dict
        Indentation patterns for XML formatting.

    Returns
    -------
    geometry : ET.Element
        XML geometry element.
    origin_xyz : np.ndarray
        Origin position for the primitive.
    origin_rpy : np.ndarray
        Origin rotation (RPY) for the primitive.
    """
    geometry = ET.Element('geometry')
    center = primitive_params['center']
    prim_type = primitive_params['type']

    if prim_type == 'box':
        extents = primitive_params['extents']
        box = ET.SubElement(geometry, 'box')
        box.set('size', f"{extents[0]} {extents[1]} {extents[2]}")
        box.tail = indent_patterns['primitive_tail']

        rotation = primitive_params.get('rotation', np.eye(3))
        ypr = matrix2ypr(rotation)
        origin_rpy = np.array([ypr[2], ypr[1], ypr[0]])

    elif prim_type == 'sphere':
        radius = primitive_params['radius']
        sphere = ET.SubElement(geometry, 'sphere')
        sphere.set('radius', str(radius))
        sphere.tail = indent_patterns['primitive_tail']

        origin_rpy = np.zeros(3)

    elif prim_type == 'cylinder':
        radius = primitive_params['radius']
        height = primitive_params['height']
        axis = primitive_params['axis']

        cylinder = ET.SubElement(geometry, 'cylinder')
        cylinder.set('radius', str(radius))
        cylinder.set('length', str(height))
        cylinder.tail = indent_patterns['primitive_tail']

        rotation = rotation_matrix_z_to_axis(axis)
        ypr = matrix2ypr(rotation)
        origin_rpy = np.array([ypr[2], ypr[1], ypr[0]])

    elif prim_type == 'capsule':
        logger.warning("Capsule primitives are not standard URDF. Converting to cylinder.")
        radius = primitive_params['radius']
        height = primitive_params['height']
        axis = primitive_params['axis']

        cylinder = ET.SubElement(geometry, 'cylinder')
        cylinder.set('radius', str(radius))
        total_height = height + 2 * radius
        cylinder.set('length', str(total_height))
        cylinder.tail = indent_patterns['primitive_tail']

        rotation = rotation_matrix_z_to_axis(axis)
        ypr = matrix2ypr(rotation)
        origin_rpy = np.array([ypr[2], ypr[1], ypr[0]])

    else:
        raise ValueError(f"Unknown primitive type: {prim_type}")

    geometry.text = indent_patterns['geometry_text']

    return geometry, center, origin_rpy


def convert_meshes_to_primitives(
        urdf_file,
        output_file=None,
        convert_visual=True,
        convert_collision=True,
        primitive_type=None):
    """Convert mesh geometries in URDF to primitive shapes.

    Parameters
    ----------
    urdf_file : str or Path
        Path to the input URDF file.
    output_file : str or Path, optional
        Path to the output URDF file. If None, modifies in place.
    convert_visual : bool, optional
        Whether to convert visual meshes to primitives. Default is True.
    convert_collision : bool, optional
        Whether to convert collision meshes to primitives. Default is True.
    primitive_type : str, optional
        Force a specific primitive type ('box', 'cylinder', 'sphere').
        If None, automatically detects the best fit for each mesh.

    Returns
    -------
    modified_count : int
        Number of geometry elements that were converted.
    """
    urdf_path = Path(urdf_file)
    urdf_dir = urdf_path.parent

    parser = ET.XMLParser(remove_blank_text=False, remove_comments=False)
    tree = ET.parse(str(urdf_file), parser)

    root = tree.getroot()

    indent_patterns = detect_indentation(root)
    logger.debug("Detected indentation patterns: %s", indent_patterns)

    modified_count = 0

    for link in root.findall('link'):
        link_name = link.get('name')
        logger.info("Processing link: %s", link_name)

        elements_to_process = []
        if convert_visual:
            elements_to_process.extend(link.findall('visual'))
        if convert_collision:
            elements_to_process.extend(link.findall('collision'))

        for elem in elements_to_process:
            elem_type = elem.tag
            geom = elem.find('geometry')
            if geom is None:
                continue

            mesh_elem = geom.find('mesh')
            if mesh_elem is None:
                continue

            mesh_file = mesh_elem.get('filename')
            scale_str = mesh_elem.get('scale', '1 1 1')
            scale = np.array([float(s) for s in scale_str.split()])

            logger.info("  Converting %s mesh: %s", elem_type, mesh_file)

            origin_elem = elem.find('origin')
            if origin_elem is not None:
                origin_xyz_str = origin_elem.get('xyz', '0 0 0')
                origin_rpy_str = origin_elem.get('rpy', '0 0 0')
                origin_xyz = np.array([float(x) for x in origin_xyz_str.split()])
                origin_rpy = np.array([float(r) for r in origin_rpy_str.split()])
            else:
                origin_xyz = np.zeros(3)
                origin_rpy = np.zeros(3)

            try:
                resolved_filename = get_filename(str(urdf_dir), mesh_file)
                if resolved_filename is None:
                    logger.warning("Could not resolve mesh file: %s", mesh_file)
                    continue

                meshes = load_meshes(resolved_filename)
                if not meshes or len(meshes) == 0:
                    logger.warning("No meshes loaded from: %s", resolved_filename)
                    continue

                from skrobot._lazy_imports import _lazy_trimesh
                trimesh = _lazy_trimesh()
                if len(meshes) == 1:
                    combined_mesh = meshes[0]
                else:
                    combined_mesh = trimesh.util.concatenate(meshes)

                if not np.allclose(scale, 1.0):
                    combined_mesh.apply_scale(scale)

                primitive_params = fit_primitive_to_mesh(
                    combined_mesh,
                    primitive_type=primitive_type
                )

                logger.info("    Fitted primitive: %s", primitive_params['type'])

                rotation_urdf = rpy_matrix(
                    origin_rpy[2],
                    origin_rpy[1],
                    origin_rpy[0]
                )
                mesh_center_link = rotation_urdf @ primitive_params['center']
                primitive_center = origin_xyz + mesh_center_link

                new_geometry, rel_center, rel_rpy = create_primitive_geometry_element(
                    primitive_params,
                    indent_patterns
                )

                if primitive_params['type'] in ['cylinder', 'capsule']:
                    axis_link = rotation_urdf @ primitive_params['axis']
                    primitive_rotation = rotation_matrix_z_to_axis(axis_link)
                    combined_rotation = primitive_rotation
                    ypr = matrix2ypr(combined_rotation)
                    final_rpy = np.array([ypr[2], ypr[1], ypr[0]])
                else:
                    final_rpy = origin_rpy

                original_order = []
                material_elem = None
                for child in elem:
                    original_order.append(child.tag)
                    if child.tag == 'material':
                        material_elem = child

                if material_elem is None:
                    rgba = extract_mesh_color(meshes)
                    if rgba is not None:
                        material_elem = ET.Element('material')
                        material_elem.set('name', f'{link_name}_{elem_type}_material')
                        color_elem = ET.SubElement(material_elem, 'color')
                        color_elem.set('rgba', f'{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}')
                        logger.info("    Extracted color from mesh: rgba=%.3f %.3f %.3f %.3f",
                                    rgba[0], rgba[1], rgba[2], rgba[3])

                elem.clear()

                if 'geometry' in original_order and 'origin' in original_order:
                    if original_order.index('geometry') < original_order.index('origin'):
                        elem.append(new_geometry)

                        if material_elem is not None:
                            elem.append(material_elem)

                        new_origin = ET.SubElement(elem, 'origin')
                        new_origin.set('xyz', f"{primitive_center[0]} {primitive_center[1]} {primitive_center[2]}")
                        new_origin.set('rpy', f"{final_rpy[0]} {final_rpy[1]} {final_rpy[2]}")
                    else:
                        new_origin = ET.SubElement(elem, 'origin')
                        new_origin.set('xyz', f"{primitive_center[0]} {primitive_center[1]} {primitive_center[2]}")
                        new_origin.set('rpy', f"{final_rpy[0]} {final_rpy[1]} {final_rpy[2]}")

                        if material_elem is not None:
                            elem.append(material_elem)

                        elem.append(new_geometry)
                else:
                    elem.append(new_geometry)

                    if material_elem is not None:
                        elem.append(material_elem)

                    new_origin = ET.SubElement(elem, 'origin')
                    new_origin.set('xyz', f"{primitive_center[0]} {primitive_center[1]} {primitive_center[2]}")
                    new_origin.set('rpy', f"{final_rpy[0]} {final_rpy[1]} {final_rpy[2]}")

                elem.text = indent_patterns['elem_indent']
                elem.tail = '\n  '

                if len(elem) > 0:
                    first_element = elem[0]
                    if len(elem) > 1:
                        first_element.tail = indent_patterns['origin_tail']
                        elem[-1].tail = '\n    '
                    else:
                        first_element.tail = '\n    '

                modified_count += 1
                logger.info("    Converted to %s primitive", primitive_params['type'])

            except Exception as e:
                logger.error("Error processing mesh %s: %s", mesh_file, e)
                continue

    output_path = output_file if output_file is not None else urdf_file

    tree.write(
        str(output_path),
        encoding='utf-8',
        xml_declaration=True,
        pretty_print=True
    )

    if output_file is not None:
        logger.info("Modified URDF saved to: %s", output_file)
    else:
        logger.info("Modified URDF in place: %s", urdf_file)

    logger.info("Converted %d geometry elements to primitives", modified_count)

    return modified_count

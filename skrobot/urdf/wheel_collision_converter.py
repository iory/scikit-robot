from logging import getLogger
from pathlib import Path

from lxml import etree as ET
import numpy as np

from skrobot._lazy_imports import _lazy_trimesh
from skrobot.coordinates.math import matrix2ypr
from skrobot.coordinates.math import rotation_matrix_z_to_axis
from skrobot.coordinates.math import rpy_matrix
from skrobot.utils.urdf import get_filename
from skrobot.utils.urdf import load_meshes


logger = getLogger(__name__)


def get_mesh_dimensions(mesh_file, scale=None, urdf_dir=None):
    """Calculate dimensions from a mesh file using skrobot utilities."""
    try:
        resolved_filename = get_filename(str(urdf_dir) if urdf_dir else None, mesh_file)
        if resolved_filename is None:
            logger.warning("Could not resolve mesh file: %s", mesh_file)
            return None, None, None, None

        meshes = load_meshes(resolved_filename)
        if not meshes or len(meshes) == 0:
            logger.warning("No meshes loaded from: %s", resolved_filename)
            return None, None, None, None

        trimesh = _lazy_trimesh()
        if len(meshes) == 1:
            combined_mesh = meshes[0]
        else:
            combined_mesh = trimesh.util.concatenate(meshes)

        if scale is not None:
            combined_mesh.apply_scale(scale)

        bounds = combined_mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        dimensions = bounds[1] - bounds[0]

        logger.debug("Mesh center: %s", center)
        logger.debug("Mesh dimensions: %s", dimensions)

        height_idx = np.argmin(dimensions)
        height = dimensions[height_idx]

        other_dims = [dimensions[i] for i in range(3) if i != height_idx]
        radius = max(other_dims) / 2

        logger.info("Detected wheel dimensions: radius=%.4f, thickness=%.4f", radius, height)

        axis = np.zeros(3)
        axis[height_idx] = 1

        return radius, height, axis, center

    except Exception as e:
        logger.error("Error loading mesh: %s", e)
        return None, None, None, None


def detect_indentation(root):
    """Detect indentation patterns from existing XML structure."""
    collision_examples = root.findall('.//collision')

    if collision_examples:
        collision = collision_examples[0]

        collision_indent = '    '
        if hasattr(collision, 'tail') and collision.tail and '\n' in collision.tail:
            lines = collision.tail.split('\n')
            if len(lines) > 1:
                collision_parent_indent = lines[-1]
                collision_indent = collision_parent_indent + '  '

        collision_child_indent = '      '
        if collision.text and '\n' in collision.text:
            lines = collision.text.split('\n')
            if len(lines) > 1:
                collision_child_indent = lines[-1]

        len(collision_child_indent)
        base_unit = '  '

        return {
            'collision_indent': f'\n{collision_child_indent}',
            'collision_tail': f'\n{collision_indent.rstrip()}',
            'origin_tail': f'\n{collision_child_indent}',
            'geometry_text': f'\n{collision_child_indent}{base_unit}',
            'geometry_tail': f'\n{collision_indent}',
            'cylinder_tail': f'\n{collision_child_indent}'
        }

    return {
        'collision_indent': '\n      ',
        'collision_tail': '\n    ',
        'origin_tail': '\n      ',
        'geometry_text': '\n        ',
        'geometry_tail': '\n    ',
        'cylinder_tail': '\n      '
    }


def convert_wheel_collisions_to_cylinders(urdf_file, output_file=None):
    """
    Convert collision meshes of continuous joint child links to cylinders.

    Parameters
    ----------
    urdf_file : str or Path
        Path to the input URDF file
    output_file : str or Path, optional
        Path to the output URDF file. If None, modifies in place

    Returns
    -------
    modified_links : list
        List of link names that were modified
    """
    urdf_path = Path(urdf_file)
    urdf_dir = urdf_path.parent

    parser = ET.XMLParser(remove_blank_text=False, remove_comments=False)
    tree = ET.parse(str(urdf_file), parser)

    root = tree.getroot()

    indent_patterns = detect_indentation(root)
    logger.debug("Detected indentation patterns: %s", indent_patterns)

    continuous_joints = []
    for joint in root.findall('joint'):
        if joint.get('type') == 'continuous':
            child_elem = joint.find('child')
            if child_elem is not None:
                continuous_joints.append({
                    'name': joint.get('name'),
                    'child': child_elem.get('link')
                })

    logger.info("Found %d continuous joints", len(continuous_joints))
    modified_links = []

    for joint_info in continuous_joints:
        joint_name = joint_info['name']
        child_link_name = joint_info['child']

        logger.info("Processing link: %s (child of joint: %s)", child_link_name, joint_name)

        link = root.find(f".//link[@name='{child_link_name}']")

        if link is None:
            logger.warning("Could not find link %s", child_link_name)
            continue

        visual = link.find('visual')
        if visual is None:
            logger.warning("No visual element for link %s", child_link_name)
            continue

        visual_geom = visual.find('geometry')
        visual_mesh = visual_geom.find('mesh') if visual_geom is not None else None

        if visual_mesh is None:
            logger.warning("No visual mesh for link %s", child_link_name)
            continue

        mesh_file = visual_mesh.get('filename')
        scale_str = visual_mesh.get('scale', '1 1 1')
        scale = np.array([float(s) for s in scale_str.split()])

        logger.info("Visual mesh: %s", mesh_file)
        logger.debug("Scale: %s", scale)

        visual_origin = visual.find('origin')
        if visual_origin is not None:
            origin_xyz_str = visual_origin.get('xyz', '0 0 0')
            origin_rpy_str = visual_origin.get('rpy', '0 0 0')
            origin_xyz = np.array([float(x) for x in origin_xyz_str.split()])
            origin_rpy = np.array([float(r) for r in origin_rpy_str.split()])
        else:
            origin_xyz = np.zeros(3)
            origin_rpy = np.zeros(3)

        rotation_urdf = rpy_matrix(origin_rpy[2], origin_rpy[1], origin_rpy[0])
        radius, height, axis, mesh_center_offset = get_mesh_dimensions(mesh_file, scale, urdf_dir)

        if radius is None:
            logger.warning("Could not calculate dimensions for %s", child_link_name)
            radius, height = 0.025, 0.02
            axis, mesh_center_offset = np.array([0, 0, 1]), np.zeros(3)
            logger.info("Using default values: radius=%.3f, height=%.3f", radius, height)

        mesh_center_link = rotation_urdf @ mesh_center_offset
        cylinder_center = origin_xyz + mesh_center_link
        axis_link = rotation_urdf @ axis
        cylinder_rotation_matrix = rotation_matrix_z_to_axis(axis_link)
        cylinder_ypr = matrix2ypr(cylinder_rotation_matrix)
        cylinder_rpy = np.array([cylinder_ypr[2], cylinder_ypr[1], cylinder_ypr[0]])

        collision = link.find('collision')
        if collision is None:
            collision = ET.SubElement(link, 'collision')

        original_order = []
        for child in collision:
            original_order.append(child.tag)

        collision.clear()

        if 'geometry' in original_order and 'origin' in original_order:
            if original_order.index('geometry') < original_order.index('origin'):
                geometry = ET.SubElement(collision, 'geometry')
                cylinder = ET.SubElement(geometry, 'cylinder')
                cylinder.set('radius', str(radius))
                cylinder.set('length', str(height))

                origin = ET.SubElement(collision, 'origin')
                origin.set('xyz', f"{cylinder_center[0]} {cylinder_center[1]} {cylinder_center[2]}")
                origin.set('rpy', f"{cylinder_rpy[0]} {cylinder_rpy[1]} {cylinder_rpy[2]}")
            else:
                origin = ET.SubElement(collision, 'origin')
                origin.set('xyz', f"{cylinder_center[0]} {cylinder_center[1]} {cylinder_center[2]}")
                origin.set('rpy', f"{cylinder_rpy[0]} {cylinder_rpy[1]} {cylinder_rpy[2]}")

                geometry = ET.SubElement(collision, 'geometry')
                cylinder = ET.SubElement(geometry, 'cylinder')
                cylinder.set('radius', str(radius))
                cylinder.set('length', str(height))
        else:
            geometry = ET.SubElement(collision, 'geometry')
            cylinder = ET.SubElement(geometry, 'cylinder')
            cylinder.set('radius', str(radius))
            cylinder.set('length', str(height))

            origin = ET.SubElement(collision, 'origin')
            origin.set('xyz', f"{cylinder_center[0]} {cylinder_center[1]} {cylinder_center[2]}")
            origin.set('rpy', f"{cylinder_rpy[0]} {cylinder_rpy[1]} {cylinder_rpy[2]}")

        collision.text = indent_patterns['collision_indent']
        collision.tail = '\n  '

        first_element = collision[0] if len(collision) > 0 else None
        second_element = collision[1] if len(collision) > 1 else None

        if first_element is not None:
            if first_element.tag == 'geometry':
                first_element.text = indent_patterns['geometry_text']
                if second_element is not None:
                    first_element.tail = indent_patterns['origin_tail']
                else:
                    first_element.tail = '\n    '
                if len(first_element) > 0:
                    first_element[0].tail = indent_patterns['cylinder_tail']
            else:
                if second_element is not None:
                    first_element.tail = indent_patterns['origin_tail']
                else:
                    first_element.tail = '\n    '

        if second_element is not None:
            if second_element.tag == 'geometry':
                second_element.text = indent_patterns['geometry_text']
                second_element.tail = '\n    '
                if len(second_element) > 0:
                    second_element[0].tail = indent_patterns['cylinder_tail']
            else:
                second_element.tail = '\n    '

        logger.info("Converted collision to cylinder: radius=%.4f, length=%.4f", radius, height)
        logger.debug("Collision origin: xyz=%s, rpy=%s", cylinder_center, cylinder_rpy)

        modified_links.append(child_link_name)

    output_path = output_file if output_file is not None else urdf_file

    tree.write(str(output_path),
               encoding='utf-8',
               xml_declaration=True,
               pretty_print=True)

    if output_file is not None:
        logger.info("Modified URDF saved to: %s", output_file)
    else:
        logger.info("Modified URDF in place: %s", urdf_file)

    if not modified_links:
        logger.info("No continuous joint child links found to convert")
    else:
        logger.info("Modified %d links: %s", len(modified_links), ', '.join(modified_links))

    return modified_links

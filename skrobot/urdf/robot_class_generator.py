"""Robot class generator from URDF geometry.

Generates robot classes with kinematic chain properties (right_arm, left_arm, etc.)
purely from URDF structure and naming conventions - no LLM required.

Example
-------
>>> from skrobot.models import Panda
>>> from skrobot.urdf.robot_class_generator import generate_robot_class_from_geometry
>>>
>>> robot = Panda()
>>> code = generate_robot_class_from_geometry(robot, output_path='/tmp/MyPanda.py')
"""

import logging
import os
import re

import networkx as nx
import numpy as np


logger = logging.getLogger(__name__)


def _convert_to_ros_package_path(urdf_path):
    """Convert absolute path to ROS package:// path if possible.

    Parameters
    ----------
    urdf_path : str
        Absolute path to URDF file.

    Returns
    -------
    str
        ROS package path (package://pkg_name/path) if inside a ROS package,
        otherwise returns the original path.
    """
    if urdf_path is None:
        return None

    urdf_path = os.path.abspath(urdf_path)

    # Walk up the directory tree to find package.xml
    current_dir = os.path.dirname(urdf_path)
    package_root = None
    package_name = None

    while current_dir and current_dir != os.path.dirname(current_dir):
        package_xml = os.path.join(current_dir, 'package.xml')
        if os.path.exists(package_xml):
            # Found a ROS package, extract package name
            package_root = current_dir
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(package_xml)
                root = tree.getroot()
                name_elem = root.find('name')
                if name_elem is not None and name_elem.text:
                    package_name = name_elem.text.strip()
                else:
                    # Fallback to directory name
                    package_name = os.path.basename(current_dir)
            except Exception:
                # Fallback to directory name
                package_name = os.path.basename(current_dir)
            break
        current_dir = os.path.dirname(current_dir)

    if package_root and package_name:
        # Calculate relative path from package root
        rel_path = os.path.relpath(urdf_path, package_root)
        return f"package://{package_name}/{rel_path}"

    return urdf_path


class GroupDefinition:
    """Container for robot group definitions.

    Parameters
    ----------
    robot_name : str
        Name of the robot.
    urdf_path : str, optional
        Path to the URDF file.
    groups : dict
        Dictionary of group definitions.
    end_effectors : dict
        Dictionary mapping group names to end effector link names.
    end_coords_info : dict
        Dictionary mapping group names to end_coords info (parent_link, pos, rot).
    """

    def __init__(self, robot_name, urdf_path=None, groups=None,
                 end_effectors=None, end_coords_info=None):
        self.robot_name = robot_name
        self.urdf_path = urdf_path
        self.groups = groups or {}
        self.end_effectors = end_effectors or {}
        self.end_coords_info = end_coords_info or {}


# ============================================================================
# End-effector estimation utilities
# ============================================================================

def _is_tool_frame(link_name):
    """Check if link name suggests it's a tool frame.

    Parameters
    ----------
    link_name : str
        Name of the link.

    Returns
    -------
    bool
        True if link name suggests it's a tool frame.
    """
    tool_keywords = [
        'tool_frame', 'tool_link', 'tcp', 'end_effector', 'ee_link',
        'optical_frame', 'finger_tip_frame', 'tip_frame', 'sensor_frame',
        'camera_frame', 'hand_tcp', 'gripper_tool_frame',
    ]
    link_lower = link_name.lower()
    return any(keyword in link_lower for keyword in tool_keywords)


def _is_hand_link(link_name):
    """Check if link name suggests it's a hand/flange link.

    Parameters
    ----------
    link_name : str
        Name of the link.

    Returns
    -------
    bool
        True if link name suggests it's a hand/flange link.
    """
    hand_keywords = ['hand', 'flange', 'palm', 'wrist_roll', 'ee_link']
    link_lower = link_name.lower()
    return any(keyword in link_lower for keyword in hand_keywords)


def _find_symmetric_gripper_midpoint(descendants, link_map):
    """Find symmetric gripper links and return their midpoint.

    Detects gripper fingers geometrically by finding links that:
    1. Share the same parent
    2. Have symmetric positions (opposite signs in x or y)

    Parameters
    ----------
    descendants : set
        Set of descendant link names.
    link_map : dict
        Dictionary mapping link names to Link objects.

    Returns
    -------
    dict or None
        If symmetric gripper found, returns:
        {'parent_link': str, 'pos': list, 'rot': None}
        Otherwise None.
    """
    # Group descendants by parent
    parent_to_children = {}
    for desc_name in descendants:
        desc_link = link_map.get(desc_name)
        if desc_link is None or desc_link.parent_link is None:
            continue
        parent_name = desc_link.parent_link.name
        if parent_name not in parent_to_children:
            parent_to_children[parent_name] = []
        parent_to_children[parent_name].append(desc_link)

    # Find parent with multiple children that are symmetric
    for parent_name, children in parent_to_children.items():
        if len(children) < 2:
            continue

        parent_link = link_map.get(parent_name)
        if parent_link is None:
            continue

        # Get positions in parent's local frame
        parent_pos = parent_link.worldpos()
        parent_rot_inv = parent_link.worldrot().T

        local_positions = []
        for child in children:
            child_world = child.worldpos()
            local = parent_rot_inv.dot(child_world - parent_pos)
            local_positions.append((child, local))

        # Check for symmetric pairs (opposite x or y coordinates)
        for i, (child1, pos1) in enumerate(local_positions):
            for child2, pos2 in local_positions[i + 1:]:
                # Check if symmetric in x or y (signs opposite, similar magnitude)
                x_symmetric = (abs(pos1[0] + pos2[0]) < 0.01
                               and abs(pos1[0]) > 0.005)
                y_symmetric = (abs(pos1[1] + pos2[1]) < 0.01
                               and abs(pos1[1]) > 0.005)
                # z should be similar
                z_similar = abs(pos1[2] - pos2[2]) < 0.02

                if (x_symmetric or y_symmetric) and z_similar:
                    # Found symmetric pair - use link origins midpoint
                    # (mesh-based tips don't work well for opposing fingers)
                    midpoint_local = (pos1 + pos2) / 2
                    midpoint_local = [
                        round(v, 6) if abs(v) > 1e-6 else 0.0
                        for v in midpoint_local]
                    return {
                        'parent_link': parent_name,
                        'pos': midpoint_local,
                        'rot': None,
                    }

    return None


def _find_best_end_coords_parent(G, link_map, tip_link_name, group_type):
    """Find the best parent link for end_coords.

    Searches both descendants and ancestors of tip_link for tool frames
    or hand links.

    Parameters
    ----------
    G : networkx.DiGraph
        The kinematic graph.
    link_map : dict
        Dictionary mapping link names to Link objects.
    tip_link_name : str
        Name of the tip link.
    group_type : str
        Type of the group (arm, head, gripper, etc.).

    Returns
    -------
    dict
        Dictionary containing:
        - 'parent_link': str, name of the best parent link
        - 'pos': list, position offset [x, y, z]
        - 'rot': list or None, rotation offset [rx, ry, rz] in degrees
    """
    # If tip link itself is a tool frame, use it directly
    if _is_tool_frame(tip_link_name):
        return {
            'parent_link': tip_link_name,
            'pos': [0.0, 0.0, 0.0],
            'rot': None,
        }

    # For grippers, just use the tip link
    if 'gripper' in group_type.lower():
        return {
            'parent_link': tip_link_name,
            'pos': [0.0, 0.0, 0.0],
            'rot': None,
        }

    max_depth = 5

    # First: Search DESCENDANTS for tool frame or hand link
    # (e.g., panda_link7 -> panda_hand)
    try:
        descendants = nx.descendants(G, tip_link_name)
    except nx.NetworkXError:
        descendants = set()

    # Check descendants up to max_depth
    for depth in range(1, max_depth + 1):
        for desc in descendants:
            try:
                path_len = nx.shortest_path_length(G, tip_link_name, desc)
                if path_len == depth:
                    if _is_tool_frame(desc):
                        return {
                            'parent_link': desc,
                            'pos': [0.0, 0.0, 0.0],
                            'rot': None,
                        }
                    if _is_hand_link(desc):
                        hand_link = link_map.get(desc)
                        if hand_link is not None:
                            return {
                                'parent_link': desc,
                                'pos': [0.0, 0.0, 0.0],
                                'rot': None,
                            }
            except nx.NetworkXNoPath:
                continue

    # Check for symmetric child links (e.g., gripper fingers)
    # Detect geometrically: same parent, symmetric positions
    result = _find_symmetric_gripper_midpoint(descendants, link_map)
    if result is not None:
        return result

    # Second: Search ANCESTORS for tool frame or hand link
    undirected = G.to_undirected()
    visited = set()
    current = tip_link_name

    for _ in range(max_depth):
        visited.add(current)
        neighbors = list(undirected.neighbors(current))
        parent_candidates = [n for n in neighbors if n not in visited
                             and n not in descendants]

        if not parent_candidates:
            break

        for parent in parent_candidates:
            if _is_tool_frame(parent):
                return {
                    'parent_link': parent,
                    'pos': [0.0, 0.0, 0.0],
                    'rot': None,
                }
            if _is_hand_link(parent):
                hand_link = link_map.get(parent)
                tip_link = link_map.get(tip_link_name)
                if hand_link is not None and tip_link is not None:
                    offset = _calculate_relative_offset(hand_link, tip_link)
                    return {
                        'parent_link': parent,
                        'pos': offset,
                        'rot': None,
                    }

        current = parent_candidates[0]

    # Fallback: use tip link directly
    return {
        'parent_link': tip_link_name,
        'pos': [0.0, 0.0, 0.0],
        'rot': None,
    }


def _calculate_relative_offset(parent_link, child_link):
    """Calculate position offset from parent to child in parent's frame.

    Parameters
    ----------
    parent_link : Link
        Parent link.
    child_link : Link
        Child link.

    Returns
    -------
    list
        Position offset [x, y, z] in parent's local frame.
    """
    parent_pos = parent_link.worldpos()
    parent_rot_inv = parent_link.worldrot().T
    child_pos = child_link.worldpos()

    offset = parent_rot_inv.dot(child_pos - parent_pos)
    # Round small values
    offset = [round(v, 6) if abs(v) > 1e-6 else 0.0 for v in offset]
    return offset


def _get_link_mesh(link):
    """Get mesh from link (collision or visual).

    Parameters
    ----------
    link : Link
        Robot link object.

    Returns
    -------
    trimesh.Trimesh or None
        Mesh object if available.
    """
    if link.collision_mesh is not None:
        return link.collision_mesh
    if link.visual_mesh is not None:
        from skrobot._lazy_imports import _lazy_trimesh
        trimesh = _lazy_trimesh()
        mesh = link.visual_mesh
        if isinstance(mesh, (list, tuple)):
            if len(mesh) == 0:
                return None
            return trimesh.util.concatenate(mesh)
        return mesh
    return None


def _get_fingertip_position(link):
    """Get fingertip position from mesh geometry.

    Parameters
    ----------
    link : Link
        Finger link.

    Returns
    -------
    np.ndarray
        Fingertip position in world coordinates.
    """
    mesh = _get_link_mesh(link)
    if mesh is not None:
        vertices = mesh.vertices
        extents = mesh.extents

        # Find primary axis (longest dimension)
        primary_axis = np.argmax(extents)

        # Get the centroid of vertices at the tip (max along primary axis)
        max_val = np.max(vertices[:, primary_axis])
        threshold = max_val - extents[primary_axis] * 0.1  # Top 10%
        tip_vertices = vertices[vertices[:, primary_axis] > threshold]
        tip_local = np.mean(tip_vertices, axis=0)

        # Transform to world
        return link.transform_vector(tip_local)
    else:
        return link.worldpos()


def _calculate_gripper_tcp_offset(G, link_map, parent_link_name):
    """Calculate TCP offset by finding gripper finger tips.

    From a parent link (e.g., panda_hand), find descendant finger links
    and calculate the midpoint of their tips as the TCP.

    Parameters
    ----------
    G : networkx.DiGraph
        The kinematic graph.
    link_map : dict
        Dictionary mapping link names to Link objects.
    parent_link_name : str
        Name of the parent link (e.g., hand, gripper base).

    Returns
    -------
    list or None
        Position offset [x, y, z] in parent's local frame, or None if
        no fingers found.
    """
    parent_link = link_map.get(parent_link_name)
    if parent_link is None:
        return None

    # Find descendant finger links
    try:
        descendants = nx.descendants(G, parent_link_name)
    except nx.NetworkXError:
        descendants = set()

    finger_links = []
    finger_patterns = ['finger', 'leftfinger', 'rightfinger', 'l_finger', 'r_finger']
    for desc_name in descendants:
        if any(p in desc_name.lower() for p in finger_patterns):
            desc_link = link_map.get(desc_name)
            if desc_link is not None:
                finger_links.append(desc_link)

    if len(finger_links) >= 2:
        # Calculate fingertip positions using mesh geometry
        fingertip_positions = []
        for finger in finger_links:
            tip_pos = _get_fingertip_position(finger)
            fingertip_positions.append(tip_pos)

        # Midpoint in world coordinates
        midpoint_world = np.mean(fingertip_positions, axis=0)

        # Convert to parent's local frame
        parent_pos = parent_link.worldpos()
        parent_rot_inv = parent_link.worldrot().T
        midpoint_local = parent_rot_inv.dot(midpoint_world - parent_pos)

        # Round small values
        midpoint_local = [round(v, 6) if abs(v) > 1e-6 else 0.0
                          for v in midpoint_local]

        return midpoint_local

    # No fingers found - try to estimate from mesh extent
    # This handles robots without gripper models (e.g., Nextage)
    return _estimate_tcp_from_mesh(parent_link)


def _estimate_tcp_from_mesh(link):
    """Estimate TCP offset from link mesh extent.

    For robots without gripper models, estimate the TCP position
    based on the mesh geometry of the end effector link.

    Parameters
    ----------
    link : Link
        The end effector link.

    Returns
    -------
    list or None
        Position offset [x, y, z] in link's local frame.
    """
    mesh = _get_link_mesh(link)
    if mesh is None:
        return None

    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    extents = mesh.extents

    # Find the primary axis (longest dimension)
    primary_axis = np.argmax(extents)

    # Check if mesh extends significantly more in one direction
    # (asymmetric = likely end effector direction)
    (bounds[0] + bounds[1]) / 2
    min_dist = abs(bounds[0][primary_axis])
    max_dist = abs(bounds[1][primary_axis])

    # If asymmetric (one side extends more), use that as TCP direction
    asymmetry_ratio = max(min_dist, max_dist) / (min(min_dist, max_dist) + 1e-6)

    if asymmetry_ratio > 2.0:  # Significant asymmetry
        # TCP is at the extended end
        if min_dist > max_dist:
            # Extends in negative direction
            tcp_pos = np.zeros(3)
            tcp_pos[primary_axis] = bounds[0][primary_axis]
        else:
            # Extends in positive direction
            tcp_pos = np.zeros(3)
            tcp_pos[primary_axis] = bounds[1][primary_axis]

        tcp_pos = [round(v, 6) if abs(v) > 1e-6 else 0.0 for v in tcp_pos]
        return tcp_pos

    return None


def _find_gripper_midpoint(link, link_map):
    """Find midpoint between gripper fingertips if applicable.

    Uses mesh geometry to find actual fingertip positions for
    more accurate TCP estimation.

    Parameters
    ----------
    link : Link
        The tip link (potentially a finger).
    link_map : dict
        Dictionary mapping link names to Link objects.

    Returns
    -------
    dict or None
        If gripper midpoint found, returns:
        {'parent_link': str, 'pos': list, 'rot': None}
        Otherwise None.
    """
    link_name = link.name.lower()

    # If this is already a tool frame, don't try to find gripper midpoint
    if _is_tool_frame(link.name):
        return None

    # Check if this looks like a finger link
    # Note: 'gripper' alone is not enough - need actual finger patterns
    finger_patterns = ['finger', 'leftfinger', 'rightfinger', 'l_finger', 'r_finger']
    if not any(p in link_name for p in finger_patterns):
        return None

    # Get parent link
    parent = link.parent_link
    if parent is None:
        return None

    # Check if parent is a tool frame
    if _is_tool_frame(parent.name):
        return {
            'parent_link': parent.name,
            'pos': [0.0, 0.0, 0.0],
            'rot': None,
        }

    # Find sibling finger links
    sibling_fingers = []
    if hasattr(parent, 'child_links'):
        for child in parent.child_links:
            if child is None or child is link:
                continue
            child_name = child.name.lower()
            if any(p in child_name for p in finger_patterns):
                sibling_fingers.append(child)

    if not sibling_fingers:
        return None

    # Calculate midpoint between fingertips using mesh geometry
    all_fingertip_positions = [_get_fingertip_position(link)]
    for sibling in sibling_fingers:
        all_fingertip_positions.append(_get_fingertip_position(sibling))

    midpoint_world = np.mean(all_fingertip_positions, axis=0)

    # Convert to parent's local frame
    parent_pos = parent.worldpos()
    parent_rot_inv = parent.worldrot().T
    midpoint_local = parent_rot_inv.dot(midpoint_world - parent_pos)
    midpoint_local = [round(v, 6) if abs(v) > 1e-6 else 0.0
                      for v in midpoint_local]

    return {
        'parent_link': parent.name,
        'pos': midpoint_local,
        'rot': None,
    }


# ============================================================================
# Graph building utilities
# ============================================================================

def _build_link_graph(robot):
    """Build a directed graph of links from robot model."""
    G = nx.DiGraph()
    link_map = {}

    for link in robot.link_list:
        link_map[link.name] = link
        G.add_node(link.name)

    for link in robot.link_list:
        if link.parent_link is not None:
            G.add_edge(link.parent_link.name, link.name)

    return G, link_map


def _find_base_link(G):
    """Find the base/root link of the robot."""
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    if len(roots) == 1:
        return roots[0]

    for root in roots:
        if 'base' in root.lower():
            return root

    return roots[0] if roots else list(G.nodes())[0]


def _detect_limb_type(link_names, tip_link=None, tip_y_coord=None,
                      movable_joint_count=0):
    """Detect what type of limb a set of links represents.

    Parameters
    ----------
    link_names : list of str
        List of link names in the chain.
    tip_link : str, optional
        Name of the tip link.
    tip_y_coord : float, optional
        Y coordinate of tip link relative to base (for left/right detection).
    movable_joint_count : int, optional
        Number of movable joints in this chain (for geometric detection).

    Returns
    -------
    str or None
        One of 'right_arm', 'left_arm', 'arm', 'right_leg', 'left_leg',
        'head', 'torso', 'gripper', or None.
    """
    right_arm_count = 0
    left_arm_count = 0
    right_leg_count = 0
    left_leg_count = 0
    head_count = 0
    torso_count = 0
    single_arm_count = 0
    gripper_count = 0

    right_arm_patterns = ['r_shoulder', 'r_upper_arm', 'r_elbow', 'r_forearm',
                          'r_wrist', 'rarm', 'right_arm', 'right_shoulder']
    left_arm_patterns = ['l_shoulder', 'l_upper_arm', 'l_elbow', 'l_forearm',
                         'l_wrist', 'larm', 'left_arm', 'left_shoulder']
    right_leg_patterns = ['r_hip', 'r_thigh', 'r_knee', 'r_ankle', 'r_foot',
                          'rleg', 'right_leg', 'right_hip']
    left_leg_patterns = ['l_hip', 'l_thigh', 'l_knee', 'l_ankle', 'l_foot',
                         'lleg', 'left_leg', 'left_hip']
    head_patterns = ['head', 'neck']
    torso_patterns = ['torso', 'chest', 'body', 'trunk']
    head_tip_patterns = ['range', 'camera', 'sensor', 'optical', 'rgb', 'depth']
    gripper_patterns = ['hand', 'gripper', 'finger', 'palm']

    for link_name in link_names:
        name_lower = link_name.lower()

        if any(p in name_lower for p in right_arm_patterns):
            right_arm_count += 1
        if any(p in name_lower for p in left_arm_patterns):
            left_arm_count += 1
        if any(p in name_lower for p in right_leg_patterns):
            right_leg_count += 1
        if any(p in name_lower for p in left_leg_patterns):
            left_leg_count += 1
        if any(p in name_lower for p in head_patterns):
            head_count += 1
        if any(p in name_lower for p in torso_patterns):
            torso_count += 1
        if any(p in name_lower for p in gripper_patterns):
            gripper_count += 1

    if tip_link:
        tip_lower = tip_link.lower()
        if any(p in tip_lower for p in head_tip_patterns):
            head_count += 2
        if any(p in tip_lower for p in gripper_patterns):
            gripper_count += 2

    # Geometric detection: long kinematic chain without left/right pattern
    # is likely a single arm (serial manipulator)
    if movable_joint_count >= 5:
        has_lr_pattern = (right_arm_count >= 2 or left_arm_count >= 2
                          or right_leg_count >= 2 or left_leg_count >= 2)
        if not has_lr_pattern:
            # This is likely a serial chain manipulator
            single_arm_count = movable_joint_count

    counts = {
        'right_arm': right_arm_count,
        'left_arm': left_arm_count,
        'right_leg': right_leg_count,
        'left_leg': left_leg_count,
        'head': head_count,
        'torso': torso_count,
        'arm': single_arm_count,
        'gripper': gripper_count,
    }

    # Use Y coordinate for geometric left/right detection
    y_threshold = 0.05
    if tip_y_coord is not None:
        has_arm_pattern = (right_arm_count >= 1 or left_arm_count >= 1
                           or single_arm_count >= 2)
        has_leg_pattern = right_leg_count >= 1 or left_leg_count >= 1

        strong_y_threshold = 0.1

        if has_arm_pattern or has_leg_pattern:
            if tip_y_coord > y_threshold:
                boost = 4 if tip_y_coord > strong_y_threshold else 2
                if has_arm_pattern:
                    counts['left_arm'] += boost
                if has_leg_pattern:
                    counts['left_leg'] += boost
            elif tip_y_coord < -y_threshold:
                boost = 4 if tip_y_coord < -strong_y_threshold else 2
                if has_arm_pattern:
                    counts['right_arm'] += boost
                if has_leg_pattern:
                    counts['right_leg'] += boost

    # Special rule: don't mix head and torso
    if head_count >= 1 and torso_count > 0:
        for link_name in link_names:
            name_lower = link_name.lower()
            if any(p in name_lower for p in head_patterns):
                counts['torso'] = 0
                break

    # Special rule: don't detect as single arm if we have left/right arm
    if right_arm_count >= 2 or left_arm_count >= 2:
        counts['arm'] = 0

    best_type = max(counts, key=counts.get)
    best_count = counts[best_type]

    thresholds = {
        'right_arm': 2, 'left_arm': 2, 'right_leg': 2, 'left_leg': 2,
        'head': 1, 'torso': 1, 'arm': 3, 'gripper': 2,
    }

    min_threshold = thresholds.get(best_type, 2)
    if best_count >= min_threshold:
        return best_type

    return None


def _find_limb_chains(G, base_link, link_map):
    """Find all limb chains from the base."""
    limbs = {}
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    arm_tip_priority = ['tool_frame', 'ee_link', 'end_effector', 'hand_link']

    preferred_leaves = []
    other_leaves = []
    for leaf in leaves:
        if any(p in leaf.lower() for p in arm_tip_priority):
            preferred_leaves.append(leaf)
        else:
            other_leaves.append(leaf)

    base_link_obj = link_map.get(base_link)
    base_pos = base_link_obj.worldpos() if base_link_obj else np.zeros(3)

    for leaf in preferred_leaves + other_leaves:
        try:
            path = nx.shortest_path(G, base_link, leaf)
        except nx.NetworkXNoPath:
            continue

        if len(path) < 2:
            continue

        tip_link_obj = link_map.get(leaf)
        tip_y_coord = None
        if tip_link_obj is not None:
            tip_pos = tip_link_obj.worldpos()
            tip_y_coord = tip_pos[1] - base_pos[1]

        # Count movable joints first for geometric detection
        movable_links = []
        for link_name in path:
            link = link_map.get(link_name)
            if link is None or link.joint is None:
                continue
            jtype = getattr(link.joint, 'joint_type', None)
            if jtype and jtype != 'fixed':
                movable_links.append(link_name)

        if not movable_links:
            continue

        limb_type = _detect_limb_type(
            path, tip_link=leaf, tip_y_coord=tip_y_coord,
            movable_joint_count=len(movable_links))
        if limb_type is None:
            continue

        if limb_type in ['torso', 'gripper']:
            continue

        if limb_type in limbs:
            existing_tip = limbs[limb_type]['tip_link']
            existing_is_preferred = any(p in existing_tip.lower() for p in arm_tip_priority)
            current_is_preferred = any(p in leaf.lower() for p in arm_tip_priority)
            if existing_is_preferred and not current_is_preferred:
                continue

        gripper_patterns = ['hand', 'gripper', 'finger', 'palm']
        actual_tip = leaf
        leaf_is_preferred = any(p in leaf.lower() for p in arm_tip_priority)

        if limb_type in ['right_arm', 'left_arm']:
            arm_movable = []
            for link_name in movable_links:
                name_lower = link_name.lower()
                if any(p in name_lower for p in ['torso', 'chest', 'body']):
                    continue
                if any(p in name_lower for p in gripper_patterns):
                    continue
                arm_movable.append(link_name)
            if arm_movable:
                movable_links = arm_movable
                if not leaf_is_preferred:
                    actual_tip = movable_links[-1]
        elif limb_type == 'arm':
            arm_movable = []
            for link_name in movable_links:
                name_lower = link_name.lower()
                if any(p in name_lower for p in ['torso', 'chest', 'body']):
                    continue
                if any(p in name_lower for p in gripper_patterns):
                    continue
                arm_movable.append(link_name)
            if arm_movable:
                movable_links = arm_movable
                # For single arm, use last movable link as tip (not gripper)
                if not leaf_is_preferred:
                    actual_tip = movable_links[-1]

        should_update = False
        if limb_type not in limbs:
            should_update = True
        else:
            existing_tip = limbs[limb_type]['tip_link']
            existing_is_preferred = any(p in existing_tip.lower() for p in arm_tip_priority)
            current_is_preferred = any(p in leaf.lower() for p in arm_tip_priority)
            if current_is_preferred and not existing_is_preferred:
                should_update = True
            elif current_is_preferred == existing_is_preferred:
                if len(movable_links) > len(limbs[limb_type]['links']):
                    should_update = True

        if should_update:
            limbs[limb_type] = {
                'links': movable_links,
                'root_link': movable_links[0] if movable_links else path[0],
                'tip_link': actual_tip,
            }

    return limbs


def _find_torso_chain(G, base_link, arm_roots, link_map):
    """Find torso chain from base to arm roots."""
    if not arm_roots:
        return None

    torso_links = set()
    for arm_root in arm_roots:
        try:
            path = nx.shortest_path(G, base_link, arm_root)
            torso_links.update(path[:-1])
        except nx.NetworkXNoPath:
            continue

    if not torso_links:
        return None

    ordered_torso = []
    try:
        for link in nx.shortest_path(G, base_link, arm_roots[0]):
            if link in torso_links:
                ordered_torso.append(link)
    except nx.NetworkXNoPath:
        ordered_torso = list(torso_links)

    movable_links = []
    for link_name in ordered_torso:
        link = link_map.get(link_name)
        if link is None or link.joint is None:
            continue
        jtype = getattr(link.joint, 'joint_type', None)
        if jtype and jtype != 'fixed':
            movable_links.append(link_name)

    if not movable_links:
        return None

    return {
        'links': movable_links,
        'root_link': ordered_torso[0] if ordered_torso else base_link,
        'tip_link': ordered_torso[-1] if ordered_torso else base_link,
    }


def _find_gripper_chains(G, arm_tip, link_map):
    """Find gripper chain starting from arm tip."""
    try:
        descendants = nx.descendants(G, arm_tip)
    except nx.NetworkXError:
        return None

    gripper_links = []
    for desc in descendants:
        if any(p in desc.lower() for p in ['gripper', 'finger', 'hand']):
            gripper_links.append(desc)

    if not gripper_links:
        return None

    max_path = []
    for gl in gripper_links:
        try:
            path = nx.shortest_path(G, arm_tip, gl)
            if len(path) > len(max_path):
                max_path = path
        except nx.NetworkXNoPath:
            continue

    if len(max_path) < 2:
        return None

    movable_links = []
    for link_name in max_path:
        link = link_map.get(link_name)
        if link is None or link.joint is None:
            continue
        jtype = getattr(link.joint, 'joint_type', None)
        if jtype and jtype != 'fixed':
            movable_links.append(link_name)

    if not movable_links:
        return None

    return {
        'links': movable_links,
        'root_link': max_path[0],
        'tip_link': max_path[-1],
    }


def generate_groups_from_geometry(robot):
    """Generate robot groups from URDF geometry without LLM.

    Parameters
    ----------
    robot : RobotModel
        Robot model instance.

    Returns
    -------
    dict
        Dictionary of group definitions.
    dict
        Dictionary of end effector links.
    dict
        Dictionary of end_coords info (parent_link, pos, rot).
    str
        Robot name.
    """
    G, link_map = _build_link_graph(robot)
    base_link = _find_base_link(G)

    limbs = _find_limb_chains(G, base_link, link_map)

    groups = {}
    end_effectors = {}

    # Process dual arms
    arm_roots = []
    for limb_type in ['right_arm', 'left_arm']:
        if limb_type in limbs:
            chain = limbs[limb_type]
            groups[limb_type] = chain
            end_effectors[limb_type] = chain['tip_link']
            arm_roots.append(chain['root_link'])

    # Process single arm
    if 'arm' in limbs and 'right_arm' not in limbs and 'left_arm' not in limbs:
        chain = limbs['arm']
        groups['arm'] = chain
        end_effectors['arm'] = chain['tip_link']
        arm_roots.append(chain['root_link'])

    # Process legs
    for limb_type in ['right_leg', 'left_leg']:
        if limb_type in limbs:
            chain = limbs[limb_type]
            groups[limb_type] = chain
            end_effectors[limb_type] = chain['tip_link']

    # Process head
    if 'head' in limbs:
        chain = limbs['head']
        groups['head'] = chain
        end_effectors['head'] = chain['tip_link']

    # Find torso
    torso_chain = _find_torso_chain(G, base_link, arm_roots, link_map)
    if torso_chain:
        groups['torso'] = torso_chain
        end_effectors['torso'] = torso_chain['tip_link']

    # Create arm+torso combinations
    for limb_type in ['right_arm', 'left_arm']:
        if limb_type in groups and torso_chain:
            combined_links = torso_chain['links'] + groups[limb_type]['links']
            groups[f'{limb_type}_torso'] = {
                'links': combined_links,
                'root_link': torso_chain['root_link'],
                'tip_link': groups[limb_type]['tip_link'],
            }
            end_effectors[f'{limb_type}_torso'] = groups[limb_type]['tip_link']

    if 'arm' in groups and torso_chain:
        combined_links = torso_chain['links'] + groups['arm']['links']
        groups['arm_torso'] = {
            'links': combined_links,
            'root_link': torso_chain['root_link'],
            'tip_link': groups['arm']['tip_link'],
        }
        end_effectors['arm_torso'] = groups['arm']['tip_link']

    # Find grippers
    gripper_prefix_map = {'right_arm': 'right', 'left_arm': 'left'}
    for limb_type in ['right_arm', 'left_arm']:
        if limb_type in groups:
            arm_links = groups[limb_type]['links']
            if arm_links:
                last_movable = arm_links[-1]
                gripper_chain = _find_gripper_chains(G, last_movable, link_map)
                if gripper_chain:
                    gripper_type = f'{gripper_prefix_map[limb_type]}_gripper'
                    groups[gripper_type] = gripper_chain
                    end_effectors[gripper_type] = gripper_chain['tip_link']

    # Calculate end_coords info for each group
    end_coords_info = {}
    for group_name, group_data in groups.items():
        if group_data is None:
            continue

        tip_link_name = group_data.get('tip_link')
        if tip_link_name is None:
            continue

        tip_link = link_map.get(tip_link_name)
        if tip_link is None:
            continue

        # Find best parent for end_coords first
        ec_info = _find_best_end_coords_parent(
            G, link_map, tip_link_name, group_name)

        # For arm groups, try to calculate gripper TCP offset
        # Skip if offset already set (e.g., by symmetric gripper detection)
        has_offset = any(abs(v) > 1e-6 for v in ec_info.get('pos', [0, 0, 0]))
        if ('arm' in group_name.lower() and 'torso' not in group_name.lower()
                and not has_offset):
            parent_link_name = ec_info['parent_link']
            parent_link = link_map.get(parent_link_name)
            if parent_link is not None:
                # Try to find gripper midpoint from the end_coords parent
                tcp_offset = _calculate_gripper_tcp_offset(
                    G, link_map, parent_link_name)
                if tcp_offset is not None:
                    ec_info['pos'] = tcp_offset

        end_coords_info[group_name] = ec_info

    robot_name = getattr(robot, 'name', None) or 'robot'

    return groups, end_effectors, end_coords_info, robot_name


def _sanitize_class_name(name):
    """Convert a string to a valid Python class name."""
    name = os.path.splitext(name)[0]
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    name = re.sub(r'^[_0-9]+', '', name)
    parts = name.split('_')
    name = ''.join(part.capitalize() for part in parts if part)
    if not name or not name[0].isalpha():
        name = 'Robot' + name
    return name


def _sanitize_attr_name(name):
    """Convert a link/joint name to a valid Python attribute name."""
    return name.replace('-', '_').replace('.', '_')


def generate_robot_class_from_geometry(robot, output_path=None,
                                        class_name=None,
                                        urdf_path_function=None,
                                        urdf_path=None):
    """Generate a Python class from robot geometry without LLM.

    Parameters
    ----------
    robot : RobotModel
        Robot model instance.
    output_path : str, optional
        Path to save the generated class.
    class_name : str, optional
        Class name for the generated class.
    urdf_path_function : str, optional
        Function name to get URDF path.
    urdf_path : str, optional
        Explicit URDF path to use. If not provided, will try to get from
        robot.default_urdf_path or robot.urdf_path.

    Returns
    -------
    str
        The generated Python code.

    Example
    -------
    >>> from skrobot.models import Panda
    >>> from skrobot.urdf.robot_class_generator import generate_robot_class_from_geometry
    >>> robot = Panda()
    >>> code = generate_robot_class_from_geometry(robot, output_path='/tmp/MyPanda.py')
    """
    result = generate_groups_from_geometry(robot)
    groups, end_effectors, end_coords_info, robot_name = result

    if urdf_path is None:
        urdf_path = getattr(robot, 'default_urdf_path', None)
        if urdf_path is None:
            urdf_path = getattr(robot, 'urdf_path', None)
        if callable(urdf_path):
            try:
                urdf_path = urdf_path()
            except NotImplementedError:
                urdf_path = None

    # Convert to ROS package:// path if inside a ROS package
    urdf_path = _convert_to_ros_package_path(urdf_path)

    group_def = GroupDefinition(
        robot_name=robot_name,
        urdf_path=urdf_path,
        groups=groups,
        end_effectors=end_effectors,
        end_coords_info=end_coords_info,
    )

    if class_name is None:
        class_name = _sanitize_class_name(group_def.robot_name)

    # Build imports
    imports = [
        "from cached_property import cached_property",
        "",
        "from skrobot.coordinates import CascadedCoords",
        "from skrobot.model import RobotModel",
        "from skrobot.models.urdf import RobotModelFromURDF",
    ]
    if urdf_path_function:
        imports.append(f"from skrobot.data import {urdf_path_function}")
    imports_code = "\n".join(imports)

    # Build __init__ method
    init_lines = []
    init_lines.append("    def __init__(self):")
    init_lines.append("        super().__init__()")
    init_lines.append("")

    for group_name, group_data in group_def.groups.items():
        if group_data is None:
            continue

        # Get end_coords info (parent link, position, rotation)
        ec_info = group_def.end_coords_info.get(group_name)
        if ec_info is None:
            # Fallback to tip_link with no offset
            tip_link = group_data.get('tip_link')
            if tip_link is None:
                continue
            ec_info = {'parent_link': tip_link, 'pos': [0.0, 0.0, 0.0], 'rot': None}

        parent_link = ec_info['parent_link']
        pos = ec_info.get('pos', [0.0, 0.0, 0.0])
        ec_info.get('rot')

        parent_attr = _sanitize_attr_name(parent_link)
        # Use underscore prefix to avoid conflict with base class properties
        end_coords_attr = f"_{group_name}_end_coords"
        end_coords_name = f"{group_name}_end_coords"

        # Check if we need pos argument
        has_offset = any(abs(v) > 1e-6 for v in pos)

        if has_offset:
            pos_str = f"[{pos[0]}, {pos[1]}, {pos[2]}]"
            init_lines.append(f"        self.{end_coords_attr} = CascadedCoords(")
            init_lines.append(f"            parent=self.{parent_attr},")
            init_lines.append(f"            pos={pos_str},")
            init_lines.append(f"            name='{end_coords_name}')")
        else:
            init_lines.append(f"        self.{end_coords_attr} = CascadedCoords(")
            init_lines.append(f"            parent=self.{parent_attr},")
            init_lines.append(f"            name='{end_coords_name}')")

    init_code = "\n".join(init_lines)

    # Build default_urdf_path property
    if urdf_path_function:
        urdf_property = f'''    @cached_property
    def default_urdf_path(self):
        return {urdf_path_function}()'''
    else:
        urdf_path_str = group_def.urdf_path or ''
        urdf_property = f'''    @cached_property
    def default_urdf_path(self):
        return "{urdf_path_str}"'''

    # Build group properties
    properties = []
    for group_name, group_data in group_def.groups.items():
        if group_data is None:
            continue

        links = group_data.get('links', [])
        if not links:
            continue

        link_attrs = [f"self.{_sanitize_attr_name(l)}" for l in links]
        links_str = ",\n            ".join(link_attrs)

        # Use underscore prefix for internal attribute
        end_coords_attr = f"_{group_name}_end_coords"

        prop_code = f'''    @cached_property
    def {group_name}(self):
        """{group_name.replace('_', ' ').title()} kinematic chain."""
        links = [
            {links_str},
        ]
        joints = [link.joint for link in links]
        r = RobotModel(link_list=links, joint_list=joints)
        r.end_coords = self.{end_coords_attr}
        return r'''

        properties.append(prop_code)

    # Add end_coords properties
    for group_name, group_data in group_def.groups.items():
        if group_data is None:
            continue
        end_coords_attr = f"_{group_name}_end_coords"
        end_coords_prop = f'''    @property
    def {group_name}_end_coords(self):
        """End coordinates for {group_name.replace('_', ' ')}."""
        return self.{end_coords_attr}'''
        properties.append(end_coords_prop)

    properties_code = "\n\n".join(properties)

    # Build backward compatibility aliases
    alias_map = {
        'right_arm': 'rarm',
        'left_arm': 'larm',
        'right_leg': 'rleg',
        'left_leg': 'lleg',
        'right_arm_torso': 'rarm_torso',
        'left_arm_torso': 'larm_torso',
        'right_gripper': 'rgripper',
        'left_gripper': 'lgripper',
    }

    aliases = []
    for group_name in group_def.groups.keys():
        if group_name in alias_map:
            old_name = alias_map[group_name]
            aliases.append(f"    {old_name} = {group_name}  # Backward compatibility")
            aliases.append(
                f"    {old_name}_end_coords = property("
                f"lambda self: self.{group_name}_end_coords)"
            )

    aliases_code = "\n".join(aliases) if aliases else ""

    # Viewer code
    group_names = [name for name, data in group_def.groups.items() if data is not None]
    axis_lines = []
    for group_name in group_names:
        axis_lines.append(
            f"    axis_{group_name} = Axis.from_coords("
            f"robot.{group_name}_end_coords, axis_radius=0.01, axis_length=0.1)"
        )
        axis_lines.append(f"    viewer.add(axis_{group_name})")
    axis_code = "\n".join(axis_lines)

    viewer_code = f'''


if __name__ == '__main__':
    from skrobot.model import Axis
    from skrobot.viewers import PyrenderViewer

    robot = {class_name}()

    viewer = PyrenderViewer()
    viewer.add(robot)

{axis_code}

    viewer.show()

    while viewer.is_active:
        viewer.redraw()
'''

    # Aliases section
    aliases_section = ""
    if aliases_code:
        aliases_section = f"\n\n    # Backward compatibility aliases\n{aliases_code}"

    # Assemble full class
    code = f'''"""Auto-generated robot class for {group_def.robot_name}."""

{imports_code}


class {class_name}(RobotModelFromURDF):
    """{class_name} Robot Model."""

{init_code}

{urdf_property}

{properties_code}{aliases_section}
{viewer_code}'''

    if output_path:
        with open(output_path, 'w') as f:
            f.write(code)

    return code

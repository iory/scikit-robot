"""
URDF Parser for configuration page.

Extracts joint and link information from URDF content.
"""


from __future__ import annotations

from typing import Any
import xml.etree.ElementTree as ET


def parse_urdf_content(urdf_content: str) -> dict[str, Any]:
    """
    Parse URDF content and extract joint/link information.

    Parameters
    ----------
    urdf_content : str
        The URDF XML content as a string.

    Returns
    -------
    dict
        Dictionary containing:
        - joints: List of joint information
        - links: List of link information
        - root_link: Name of the root link

    Raises
    ------
    ValueError
        If the URDF content is invalid.
    """
    try:
        root = ET.fromstring(urdf_content)
    except ET.ParseError as e:
        raise ValueError(f"Invalid URDF XML: {e}") from e

    if root.tag != "robot":
        raise ValueError(f"Expected <robot> root element, got <{root.tag}>")

    # Extract joints
    joints = []
    for joint_elem in root.findall("joint"):
        joint_info = _parse_joint(joint_elem)
        joints.append(joint_info)

    # Extract links
    links = []
    for link_elem in root.findall("link"):
        link_info = _parse_link(link_elem)
        links.append(link_info)

    # Find root link (link that is not a child of any joint)
    child_links = {j["childLink"] for j in joints}
    link_names = {link["name"] for link in links}
    root_links = link_names - child_links

    root_link = None
    if len(root_links) == 1:
        root_link = root_links.pop()
    elif len(root_links) > 1:
        # If multiple root candidates, pick the one that's a parent
        parent_links = {j["parentLink"] for j in joints}
        candidates = root_links & parent_links
        if candidates:
            root_link = candidates.pop()

    return {
        "joints": joints,
        "links": links,
        "root_link": root_link,
    }


def _parse_joint(joint_elem: ET.Element) -> dict[str, Any]:
    """Parse a single joint element."""
    joint_name = joint_elem.get("name", "unnamed_joint")
    joint_type = joint_elem.get("type", "fixed")

    # Parent and child links
    parent_elem = joint_elem.find("parent")
    child_elem = joint_elem.find("child")
    parent_link = parent_elem.get("link", "") if parent_elem is not None else ""
    child_link = child_elem.get("link", "") if child_elem is not None else ""

    # Axis
    axis_elem = joint_elem.find("axis")
    if axis_elem is not None:
        xyz = axis_elem.get("xyz", "0 0 1")
        axis = [float(x) for x in xyz.split()]
    else:
        axis = [0.0, 0.0, 1.0]

    # Limits
    limit_elem = joint_elem.find("limit")
    lower_limit = 0.0
    upper_limit = 0.0
    velocity_limit = 0.0
    effort_limit = 0.0

    if limit_elem is not None:
        lower_limit = float(limit_elem.get("lower", "0"))
        upper_limit = float(limit_elem.get("upper", "0"))
        velocity_limit = float(limit_elem.get("velocity", "0"))
        effort_limit = float(limit_elem.get("effort", "0"))

    # Mimic: a passive joint whose motion follows a driving joint.
    mimic_elem = joint_elem.find("mimic")
    is_mimic = mimic_elem is not None
    mimic_joint = mimic_elem.get("joint", "") if mimic_elem is not None else None

    return {
        "name": joint_name,
        "type": joint_type,
        "parentLink": parent_link,
        "childLink": child_link,
        "axis": axis,
        "lowerLimit": lower_limit,
        "upperLimit": upper_limit,
        "velocityLimit": velocity_limit,
        "effortLimit": effort_limit,
        "isMimic": is_mimic,
        "mimicJoint": mimic_joint,
    }


def _parse_link(link_elem: ET.Element) -> dict[str, Any]:
    """Parse a single link element."""
    link_name = link_elem.get("name", "unnamed_link")

    has_visual = link_elem.find("visual") is not None
    has_collision = link_elem.find("collision") is not None
    has_inertial = link_elem.find("inertial") is not None

    return {
        "name": link_name,
        "hasVisual": has_visual,
        "hasCollision": has_collision,
        "hasInertial": has_inertial,
    }

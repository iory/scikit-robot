#!/usr/bin/env python

import os
import subprocess
import tempfile

from lxml import etree


def add_prefix_to_name(name):
    """Add prefix to a name if it doesn't already have one.

    Parameters
    ----------
    name : str
        The name to potentially prefix

    Returns
    -------
    str
        Name with "${prefix}" prefix if not already present
    """
    # If name is already a variable (e.g., from a property), don't prefix it.
    if name.startswith("$"):
        return name
    return "${prefix}" + name if not name.startswith("${prefix}") else name


def indent_element(elem, level=1, indent_str="  "):
    """Apply proper indentation to XML elements.

    Parameters
    ----------
    elem : lxml.etree.Element
        XML element to indent
    level : int, optional
        Current indentation level, by default 1
    indent_str : str, optional
        String to use for each indentation level, by default "  "
    """
    i = "\n" + level * indent_str
    if len(elem):
        # Set text content after opening tag (indentation for first child)
        if not elem.text or not elem.text.strip():
            elem.text = i + indent_str

        # Recursively process all child elements
        for child in elem:
            indent_element(child, level + 1, indent_str)

        # Set tail of last child to align parent's closing tag properly
        # This ensures closing tags like </joint> appear at correct position
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i

    # Set text after closing tag (indentation for next sibling)
    if not elem.tail or not elem.tail.strip():
        elem.tail = i


def find_root_link(input_path):
    """Find the root link in a URDF file.

    The root link is the link that appears as a parent but never as a child
    in any joint. For single-link URDFs, returns the only link.

    Parameters
    ----------
    input_path : str
        Path to the URDF file

    Returns
    -------
    str
        Name of the root link

    Raises
    ------
    ValueError
        If root link cannot be determined
    """
    tree = etree.parse(input_path)
    root = tree.getroot()
    parent_links = set()
    child_links = set()

    # Collect all parent and child links from joints
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None:
            parent_links.add(parent.attrib["link"])
        if child is not None:
            child_links.add(child.attrib["link"])

    # Root links are parents that are never children
    root_candidates = parent_links - child_links
    if root_candidates:
        return next(iter(root_candidates))
    else:
        # Handle single-link URDF case
        root_links = root.findall("link")
        if len(root_links) == 1:
            return root_links[0].attrib["name"]
        raise ValueError("Could not determine root link. Check that the URDF contains valid joint definitions.")


def transform_urdf_to_macro(input_path, connector_link, no_prefix):
    """Transform a URDF file into a Xacro macro with connector joint.

    This function converts a URDF file into a reusable Xacro macro that can be
    attached to other robots via a connector joint. Optionally adds prefixes
    to avoid naming conflicts.

    Parameters
    ----------
    input_path : str
        Path to input URDF or Xacro file
    connector_link : str
        Name of link to connect to parent robot
    no_prefix : bool
        If True, don't add prefixes to element names

    Returns
    -------
    tuple[lxml.etree.Element, str]
        Tuple of (xacro_root_element, robot_name)
    """
    XACRO_NS = "http://ros.org/wiki/xacro"
    NSMAP = {"xacro": XACRO_NS}

    # Process Xacro files by converting to URDF first
    if input_path.endswith(".xacro"):
        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as tmp:
            tmp_path = tmp.name
            subprocess.run(["xacro", input_path, "-o", tmp_path], check=True)
            tree = etree.parse(tmp_path)
        os.remove(tmp_path)
    else:
        tree = etree.parse(input_path)
    urdf_root = tree.getroot()
    robot_name = urdf_root.attrib.get("name", "robot_macro")

    # Create Xacro root element
    xacro_root = etree.Element("robot", nsmap=NSMAP)
    xacro_root.set("name", robot_name)

    # Define macro parameters with default values
    macro_params = [
        "parent_link:=world",
        "xyz:='0 0 0'",
        "rpy:='0 0 0'"
    ]
    if not no_prefix:
        # Provide a default empty string for the prefix.
        macro_params.insert(0, "prefix:=''")

    # Create macro element
    macro = etree.Element("{}macro".format("{" + XACRO_NS + "}"))
    macro.set("name", robot_name)
    macro.set("params", " ".join(macro_params))
    xacro_root.append(macro)

    # Create connector joint to attach this macro to parent
    connector_joint = etree.Element("joint")
    connector_link_with_prefix = add_prefix_to_name(connector_link) if not no_prefix else connector_link
    connector_joint_name = f"${{parent_link}}_to_{connector_link_with_prefix}_joint"
    connector_joint.set("name", connector_joint_name)
    connector_joint.set("type", "fixed")

    parent_elem = etree.SubElement(connector_joint, "parent")
    parent_elem.set("link", "${parent_link}")

    child_elem = etree.SubElement(connector_joint, "child")
    child_elem.set("link", connector_link_with_prefix)

    # Replace the insert_block with a standard origin tag using parameters
    origin_elem = etree.SubElement(connector_joint, "origin")
    origin_elem.set("xyz", "${xyz}")
    origin_elem.set("rpy", "${rpy}")

    macro.append(connector_joint)

    # Copy all elements from original URDF, adding prefixes if needed
    for elem in urdf_root:
        if not no_prefix:
            # Add prefix to element names
            if "name" in elem.attrib:
                elem.attrib["name"] = add_prefix_to_name(elem.attrib["name"])
            # Update joint references to use prefixed names
            if elem.tag in ["joint"]:
                for sub in elem.findall("parent"):
                    sub.attrib["link"] = add_prefix_to_name(sub.attrib["link"])
                for sub in elem.findall("child"):
                    sub.attrib["link"] = add_prefix_to_name(sub.attrib["link"])
                for sub in elem.findall("mimic"):
                    if "joint" in sub.attrib:
                        sub.attrib["joint"] = add_prefix_to_name(sub.attrib["joint"])
        macro.append(elem)

    indent_element(macro, level=1)
    indent_element(xacro_root, level=0)
    xacro_root.tail = None  # Ensure no tail text at the root level

    return xacro_root, robot_name

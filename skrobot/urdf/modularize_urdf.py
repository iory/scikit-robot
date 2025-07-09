#!/usr/bin/env python

import os
import subprocess
import tempfile

from lxml import etree


def add_prefix_to_name(name):
    return "${prefix}" + name if not name.startswith("${prefix}") else name


def indent_element(elem, level=1, indent_str="    "):
    i = "\n" + level * indent_str
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent_str
        for child in elem:
            indent_element(child, level + 1, indent_str)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if not elem.tail or not elem.tail.strip():
            elem.tail = i


def find_root_link(input_path):
    tree = etree.parse(input_path)
    root = tree.getroot()
    parent_links = set()
    child_links = set()

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None:
            parent_links.add(parent.attrib["link"])
        if child is not None:
            child_links.add(child.attrib["link"])

    root_candidates = parent_links - child_links
    if root_candidates:
        return next(iter(root_candidates))  # root link name as string
    else:
        raise ValueError("Could not determine root link. Check that the URDF contains valid joint definitions.")


def transform_urdf_to_macro(input_path, connector_link, no_prefix):
    XACRO_NS = "http://ros.org/wiki/xacro"
    NSMAP = {"xacro": XACRO_NS}

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

    xacro_root = etree.Element("robot", nsmap=NSMAP)
    xacro_root.set("name", robot_name)

    macro_params = ["parent_link", "*origin"]
    if not no_prefix:
        macro_params.insert(0, "prefix")

    macro = etree.Element("{}macro".format("{" + XACRO_NS + "}"))
    macro.set("name", robot_name)
    macro.set("params", " ".join(macro_params))
    xacro_root.append(macro)

    connector_joint = etree.Element("joint")
    connector_joint_name = "{}{}_to_${{parent_link}}_joint".format('${prefix}' if not no_prefix else '', connector_link)
    connector_joint.set("name", connector_joint_name)
    connector_joint.set("type", "fixed")

    parent_elem = etree.SubElement(connector_joint, "parent")
    parent_elem.set("link", "${parent_link}")

    child_elem = etree.SubElement(connector_joint, "child")
    child_elem.set("link", add_prefix_to_name(connector_link) if not no_prefix else connector_link)

    origin_block = etree.SubElement(connector_joint, "{}insert_block".format("{" + XACRO_NS + "}"))
    origin_block.set("name", "origin")

    macro.append(connector_joint)

    for elem in urdf_root:
        if not no_prefix:
            if "name" in elem.attrib:
                elem.attrib["name"] = add_prefix_to_name(elem.attrib["name"])
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

    return xacro_root, robot_name

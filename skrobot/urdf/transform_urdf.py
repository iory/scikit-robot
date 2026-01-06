import math
import os
import xml.etree.ElementTree as ET

from skrobot.urdf.modularize_urdf import find_root_link


def transform_urdf_with_world_link(input_file, output_file,
                                   x=0.0, y=0.0, z=0.0,
                                   roll=0.0, pitch=0.0, yaw=0.0,
                                   world_link_name="world"):
    """Add a transformed world link to a URDF file.

    Parameters
    ----------
    input_file : str
        Path to the input URDF file
    output_file : str
        Path for the output URDF file
    x : float, optional
        Translation in X (meters). Default: 0.0
    y : float, optional
        Translation in Y (meters). Default: 0.0
    z : float, optional
        Translation in Z (meters). Default: 0.0
    roll : float, optional
        Rotation around X-axis (degrees). Default: 0.0
    pitch : float, optional
        Rotation around Y-axis (degrees). Default: 0.0
    yaw : float, optional
        Rotation around Z-axis (degrees). Default: 0.0
    world_link_name : str, optional
        Name for the new world link. Default: 'world'

    Raises
    ------
    FileNotFoundError
        If the input file does not exist
    ValueError
        If the world link name already exists in the URDF or if root link cannot be determined
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"URDF file not found: {input_file}")

    # Register xacro namespace if present
    ET.register_namespace('xacro', "http://ros.org/wiki/xacro")

    tree = ET.parse(input_file)
    root = tree.getroot()

    # Find the original root link
    original_root_link = find_root_link(input_file)

    # Check if the new world link name already exists
    if root.find(f"./link[@name='{world_link_name}']") is not None:
        raise ValueError(
            f"Link '{world_link_name}' already exists in the URDF. "
            f"Choose a different world link name.")

    # Create the new world link element
    world_link = ET.Element('link', name=world_link_name)

    # Create the new fixed joint to connect world to the original root
    joint_name = f"{world_link_name}_to_{original_root_link}"
    new_joint = ET.Element('joint', name=joint_name, type='fixed')

    # Set parent (world) and child (original root)
    ET.SubElement(new_joint, 'parent', link=world_link_name)
    ET.SubElement(new_joint, 'child', link=original_root_link)

    # Create the origin element with the specified transform
    # Convert degrees to radians for RPY
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)

    xyz_str = f"{x} {y} {z}"
    rpy_str = f"{roll_rad} {pitch_rad} {yaw_rad}"

    ET.SubElement(new_joint, 'origin', xyz=xyz_str, rpy=rpy_str)

    # Insert the new elements into the XML tree (at the beginning)
    root.insert(0, new_joint)
    root.insert(0, world_link)

    # Try to use pretty printing if available (Python 3.9+)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    tree.write(output_file, encoding='utf-8', xml_declaration=True)

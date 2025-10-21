#!/usr/bin/env python

import networkx as nx

from skrobot.utils.urdf import URDF


def extract_sub_urdf(
        input_urdf_path,
        root_link_name,
        to_link_name=None,
        output_urdf_path=None,
        keep_fixed_joints=True):
    """Extract a sub-URDF starting from a specified root link.

    This function extracts a portion of a URDF file, creating a new URDF
    that starts from the specified root link and includes all descendant
    links. Optionally, you can specify a target link to extract only the
    path from root to that target link.

    Parameters
    ----------
    input_urdf_path : str
        Path to the input URDF file
    root_link_name : str
        Name of the link to use as the new root link in the sub-URDF
    to_link_name : str, optional
        If specified, extract only the path from root_link to this link.
        If None, extract all descendants of root_link. Default is None.
    output_urdf_path : str, optional
        Path to save the output URDF file.
        If None, the URDF object is returned but not saved.
    keep_fixed_joints : bool, optional
        Whether to keep fixed joints in the sub-URDF. Default is True.

    Returns
    -------
    URDF
        The extracted sub-URDF object

    Raises
    ------
    ValueError
        If root_link_name or to_link_name is not found in the URDF,
        or if to_link_name is not a descendant of root_link_name

    Examples
    --------
    >>> from skrobot.data import pr2_urdfpath
    >>> from skrobot.urdf import extract_sub_urdf
    >>> # Extract right arm only
    >>> sub_urdf = extract_sub_urdf(
    ...     pr2_urdfpath(),
    ...     root_link_name='r_shoulder_pan_link',
    ...     output_urdf_path='pr2_right_arm.urdf'
    ... )
    >>> # Extract from torso to head
    >>> head_urdf = extract_sub_urdf(
    ...     pr2_urdfpath(),
    ...     root_link_name='torso_lift_link',
    ...     to_link_name='head_tilt_link',
    ...     output_urdf_path='pr2_head.urdf'
    ... )
    """
    # Load the original URDF
    urdf = URDF.load(input_urdf_path)

    # Validate that root_link_name exists
    if root_link_name not in urdf.link_map:
        raise ValueError(
            f"Root link '{root_link_name}' not found in URDF. "
            f"Available links: {list(urdf.link_map.keys())}"
        )

    root_link = urdf.link_map[root_link_name]

    # Determine which links to include
    if to_link_name is not None:
        # Extract path from root_link to to_link
        if to_link_name not in urdf.link_map:
            raise ValueError(
                f"Target link '{to_link_name}' not found in URDF. "
                f"Available links: {list(urdf.link_map.keys())}"
            )

        to_link = urdf.link_map[to_link_name]

        # Get all paths from to_link to root_link (using the child->parent graph)
        # The graph _G has edges from child to parent
        try:
            # Find path from to_link to root_link (going up the tree)
            path_to_root = nx.shortest_path(urdf._G, to_link, root_link)
            # The path includes links from to_link to root_link
            links_to_include = set(path_to_root)
        except nx.NetworkXNoPath:
            raise ValueError(
                f"No path found from '{to_link_name}' to '{root_link_name}'. "
                f"'{to_link_name}' is not a descendant of '{root_link_name}'."
            )
    else:
        # Extract all descendants of root_link
        # Create reversed graph (parent -> child)
        reversed_graph = urdf._G.reverse()

        # Get all descendants in the reversed graph
        descendants = nx.descendants(reversed_graph, root_link)

        # Include root_link and all its descendants
        links_to_include = {root_link} | descendants

    # Filter joints that connect the included links
    joints_to_include = []
    for joint in urdf.joints:
        parent_link = urdf.link_map.get(joint.parent)
        child_link = urdf.link_map.get(joint.child)

        # Include joint if both parent and child are in the link set
        if parent_link in links_to_include and child_link in links_to_include:
            # Optionally filter out fixed joints
            if not keep_fixed_joints and joint.joint_type == 'fixed':
                continue
            joints_to_include.append(joint)

    # Filter materials referenced by the included links
    materials_to_include = []
    material_names_used = set()

    for link in links_to_include:
        for visual in link.visuals:
            if visual.material is not None and visual.material.name:
                material_names_used.add(visual.material.name)

    # Include materials that are actually used
    for material in urdf.materials:
        if material.name in material_names_used:
            materials_to_include.append(material)

    # Filter transmissions (keep only if all joints are included)
    transmissions_to_include = []
    joint_names_included = {j.name for j in joints_to_include}

    for transmission in urdf.transmissions:
        # Check if all actuator/joint references are in our joint set
        transmission_joints = []
        for actuator in transmission.actuators:
            if hasattr(actuator, 'joint'):
                transmission_joints.append(actuator.joint)

        if all(jname in joint_names_included for jname in transmission_joints):
            transmissions_to_include.append(transmission)

    # Create new URDF with filtered components
    sub_urdf = URDF(
        name=f"{urdf.name}_sub",
        links=list(links_to_include),
        joints=joints_to_include,
        materials=materials_to_include,
        transmissions=transmissions_to_include,
        other_xml=urdf.other_xml
    )

    # Save to file if output path is specified
    if output_urdf_path is not None:
        sub_urdf.save(output_urdf_path)

    return sub_urdf

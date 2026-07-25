"""
MoveIt 2 package builder.

Generates a complete MoveIt 2 ROS package from URDF content: launch
files, config files (SRDF, kinematics, controllers) and ROS 2 package
metadata (package.xml, CMakeLists.txt).
"""

from collections import defaultdict
import contextlib
import logging
from pathlib import Path
import re
from string import Template
import xml.dom.minidom
import xml.etree.ElementTree as ET

import yaml

from skrobot.data import data_dir
from skrobot.urdf.ros_package import rewrite_mesh_package_references


logger = logging.getLogger(__name__)


TEMPLATES_DIR = Path(data_dir) / "ros_config_templates"


def _load_template(name):
    """Load a template file from the templates directory.

    Parameters
    ----------
    name : str
        Template filename.

    Returns
    -------
    Template
        string.Template instance.
    """
    with open(TEMPLATES_DIR / name) as f:
        return Template(f.read())


def _write_file(content, output_path):
    """Write content to a file, creating parent directories as needed.

    Parameters
    ----------
    content : str
        File content.
    output_path : Path
        Destination path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)


def _write_yaml(data, output_path):
    """Write a dict as YAML with clean formatting.

    Parameters
    ----------
    data : dict
        Data to serialize.
    output_path : Path
        Destination path.
    """

    class _CleanDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, Dumper=_CleanDumper, default_flow_style=False, sort_keys=False)


def _parse_non_fixed_joints(urdf_string):
    """Extract non-fixed joint info from URDF.

    Parameters
    ----------
    urdf_string : str
        URDF XML content.

    Returns
    -------
    list of dict
        Sorted list of dicts with 'name' and 'max_velocity' keys.
    """
    root = ET.fromstring(urdf_string)
    joints_data = []
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        name = joint.get("name")
        if not name:
            continue
        max_velocity = 0.0
        limit_el = joint.find("limit")
        if limit_el is not None:
            vel_str = limit_el.get("velocity")
            if vel_str:
                with contextlib.suppress(ValueError, TypeError):
                    max_velocity = float(vel_str)
        joints_data.append({"name": name, "max_velocity": max_velocity})
    return sorted(joints_data, key=lambda x: x["name"])


def _get_kinematic_chains(root):
    """Detect kinematic chains (limbs) from URDF element tree.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        Root element of the URDF.

    Returns
    -------
    tuple
        (base_link_name, chains_dict, connected_links_list)
        chains_dict maps group names to lists of non-fixed joint names.
    """
    joints = {j.get("name"): j for j in root.findall("joint")}

    parent_to_child = defaultdict(list)
    child_to_parent = {}
    connected_links = []

    for joint in joints.values():
        parent_link = joint.find("parent").get("link")
        child_link = joint.find("child").get("link")
        connected_links.append((parent_link, child_link))
        parent_to_child[parent_link].append(joint)
        child_to_parent[child_link] = joint

    parent_links = set(parent_to_child.keys())
    child_links = set(child_to_parent.keys())
    base_links = parent_links - child_links

    if not base_links:
        if parent_links:
            base_link_name = next(iter(parent_links))
        else:
            return None, {}, []
    else:
        base_link_name = next(iter(base_links))

    chains = {}
    group_count = 1

    # Collect chains starting from each branch off the base link.
    # Use a queue so we can traverse through fixed-joint-only segments
    # to reach the actual branching point (e.g. world -> base_link -> limbs).
    queue = list(parent_to_child.get(base_link_name, []))
    while queue:
        start_joint = queue.pop(0)
        chain = []
        current_joint = start_joint
        # An Element's truth value reflects its child count, not its
        # existence, so compare against None explicitly.
        while current_joint is not None:
            if current_joint.get("type") != "fixed":
                chain.append(current_joint.get("name"))
            child_el = current_joint.find("child")
            if child_el is None:
                break
            child_link = child_el.get("link")
            next_joints = parent_to_child.get(child_link, [])
            if len(next_joints) == 1:
                current_joint = next_joints[0]
            elif len(next_joints) > 1:
                # Branching point reached.  If we haven't collected any
                # non-fixed joints yet, this is just a fixed-joint chain
                # leading to the real branching point – fan out from here.
                if not chain:
                    queue.extend(next_joints)
                # Otherwise, stop this chain (can't go further without
                # choosing a branch).
                current_joint = None
            else:
                current_joint = None

        if chain:
            chains[f"limb_{group_count}"] = chain
            group_count += 1

    return base_link_name, chains, connected_links


def _generate_srdf(robot_name, base_link_name, chains, connected_links, all_link_names=None):
    """Generate SRDF XML string.

    Parameters
    ----------
    robot_name : str
        Robot name.
    base_link_name : str
        Name of the base link.
    chains : dict
        Kinematic chain mapping.
    connected_links : list of tuple
        Adjacent link pairs.
    all_link_names : list of str, optional
        All link names in the URDF.  When provided, collisions are
        disabled for every pair (modular robots have many links in
        close proximity that are not directly adjacent).

    Returns
    -------
    str
        Pretty-printed SRDF XML.
    """
    from itertools import combinations

    root = ET.Element("robot", name=robot_name)

    for group_name, joint_names in chains.items():
        group_el = ET.SubElement(root, "group", name=group_name)
        for jn in joint_names:
            ET.SubElement(group_el, "joint", name=jn)

    ET.SubElement(
        root,
        "virtual_joint",
        name="world_virtual_joint",
        type="fixed",
        parent_frame="world",
        child_link=base_link_name,
    )

    for group_name, joint_names in chains.items():
        state_el = ET.SubElement(root, "group_state", name="home", group=group_name)
        for jn in joint_names:
            ET.SubElement(state_el, "joint", name=jn, value="0.0")

    # Disable collisions for all link pairs — modular robots have
    # many links in close proximity that cause false positives.
    if all_link_names:
        for link1, link2 in combinations(all_link_names, 2):
            ET.SubElement(root, "disable_collisions", link1=link1, link2=link2, reason="Default")

    xml_str = ET.tostring(root, "utf-8")
    dom = xml.dom.minidom.parseString(xml_str)
    return dom.toprettyxml(indent="  ")


def _generate_ros2_control_xacro(robot_name, chains):
    """Generate ros2_control xacro content.

    Parameters
    ----------
    robot_name : str
        Robot name.
    chains : dict
        Kinematic chain mapping.

    Returns
    -------
    str
        Xacro file content.
    """
    joint_tmpl = _load_template("ros2_control.xacro.joint.template")
    joint_blocks = ""
    for chain_joints in chains.values():
        for jn in chain_joints:
            joint_blocks += joint_tmpl.substitute(joint_name=jn) + "\n"

    main_tmpl = _load_template("ros2_control.xacro.template")
    return main_tmpl.substitute(robot_name=robot_name, joint_blocks=joint_blocks)


def _generate_ros2_urdf(urdf_string, package_name):
    """Add ros2_control and Gazebo plugin elements to URDF.

    Parameters
    ----------
    urdf_string : str
        Original URDF content.
    package_name : str
        Package name for controller config path.

    Returns
    -------
    str
        Modified URDF with ros2_control elements.
    """
    root = ET.fromstring(urdf_string)
    non_fixed = [j for j in root.findall("joint") if j.get("type") != "fixed"]

    # Widen joint limits slightly so MoveIt/OMPL can plan even when
    # joints drift to limit values due to gravity before controllers start.
    import math

    LIMIT_MARGIN = math.radians(5)  # 5 degrees
    for joint in non_fixed:
        limit_el = joint.find("limit")
        if limit_el is not None:
            lower = limit_el.get("lower")
            upper = limit_el.get("upper")
            if lower is not None:
                limit_el.set("lower", str(float(lower) - LIMIT_MARGIN))
            if upper is not None:
                limit_el.set("upper", str(float(upper) + LIMIT_MARGIN))

    if non_fixed:
        ros2_ctrl = ET.Element("ros2_control", attrib={"name": "ModularRobotHardware", "type": "system"})
        hw = ET.SubElement(ros2_ctrl, "hardware")
        plugin = ET.SubElement(hw, "plugin")
        plugin.text = "gz_ros2_control/GazeboSimSystem"

        for joint in non_fixed:
            jname = joint.get("name")
            if not jname:
                continue
            # Mimic joints are passive: their motion is constrained to a
            # driving joint via the URDF <mimic> tag, which gz_ros2_control
            # enforces directly. Declaring a command interface for them would
            # double-control the joint and conflict with the mimic constraint.
            if joint.find("mimic") is not None:
                continue
            jel = ET.SubElement(ros2_ctrl, "joint", attrib={"name": jname})
            ET.SubElement(jel, "command_interface", attrib={"name": "position"})
            ET.SubElement(jel, "state_interface", attrib={"name": "position"})
            ET.SubElement(jel, "state_interface", attrib={"name": "velocity"})

        gazebo_el = ET.Element("gazebo")
        gz_plugin = ET.SubElement(
            gazebo_el,
            "plugin",
            attrib={
                "filename": "libgz_ros2_control-system.so",
                "name": "gz_ros2_control::GazeboSimROS2ControlPlugin",
            },
        )
        params = ET.SubElement(gz_plugin, "parameters")
        params.text = f"$(find {package_name})/config/ros2_controllers.yaml"

        root.append(gazebo_el)
        root.append(ros2_ctrl)

    if hasattr(ET, "indent"):
        ET.indent(root, space="  ", level=0)
    return ET.tostring(root, encoding="unicode", method="xml")


def _patch_dae_for_gazebo(dae_path):
    """Post-process a COLLADA file for Gazebo compatibility.

    - Replaces ``Y_UP`` with ``Z_UP`` (trimesh writes Y_UP but mesh
      data is in Z_UP / ROS convention).
    - Removes ``<transparent>`` and ``<transparency>`` elements.
      trimesh writes ``transparency=1.0`` which means fully transparent
      in COLLADA's A_ONE mode and causes black rendering in Gazebo.
    """
    dae_path = Path(dae_path)
    data = dae_path.read_text()
    data = data.replace("<up_axis>Y_UP</up_axis>", "<up_axis>Z_UP</up_axis>")
    data = re.sub(r"\s*<transparent>\s*<color>[^<]+</color>\s*</transparent>", "", data)
    data = re.sub(r"\s*<transparency>\s*<float>[^<]+</float>\s*</transparency>", "", data)
    dae_path.write_text(data)


def _convert_urdf_meshes(urdf_path, output_path, visual_format="stl", collision_format="stl"):
    """Convert URDF meshes using scikit-robot's export pipeline.

    This mirrors the ``convert-urdf-mesh`` CLI tool: load the URDF via
    :class:`~skrobot.model.RobotModel`, then re-save it inside an
    :func:`~skrobot.utils.urdf.export_mesh_format` context so that every
    mesh is converted to the requested format.

    Parameters
    ----------
    urdf_path : str or Path
        Input URDF file path.
    output_path : str or Path
        Output URDF file path.
    visual_format : str
        Target format for visual meshes (``'dae'``, ``'stl'``, ``'glb'``).
    collision_format : str
        Target format for collision meshes (``'dae'``, ``'stl'``, ``'glb'``).

    Returns
    -------
    bool
        True if conversion succeeded.
    """
    try:
        from skrobot.model import RobotModel
        from skrobot.utils.draco import is_dracopy_available
        from skrobot.utils.draco import register_dracopy_handlers
        from skrobot.utils.urdf import _CONFIGURABLE_VALUES
        from skrobot.utils.urdf import apply_scale
        from skrobot.utils.urdf import export_mesh_format

        if is_dracopy_available():
            register_dracopy_handlers()

        urdf_path = str(Path(urdf_path).resolve())
        _CONFIGURABLE_VALUES["_source_urdf_path"] = str(Path(urdf_path).parent)

        r = RobotModel.from_urdf(urdf_path)

        with export_mesh_format(
            "." + visual_format,
            collision_mesh_format="." + collision_format,
            overwrite_mesh=True,
        ), apply_scale(1.0):
            r.urdf_robot_model.save(str(output_path))

        # Fix DAE up_axis: trimesh always writes Y_UP but mesh data is
        # in Z_UP (ROS convention).  Gazebo would apply an unwanted
        # Y→Z rotation, so patch the header to Z_UP.
        urdf_dir = Path(urdf_path).parent
        meshes_dir = urdf_dir.parent / "meshes"
        if meshes_dir.exists():
            for dae_file in meshes_dir.rglob("*.dae"):
                _patch_dae_for_gazebo(dae_file)

        return True
    except Exception as e:
        logger.warning("Failed to convert URDF meshes: %s", e, exc_info=True)
        return False


def build_moveit_package(
    package_dir,
    urdf_content,
    robot_name,
    package_name=None,
):
    """Build a complete MoveIt 2 ROS package directory.

    Generates all config files, launch files, and package metadata
    needed for ``colcon build`` and ``ros2 launch``.

    Parameters
    ----------
    package_dir : str or Path
        Root directory for the package output.
    urdf_content : str
        URDF XML content.
    robot_name : str
        Name of the robot (used in filenames and configs).
    package_name : str, optional
        ROS package name. Defaults to ``robot_name``.
    """
    package_dir = Path(package_dir)
    if package_name is None:
        package_name = robot_name

    try:
        urdf_root = ET.fromstring(urdf_content)
    except ET.ParseError as e:
        logger.error("Error parsing URDF: %s", e)
        return

    joints_data = _parse_non_fixed_joints(urdf_content)
    if not joints_data:
        logger.warning(
            "No controllable joints found. Skipping MoveIt config generation.")
        return

    config_dir = package_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    launch_dir = package_dir / "launch"
    launch_dir.mkdir(parents=True, exist_ok=True)
    urdf_dir = package_dir / "urdf"
    urdf_dir.mkdir(parents=True, exist_ok=True)

    # --- Rewrite mesh paths to use the target package name ---
    urdf_content = rewrite_mesh_package_references(urdf_content, package_name)

    # --- URDF file (in urdf/ directory, standard ROS convention) ---
    _write_file(urdf_content, urdf_dir / f"{robot_name}.urdf")

    # --- Convert all meshes to STL ---
    gazebo_urdf_path = urdf_dir / f"{robot_name}.urdf"
    _convert_urdf_meshes(
        gazebo_urdf_path,
        gazebo_urdf_path,
        visual_format="stl",
        collision_format="stl",
    )

    # --- Config files ---

    # Re-read the converted URDF
    with open(urdf_dir / f"{robot_name}.urdf") as f:
        urdf_content = f.read()

    # URDF with ros2_control (for Gazebo / controllers)
    ros2_urdf = _generate_ros2_urdf(urdf_content, package_name)
    _write_file(ros2_urdf, config_dir / f"{robot_name}.urdf")

    # Initial positions
    initial_positions = {j["name"]: 0 for j in joints_data}
    _write_yaml({"initial_positions": initial_positions}, config_dir / "initial_positions.yaml")

    # Joint limits
    joint_limits = {}
    for j in joints_data:
        max_vel = j.get("max_velocity", 0.0)
        has_vel = max_vel > 0
        joint_limits[j["name"]] = {
            "has_velocity_limits": has_vel,
            "max_velocity": max_vel,
            "has_acceleration_limits": has_vel,
            "max_acceleration": 2.0 if has_vel else 0.0,
        }
    _write_yaml(
        {
            "default_velocity_scaling_factor": 0.1,
            "default_acceleration_scaling_factor": 0.1,
            "joint_limits": joint_limits,
        },
        config_dir / "joint_limits.yaml",
    )

    # Pilz cartesian limits
    _write_yaml(
        {
            "cartesian_limits": {
                "max_trans_vel": 1.0,
                "max_trans_acc": 2.25,
                "max_trans_dec": -5.0,
                "max_rot_vel": 1.57,
            }
        },
        config_dir / "pilz_cartesian_limits.yaml",
    )

    # Kinematic chains
    base_link_name, chains, connected_links = _get_kinematic_chains(urdf_root)

    if chains:
        # SRDF
        all_link_names = [link.get("name") for link in urdf_root.findall("link")]
        srdf_content = _generate_srdf(
            robot_name,
            base_link_name,
            chains,
            connected_links,
            all_link_names=all_link_names,
        )
        _write_file(srdf_content, config_dir / f"{robot_name}.srdf")

        # ros2_control xacro
        ros2_ctrl_xacro = _generate_ros2_control_xacro(robot_name, chains)
        _write_file(ros2_ctrl_xacro, config_dir / f"{robot_name}.ros2_control.xacro")

        # Kinematics
        kinematics = {}
        for group_name in chains:
            kinematics[group_name] = {
                "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
                "kinematics_solver_search_resolution": 0.005,
                "kinematics_solver_timeout": 0.005,
            }
        _write_yaml(kinematics, config_dir / "kinematics.yaml")

        # All joint names across chains, excluding mimic (passive) joints.
        # Mimic joints follow a driving joint via the URDF <mimic> tag, so a
        # trajectory controller must not command them directly.
        mimic_joint_names = {j.get("name") for j in urdf_root.findall("joint") if j.find("mimic") is not None}
        all_joints = [jn for chain_joints in chains.values() for jn in chain_joints if jn not in mimic_joint_names]

        # MoveIt controllers
        _write_yaml(
            {
                "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
                "moveit_simple_controller_manager": {
                    "controller_names": ["all_joints_controller"],
                    "all_joints_controller": {
                        "type": "FollowJointTrajectory",
                        "action_ns": "follow_joint_trajectory",
                        "joints": all_joints,
                    },
                },
            },
            config_dir / "moveit_controllers.yaml",
        )

        # ROS 2 controllers
        _write_yaml(
            {
                "controller_manager": {
                    "ros__parameters": {
                        "update_rate": 100,
                        "all_joints_controller": {
                            "type": "joint_trajectory_controller/JointTrajectoryController",
                        },
                        "joint_state_broadcaster": {
                            "type": "joint_state_broadcaster/JointStateBroadcaster",
                        },
                    },
                },
                "all_joints_controller": {
                    "ros__parameters": {
                        "joints": all_joints,
                        "command_interfaces": ["position"],
                        "state_interfaces": ["position", "velocity"],
                        "allow_nonzero_velocity_at_trajectory_end": True,
                    },
                },
            },
            config_dir / "ros2_controllers.yaml",
        )
    else:
        base_link_name = base_link_name or "base_link"
        logger.warning(
            "No kinematic chains detected. Skipping SRDF/controller generation.")

    # RViz config
    rviz_tmpl = _load_template("moveit.rviz.template")
    _write_file(
        rviz_tmpl.substitute(base_link_name=base_link_name),
        config_dir / "moveit.rviz",
    )

    # --- Launch files ---
    # Gazebo launch (Gazebo + RViz + controllers)
    gazebo_tmpl = _load_template("gazebo.launch.py.template")
    _write_file(
        gazebo_tmpl.substitute(robot_name=robot_name, package_name=package_name),
        launch_dir / "gazebo.launch.py",
    )

    # Gazebo + MoveIt + RViz launch
    gazebo_moveit_tmpl = _load_template("gazebo_moveit.launch.py.template")
    _write_file(
        gazebo_moveit_tmpl.substitute(robot_name=robot_name, package_name=package_name),
        launch_dir / "gazebo_moveit.launch.py",
    )

    # --- Package metadata ---
    pkg_xml_tmpl = _load_template("package.xml.ros2.template")
    _write_file(pkg_xml_tmpl.substitute(package_name=package_name), package_dir / "package.xml")

    cmake_tmpl = _load_template("CMakeLists.txt.ros2.template")
    _write_file(cmake_tmpl.substitute(package_name=package_name), package_dir / "CMakeLists.txt")

    # .setup_assistant
    sa_tmpl = _load_template("setup_assistant.template")
    _write_file(
        sa_tmpl.substitute(package_name=package_name, robot_name=robot_name),
        package_dir / ".setup_assistant",
    )

    logger.info("MoveIt package generated at: %s", package_dir)

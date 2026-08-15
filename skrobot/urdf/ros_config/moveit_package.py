"""
MoveIt 2 package builder.

Generates a complete MoveIt 2 ROS package from URDF content: launch
files, config files (SRDF, kinematics, controllers) and ROS 2 package
metadata (package.xml, CMakeLists.txt).
"""

from itertools import combinations
import logging
from pathlib import Path
import re
from string import Template
import xml.etree.ElementTree as ET

import yaml

from skrobot.data import data_dir
from skrobot.urdf.ros_config.gazebo_generator import build_ros2_control_elements
from skrobot.urdf.ros_config.moveit_generator import generate_srdf
from skrobot.urdf.ros_config.urdf_parser import parse_urdf_content
from skrobot.urdf.ros_package import rewrite_mesh_package_references

# _load_structure is private to skrobot.urdf.structure but is the
# single place that turns a URDF into a link/joint graph; re-deriving
# the parent/child maps here would be a second copy of it.
from skrobot.urdf.structure import _load_structure


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
    with open(TEMPLATES_DIR / name, encoding="utf-8") as f:
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
    with open(output_path, "w", encoding="utf-8") as f:
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
    with open(output_path, "w", encoding="utf-8") as f:
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
    joints = parse_urdf_content(urdf_string)["joints"]
    return sorted(
        ({"name": joint["name"],
          "max_velocity": joint["velocity_limit"]}
         for joint in joints if joint["type"] != "fixed"),
        key=lambda joint: joint["name"])


def _get_kinematic_chains(structure):
    """Detect kinematic chains (limbs) from a parsed URDF structure.

    Parameters
    ----------
    structure : skrobot.urdf.structure._UrdfStructure
        Parsed URDF, as returned by ``_load_structure``.

    Returns
    -------
    tuple
        (base_link_name, chains_dict, connected_links_list)
        chains_dict maps group names to lists of non-fixed joint names.
    """
    joints = list({joint.name: joint for joint in structure.joints}.values())
    for joint in joints:
        if joint.parent is None or joint.child is None:
            raise ValueError(
                "joint '{}' is missing its <parent> or <child> link".format(
                    joint.name))

    # Indexed by parent link, not by child: two joints declaring the same
    # child is malformed, but indexing by child would silently attribute
    # one parent's joint to the other and produce an inconsistent walk.
    joints_below = {}
    for joint in joints:
        joints_below.setdefault(joint.parent, []).append(joint)
    child_links = {joint.child for joint in joints}
    connected_links = [(joint.parent, joint.child) for joint in joints]

    # Ordered by first appearance among the joints rather than taken from a
    # set, so a URDF with several roots always picks the same one -- string
    # hashing is salted per process, and set iteration order follows it.
    parent_links = list(dict.fromkeys(joint.parent for joint in joints))
    base_links = [link for link in parent_links if link not in child_links]

    if base_links:
        base_link_name = base_links[0]
    elif parent_links:
        base_link_name = parent_links[0]
    else:
        return None, {}, []

    chains = {}
    group_count = 1

    # Collect chains starting from each branch off the base link.
    # Use a queue so we can traverse through fixed-joint-only segments
    # to reach the actual branching point (e.g. world -> base_link -> limbs).
    queue = list(joints_below.get(base_link_name, []))
    while queue:
        current_joint = queue.pop(0)
        chain = []
        # A URDF kinematic tree cannot contain a loop, but nothing here
        # validates that; without this the walk would spin forever on one.
        walked = set()
        while current_joint is not None and current_joint.name not in walked:
            walked.add(current_joint.name)
            if current_joint.joint_type != "fixed":
                chain.append(current_joint.name)
            next_joints = joints_below.get(current_joint.child, [])
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
    """Generate SRDF XML string for a set of kinematic chains.

    Thin wrapper over :func:`~skrobot.urdf.ros_config.moveit_generator.generate_srdf`
    that turns detected chains into planning groups and gives every group
    a zeroed ``home`` state.

    Parameters
    ----------
    robot_name : str
        Robot name.
    base_link_name : str
        Name of the base link, anchored to the ``world`` frame by a
        fixed virtual joint.
    chains : dict
        Kinematic chain mapping.
    connected_links : list of tuple
        Adjacent link pairs, used when ``all_link_names`` is not given.
    all_link_names : list of str, optional
        All link names in the URDF.  When provided, collisions are
        disabled for every pair (modular robots have many links in
        close proximity that are not directly adjacent).

    Returns
    -------
    str
        SRDF XML.
    """
    planning_groups = [{"name": group_name, "joints": joint_names}
                       for group_name, joint_names in chains.items()]
    group_states = [{"name": "home", "group": group_name,
                     "joint_values": {jn: "0.0" for jn in joint_names}}
                    for group_name, joint_names in chains.items()]
    if all_link_names:
        disabled_pairs = list(combinations(all_link_names, 2))
    else:
        disabled_pairs = list(connected_links)

    return generate_srdf(
        robot_name,
        planning_groups,
        disabled_pairs,
        virtual_joint={"name": "world_virtual_joint",
                       "type": "fixed",
                       "parent_frame": "world",
                       "child_link": base_link_name},
        group_states=group_states,
        disabled_collision_reason="Default")


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
        # Mimic joints are passive: their motion is constrained to a driving
        # joint via the URDF <mimic> tag, which gz_ros2_control enforces
        # directly. Declaring a command interface for them would
        # double-control the joint and conflict with the mimic constraint.
        joint_names = [
            joint.get("name")
            for joint in non_fixed
            if joint.get("name") and joint.find("mimic") is None
        ]
        ros2_ctrl, gazebo_el = build_ros2_control_elements(
            joint_names,
            package_name=package_name,
            controllers_file="ros2_controllers.yaml",
            name="ModularRobotHardware",
        )

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
    data = dae_path.read_text(encoding="utf-8")
    data = data.replace("<up_axis>Y_UP</up_axis>", "<up_axis>Z_UP</up_axis>")
    data = re.sub(r"\s*<transparent>\s*<color>[^<]+</color>\s*</transparent>", "", data)
    data = re.sub(r"\s*<transparency>\s*<float>[^<]+</float>\s*</transparency>", "", data)
    dae_path.write_text(data, encoding="utf-8")


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
        from skrobot.utils.draco import is_dracopy_available
        from skrobot.utils.draco import register_dracopy_handlers
        from skrobot.utils.urdf import convert_urdf_meshes

        # Registered globally so a .drc mesh referenced by the URDF can be
        # read back while saving.
        if is_dracopy_available():
            register_dracopy_handlers()

        urdf_path = str(Path(urdf_path).resolve())
        convert_urdf_meshes(
            urdf_path,
            output_path,
            "." + visual_format,
            collision_mesh_format="." + collision_format,
            overwrite_mesh=True,
        )

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
        # Raises ValueError on a link or joint without a name, which no
        # amount of downstream config generation could recover from.
        structure = _load_structure(urdf_content)
    except (ET.ParseError, ValueError) as e:
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
    with open(urdf_dir / f"{robot_name}.urdf", encoding="utf-8") as f:
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
    base_link_name, chains, connected_links = _get_kinematic_chains(structure)

    if chains:
        # SRDF
        all_link_names = structure.links
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

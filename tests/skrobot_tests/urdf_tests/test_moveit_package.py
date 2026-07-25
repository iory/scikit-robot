"""Characterization tests for :func:`build_moveit_package`.

These pin the output of the MoveIt 2 package generator so it can be
refactored without silently changing what lands in an exported
package.  They assert structure and semantics (which files exist,
which joints end up in which config) rather than byte-exact content,
so formatting changes stay cheap.

Everything runs off inline, mesh-free URDFs so the tests need no
robot description on disk.
"""

import ast
from itertools import combinations
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

from skrobot.urdf.ros_config import build_moveit_package


ROBOT_NAME = "probe_robot"
PACKAGE_NAME = "probe_pkg"

# 5 degrees, the margin _generate_ros2_urdf adds to every joint limit.
LIMIT_MARGIN = 0.0872664625997165


def _link(name):
    return (
        f'<link name="{name}">'
        '<visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>'
        '<inertial><mass value="1.0"/>'
        '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>'
        "</link>"
    )


def _joint(name, parent, child, jtype="revolute", velocity="2.0", mimic=None):
    body = f'<parent link="{parent}"/><child link="{child}"/><origin xyz="0 0 0.1" rpy="0 0 0"/>'
    if jtype != "fixed":
        body += '<axis xyz="0 0 1"/>'
        body += f'<limit lower="-1.0" upper="1.0" effort="10" velocity="{velocity}"/>'
    if mimic:
        body += f'<mimic joint="{mimic}" multiplier="1.0" offset="0.0"/>'
    return f'<joint name="{name}" type="{jtype}">{body}</joint>'


def _robot(links, joints, name="r"):
    return (
        '<?xml version="1.0"?>\n'
        f'<robot name="{name}">\n' + "".join(_link(link) for link in links) + "".join(joints) + "\n</robot>\n"
    )


# A single limb: base -> link1 -> link2 -> finger, where ``finger_mimic``
# is passively driven by ``joint2``.
SERIAL_URDF = _robot(
    ["base_link", "link1", "link2", "finger"],
    [
        _joint("joint1", "base_link", "link1", velocity="2.0"),
        _joint("joint2", "link1", "link2", velocity="3.0"),
        _joint("finger_mimic", "link2", "finger", velocity="1.0", mimic="joint2"),
    ],
)

# Two limbs branching off a shared base link.
BRANCHING_URDF = _robot(
    ["base_link", "l_a", "l_b", "r_a", "r_b"],
    [
        _joint("l_j1", "base_link", "l_a"),
        _joint("l_j2", "l_a", "l_b"),
        _joint("r_j1", "base_link", "r_a"),
        _joint("r_j2", "r_a", "r_b"),
    ],
)

# Nothing controllable -- only a fixed joint.
FIXED_ONLY_URDF = _robot(["base_link", "tip"], [_joint("fix", "base_link", "tip", jtype="fixed")])


@pytest.fixture(scope="module")
def serial_package(tmp_path_factory):
    """Build the single-limb package once and share it across tests."""
    package_dir = tmp_path_factory.mktemp("serial") / PACKAGE_NAME
    build_moveit_package(package_dir, SERIAL_URDF, ROBOT_NAME, package_name=PACKAGE_NAME)
    return package_dir


def _read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_generated_package_tree(serial_package):
    """Every file colcon/MoveIt needs is written, at its expected path."""
    expected = {
        ".setup_assistant",
        "CMakeLists.txt",
        "package.xml",
        f"urdf/{ROBOT_NAME}.urdf",
        f"config/{ROBOT_NAME}.urdf",
        f"config/{ROBOT_NAME}.srdf",
        f"config/{ROBOT_NAME}.ros2_control.xacro",
        "config/initial_positions.yaml",
        "config/joint_limits.yaml",
        "config/kinematics.yaml",
        "config/moveit.rviz",
        "config/moveit_controllers.yaml",
        "config/pilz_cartesian_limits.yaml",
        "config/ros2_controllers.yaml",
        "launch/gazebo.launch.py",
        "launch/gazebo_moveit.launch.py",
    }
    actual = {p.relative_to(serial_package).as_posix() for p in serial_package.rglob("*") if p.is_file()}
    assert expected <= actual, expected - actual


def test_srdf_group_virtual_joint_and_home_state(serial_package):
    """The chain becomes one planning group with a home state at zero."""
    root = ET.parse(serial_package / "config" / f"{ROBOT_NAME}.srdf").getroot()

    groups = root.findall("group")
    assert [g.get("name") for g in groups] == ["limb_1"]
    # The mimic joint stays in the planning group even though the
    # controllers below exclude it -- see the note in
    # test_mimic_joint_is_excluded_from_controllers.
    assert [j.get("name") for j in groups[0].findall("joint")] == ["joint1", "joint2", "finger_mimic"]

    virtual = root.find("virtual_joint")
    assert virtual.get("type") == "fixed"
    assert virtual.get("parent_frame") == "world"
    assert virtual.get("child_link") == "base_link"

    state = root.find("group_state")
    assert state.get("name") == "home"
    assert state.get("group") == "limb_1"
    assert {j.get("value") for j in state.findall("joint")} == {"0.0"}


def test_srdf_disables_every_link_pair(serial_package):
    """Modular robots pack links close together, so all pairs are disabled."""
    root = ET.parse(serial_package / "config" / f"{ROBOT_NAME}.srdf").getroot()
    links = ["base_link", "link1", "link2", "finger"]

    disabled = {frozenset((el.get("link1"), el.get("link2"))) for el in root.findall("disable_collisions")}
    assert disabled == {frozenset(pair) for pair in combinations(links, 2)}


def test_mimic_joint_is_excluded_from_controllers(serial_package):
    """A mimic joint is driven through the URDF <mimic> tag, not commanded.

    Commanding it as well would fight the mimic constraint, so it must
    not appear in either controller config.
    """
    ros2 = _read_yaml(serial_package / "config" / "ros2_controllers.yaml")
    moveit = _read_yaml(serial_package / "config" / "moveit_controllers.yaml")

    assert ros2["all_joints_controller"]["ros__parameters"]["joints"] == ["joint1", "joint2"]
    assert moveit["moveit_simple_controller_manager"]["all_joints_controller"]["joints"] == ["joint1", "joint2"]

    controller_types = ros2["controller_manager"]["ros__parameters"]
    assert controller_types["all_joints_controller"]["type"] == (
        "joint_trajectory_controller/JointTrajectoryController"
    )
    assert controller_types["joint_state_broadcaster"]["type"] == "joint_state_broadcaster/JointStateBroadcaster"


def test_config_urdf_carries_gazebo_sim_ros2_control(serial_package):
    """config/<robot>.urdf is the simulation URDF: gz_ros2_control wired in."""
    root = ET.parse(serial_package / "config" / f"{ROBOT_NAME}.urdf").getroot()

    ros2_control = root.find("ros2_control")
    assert ros2_control.get("type") == "system"
    assert ros2_control.find("hardware/plugin").text == "gz_ros2_control/GazeboSimSystem"

    commanded = {j.get("name") for j in ros2_control.findall("joint")}
    assert commanded == {"joint1", "joint2"}, "mimic joints must not get a command interface"

    plugin = root.find("gazebo/plugin")
    assert plugin.get("filename") == "libgz_ros2_control-system.so"
    assert plugin.find("parameters").text == f"$(find {PACKAGE_NAME})/config/ros2_controllers.yaml"


def test_config_urdf_widens_joint_limits(serial_package):
    """Limits are padded so OMPL can still plan when gravity parks a joint at its stop."""
    root = ET.parse(serial_package / "config" / f"{ROBOT_NAME}.urdf").getroot()

    limit = root.find(".//joint[@name='joint1']/limit")
    assert float(limit.get("lower")) == pytest.approx(-1.0 - LIMIT_MARGIN)
    assert float(limit.get("upper")) == pytest.approx(1.0 + LIMIT_MARGIN)


def test_joint_limits_and_initial_positions(serial_package):
    """Velocity limits come from the URDF; every joint starts at zero."""
    limits = _read_yaml(serial_package / "config" / "joint_limits.yaml")
    assert limits["default_velocity_scaling_factor"] == 0.1
    assert limits["joint_limits"]["joint1"]["max_velocity"] == 2.0
    assert limits["joint_limits"]["joint2"]["max_velocity"] == 3.0
    assert limits["joint_limits"]["joint1"]["has_velocity_limits"] is True

    initial = _read_yaml(serial_package / "config" / "initial_positions.yaml")
    assert initial["initial_positions"] == {"joint1": 0, "joint2": 0, "finger_mimic": 0}


def test_kinematics_solver_per_group(serial_package):
    kinematics = _read_yaml(serial_package / "config" / "kinematics.yaml")
    assert set(kinematics) == {"limb_1"}
    assert kinematics["limb_1"]["kinematics_solver"] == "kdl_kinematics_plugin/KDLKinematicsPlugin"


def test_package_metadata_is_substituted(serial_package):
    """Templates are rendered with the package/robot name, not left raw."""
    assert f"<name>{PACKAGE_NAME}</name>" in (serial_package / "package.xml").read_text()
    assert f"project({PACKAGE_NAME})" in (serial_package / "CMakeLists.txt").read_text()

    setup_assistant = _read_yaml(serial_package / ".setup_assistant")["moveit_setup_assistant_config"]
    assert setup_assistant["urdf"]["package"] == PACKAGE_NAME
    assert setup_assistant["urdf"]["relative_path"] == f"config/{ROBOT_NAME}.urdf"
    assert setup_assistant["srdf"]["relative_path"] == f"config/{ROBOT_NAME}.srdf"


def test_launch_files_are_valid_python(serial_package):
    """The launch files ship as-is to the user; a template typo must fail here."""
    for launch_file in sorted((serial_package / "launch").glob("*.py")):
        ast.parse(launch_file.read_text(), filename=str(launch_file))


def test_branching_urdf_produces_one_group_per_limb(tmp_path):
    """Each branch off the base link becomes its own planning group."""
    package_dir = tmp_path / PACKAGE_NAME
    build_moveit_package(package_dir, BRANCHING_URDF, "r", package_name=PACKAGE_NAME)

    root = ET.parse(package_dir / "config" / "r.srdf").getroot()
    groups = {g.get("name"): [j.get("name") for j in g.findall("joint")] for g in root.findall("group")}

    assert set(groups) == {"limb_1", "limb_2"}
    assert sorted(sorted(joints) for joints in groups.values()) == [["l_j1", "l_j2"], ["r_j1", "r_j2"]]
    assert set(_read_yaml(package_dir / "config" / "kinematics.yaml")) == {"limb_1", "limb_2"}


def test_urdf_without_controllable_joints_generates_nothing(tmp_path):
    """A fixed-only URDF bails out before creating the package directory."""
    package_dir = tmp_path / PACKAGE_NAME
    build_moveit_package(package_dir, FIXED_ONLY_URDF, "r", package_name=PACKAGE_NAME)

    assert not package_dir.exists()


def test_unparsable_urdf_is_reported_not_raised(tmp_path):
    """Malformed URDF returns early instead of propagating a parse error."""
    package_dir = tmp_path / PACKAGE_NAME
    build_moveit_package(package_dir, "<robot><not-closed>", "r", package_name=PACKAGE_NAME)

    assert not package_dir.exists()


def test_meshes_are_not_required(serial_package):
    """The exported URDF keeps its primitive geometry through mesh conversion."""
    urdf_text = Path(serial_package / "urdf" / f"{ROBOT_NAME}.urdf").read_text()
    assert "<box" in urdf_text

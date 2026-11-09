"""PybulletRobotInterface as a RobotInterfaceBase (timed motions)."""
import numpy as np
import pytest

from skrobot.interfaces._pybullet import _check_available


pytestmark = pytest.mark.skipif(
    _check_available(silent=True) is False,
    reason="Pybullet is not available")


_URDF = """<?xml version="1.0"?>
<robot name="toy">
  <link name="base_link">
    <inertial><mass value="1.0"/>
      <inertia ixx="1e-2" ixy="0" ixz="0" iyy="1e-2" iyz="0" izz="1e-2"/>
    </inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/><child link="upper"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="10.0" velocity="2.0"/>
  </joint>
  <link name="upper">
    <inertial><mass value="0.5"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
  </link>
  <joint name="slider" type="prismatic">
    <parent link="upper"/><child link="slide_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="0.2" effort="10.0" velocity="0.1"/>
  </joint>
  <link name="slide_link">
    <inertial><mass value="0.2"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
  </link>
</robot>
"""


@pytest.fixture
def ri(tmp_path):
    import pybullet

    from skrobot.interfaces._pybullet import PybulletRobotInterface
    from skrobot.model import RobotModel

    urdf_path = tmp_path / 'toy.urdf'
    urdf_path.write_text(_URDF)
    robot = RobotModel.from_urdf(str(urdf_path))
    interface = PybulletRobotInterface(
        robot, urdf_path=str(urdf_path), use_fixed_base=True, connect=2)
    yield interface
    pybullet.resetSimulation()


def _index(ri, name):
    return [j.name for j in ri.robot.joint_list].index(name)


def test_is_a_robot_interface_and_coordinates(ri):
    from skrobot.coordinates import Coordinates
    from skrobot.interfaces.base import RobotInterfaceBase
    assert isinstance(ri, RobotInterfaceBase)
    assert isinstance(ri, Coordinates)
    assert ri.joint_names() == ['shoulder', 'slider']
    ri.translate([0.0, 0.0, 0.1])
    assert ri.pose.translation[2] == pytest.approx(0.1)


def test_timed_motion_reaches_the_goal(ri):
    ri.angle_vector({'shoulder': 0.5}, 1.0)
    assert ri.is_interpolating()
    start = ri.sim_time
    assert ri.wait_interpolation() is True
    assert ri.sim_time - start >= 1.0 - 1e-6
    assert not ri.is_interpolating()
    assert ri.measured_joint_angles()['shoulder'] == pytest.approx(
        0.5, abs=1e-2)


def test_timed_motion_is_on_the_way_halfway(ri):
    ri.angle_vector({'shoulder': 0.5}, 1.0)
    dt = 1.0 / 240.0
    ri.step(int(round(0.5 / dt)))
    ri.update_robot_state()
    middle = ri.measured_joint_angles()['shoulder']
    assert 0.05 < middle < 0.45


def test_duration_from_velocity_limits(ri):
    # slider 0.1 m at 0.1 m/s -> 1 s, times time_scale 2
    ri.angle_vector({'slider': 0.1}, time_scale=2.0)
    start = ri.sim_time
    ri.wait_interpolation()
    assert ri.sim_time - start >= 2.0 - 1e-6
    assert ri.measured_joint_angles()['slider'] == pytest.approx(
        0.1, abs=1e-2)


def test_dict_goal_moves_only_named_joints(ri):
    ri.angle_vector({'slider': 0.15}, 2.0)
    ri.wait_interpolation()
    ri.angle_vector({'shoulder': -0.3}, 1.0)
    ri.wait_interpolation()
    measured = ri.measured_joint_angles()
    assert measured['shoulder'] == pytest.approx(-0.3, abs=1e-2)
    assert measured['slider'] == pytest.approx(0.15, abs=1e-2)


def test_unknown_joint_is_refused(ri):
    with pytest.raises(KeyError):
        ri.angle_vector({'no_such_joint': 0.1}, 1.0)


def test_old_call_still_resets_joints_at_once(ri):
    av = np.zeros(len(ri.robot.joint_list))
    av[_index(ri, 'shoulder')] = 0.4
    returned = ri.angle_vector(av)
    assert returned is av
    ri.update_robot_state()
    assert ri.measured_joint_angles()['shoulder'] == pytest.approx(0.4)
    assert not ri.is_interpolating()
    # Old positional realtime_simulation flag: motor target only.
    av[_index(ri, 'shoulder')] = -0.4
    ri.angle_vector(av, True)
    assert ri.realtime_simulation is True
    ri.update_robot_state()
    assert ri.measured_joint_angles()['shoulder'] == pytest.approx(0.4)
    assert ri.wait_interpolation() is True
    assert ri.measured_joint_angles()['shoulder'] == pytest.approx(
        -0.4, abs=1e-2)


def test_measured_vector_is_not_clamped(ri):
    av = np.zeros(len(ri.robot.joint_list))
    av[_index(ri, 'shoulder')] = 1.3  # past the 1.0 limit
    ri.angle_vector(av)  # resetJointState puts it there
    ri.update_robot_state()
    assert ri.measured_angle_vector()[_index(ri, 'shoulder')] \
        == pytest.approx(1.3)
    report = ri.outside_limits()
    assert list(report) == ['shoulder']
    assert report['shoulder'] == pytest.approx((1.3, -1.0, 1.0))


def test_cancel_stops_where_it_is(ri):
    ri.angle_vector({'shoulder': 0.8}, 2.0)
    ri.step(240)
    ri.cancel_angle_vector()
    assert not ri.is_interpolating()
    ri.update_robot_state()
    stopped = ri.measured_joint_angles()['shoulder']
    assert 0.05 < stopped < 0.75
    ri.wait_interpolation()
    assert ri.measured_joint_angles()['shoulder'] == pytest.approx(
        stopped, abs=2e-2)


def test_servo_hooks_default_to_not_implemented(ri):
    with pytest.raises(NotImplementedError):
        ri.servo_on()

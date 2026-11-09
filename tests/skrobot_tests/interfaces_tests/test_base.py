import time

import numpy as np
import pytest

from skrobot.interfaces.base import RobotInterfaceBase
from skrobot.interfaces.fake import FakeRobotInterface
from skrobot.model import RobotModel


_URDF = """<?xml version="1.0"?>
<robot name="toy">
  <link name="base_link"/>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/><child link="upper"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="10.0" velocity="2.0"/>
  </joint>
  <link name="upper"/>
  <joint name="wheel" type="continuous">
    <parent link="upper"/><child link="wheel_link"/>
    <axis xyz="0 0 1"/>
    <limit effort="10.0" velocity="4.0"/>
  </joint>
  <link name="wheel_link"/>
  <joint name="slider" type="prismatic">
    <parent link="upper"/><child link="slide_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="0.0" upper="0.2" effort="10.0" velocity="0.1"/>
  </joint>
  <link name="slide_link"/>
</robot>
"""


@pytest.fixture
def robot():
    return RobotModel.from_urdf(_URDF)


@pytest.fixture
def ri(robot):
    return FakeRobotInterface(robot)


def _index(robot, name):
    return [j.name for j in robot.joint_list].index(name)


def test_base_is_abstract(robot):
    with pytest.raises(TypeError):
        RobotInterfaceBase(robot)


def test_vector_goal_moves_whole_group(ri, robot):
    av = np.zeros(len(robot.joint_list))
    av[_index(robot, 'shoulder')] = 0.5
    av[_index(robot, 'slider')] = 0.1
    ri.angle_vector(av, 2.0)
    sent = ri.sent[-1]
    assert sent['joint_names'] == [j.name for j in robot.joint_list]
    assert sent['times'].tolist() == [2.0]
    assert ri.is_interpolating()
    assert ri.wait_interpolation() == [False]
    np.testing.assert_allclose(ri.potentio_vector(), av)


def test_dict_goal_moves_only_named_joints(ri, robot):
    ri.angle_vector({'shoulder': 0.5}, 1.0)
    assert ri.sent[-1]['joint_names'] == ['shoulder']
    ri.wait_interpolation()
    measured = ri.measured_joint_angles()
    assert measured['shoulder'] == pytest.approx(0.5)
    assert measured['slider'] == pytest.approx(0.0)


def test_vector_and_dict_give_same_trajectory(robot):
    a = FakeRobotInterface(robot)
    b = FakeRobotInterface(robot)
    av = np.array([0.3, 1.0, 0.05])
    order = [j.name for j in robot.joint_list]
    a.angle_vector(av, 2.0)
    b.angle_vector(dict(zip(order, av)), 2.0)
    np.testing.assert_allclose(a.sent[-1]['positions'],
                               b.sent[-1]['positions'])


def test_wrong_vector_length_is_refused(ri):
    with pytest.raises(ValueError):
        ri.angle_vector([0.1, 0.2])


def test_unknown_joint_names_are_refused(ri):
    with pytest.raises(KeyError):
        ri.angle_vector({'no_such_joint': 0.1})
    with pytest.raises(KeyError):
        ri.measured_joint_angles(['no_such_joint'])
    with pytest.raises(KeyError):
        ri.servo_on(['no_such_joint'])
    assert ri.sent == []


def test_joint_outside_controller_is_refused(robot):
    ri = FakeRobotInterface(robot, controllers={
        'arm': ['shoulder', 'slider'], 'base': ['wheel']})
    with pytest.raises(ValueError):
        ri.angle_vector({'wheel': 0.1}, controller_type='arm')
    ri.angle_vector({'wheel': 0.1}, 1.0, controller_type='base')
    assert ri.is_interpolating('base')
    assert not ri.is_interpolating('arm')
    with pytest.raises(KeyError):
        ri.angle_vector({'wheel': 0.1}, controller_type='legs')


def test_duration_from_velocity_limits(ri):
    # shoulder 0.5 rad at 2 rad/s -> 0.25 s; slider 0.1 m at 0.1 m/s -> 1 s
    ri.angle_vector({'shoulder': 0.5, 'slider': 0.1})
    assert ri.sent[-1]['times'][-1] == pytest.approx(5.0 * 1.0)
    ri.wait_interpolation()
    ri.angle_vector({'shoulder': -0.5}, time_scale=1.0)
    assert ri.sent[-1]['times'][-1] == pytest.approx(0.5)


def test_too_short_time_is_raised_to_velocity_limit(ri):
    ri.angle_vector({'shoulder': 1.0}, 0.1)
    assert ri.sent[-1]['times'][-1] == pytest.approx(0.5)


def test_time_scale_below_one_is_refused(ri):
    with pytest.raises(ValueError):
        ri.angle_vector({'shoulder': 0.5}, time_scale=0.5)


def test_continuous_joint_takes_short_way(ri):
    ri.set_joint_positions({'wheel': 3.0})
    # 3.0 -> -3.0 is 0.28 rad the short way round, not 6 rad.
    durations = ri.angle_vector_duration(
        {'wheel': 3.0}, {'wheel': -3.0}, return_joint_names=True)
    assert durations[0][0] == pytest.approx(
        (2 * np.pi - 6.0) / 4.0)


def test_measured_state_outside_limits_is_reported_not_clamped(ri, robot):
    ri.set_joint_positions({'shoulder': 1.3})
    av = ri.angle_vector()
    assert av[_index(robot, 'shoulder')] == pytest.approx(1.3)
    assert ri.potentio_vector()[_index(robot, 'shoulder')] \
        == pytest.approx(1.3)
    report = ri.outside_limits()
    assert list(report) == ['shoulder']
    assert report['shoulder'] == pytest.approx((1.3, -1.0, 1.0))
    assert ri.outside_limits(margin=0.5) == {}
    # The model clamps; the interface must not.
    robot.angle_vector(av)
    assert robot.angle_vector()[_index(robot, 'shoulder')] \
        == pytest.approx(1.0)


def test_replaying_measured_state_does_not_jump(ri, robot):
    ri.set_joint_positions({'shoulder': 1.3})
    ri.angle_vector(ri.angle_vector(), 1.0)
    sent = ri.sent[-1]
    i = sent['joint_names'].index('shoulder')
    assert sent['positions'][0, i] == pytest.approx(1.3)


def test_dict_goal_does_not_need_other_joints(ri):
    # A dict goal moves only what it names, so an out-of-limit joint
    # elsewhere is neither clamped nor commanded.
    ri.set_joint_positions({'slider': 0.5})
    ri.angle_vector({'shoulder': 0.2}, 1.0)
    assert ri.sent[-1]['joint_names'] == ['shoulder']
    ri.wait_interpolation()
    assert ri.measured_joint_angles()['slider'] == pytest.approx(0.5)


def test_outside_limits_of_a_command(ri, robot):
    assert list(ri.outside_limits({'slider': 0.3})) == ['slider']
    av = np.zeros(len(robot.joint_list))
    assert ri.outside_limits(av) == {}


def test_wait_with_timeout_and_cancel(ri):
    ri.angle_vector({'shoulder': 1.0}, 2.0)
    assert ri.wait_interpolation(timeout=1.0) == [True]
    middle = ri.measured_joint_angles()['shoulder']
    assert 0.0 < middle < 1.0
    ri.cancel_angle_vector()
    assert not ri.is_interpolating()
    ri.advance(5.0)
    assert ri.angle_vector()[_index(ri.robot, 'shoulder')] == pytest.approx(middle)


def test_sequence_passes_via_points(ri):
    avs = [{'shoulder': 0.4}, {'shoulder': 0.8}, {'shoulder': 0.2}]
    ri.angle_vector_sequence(avs, [1.0, 1.0, 1.0])
    sent = ri.sent[-1]
    np.testing.assert_allclose(sent['times'], [1.0, 2.0, 3.0])
    # Rising then turning back: mean velocity at the first via point,
    # zero where the joint reverses, zero at the end.
    np.testing.assert_allclose(sent['velocities'][:, 0], [0.4, 0.0, 0.0])
    ri.advance(1.0)
    assert ri.angle_vector()[_index(ri.robot, 'shoulder')] == pytest.approx(0.4)
    ri.wait_interpolation()
    assert ri.angle_vector()[_index(ri.robot, 'shoulder')] == pytest.approx(0.2)


def test_sequence_needs_same_joints(ri):
    with pytest.raises(ValueError):
        ri.angle_vector_sequence([{'shoulder': 0.4}, {'slider': 0.1}])


def test_wall_clock_wait_uses_default_polling(robot):
    ri = FakeRobotInterface(robot, clock=time.monotonic)
    ri.poll_period = 0.005
    ri.angle_vector({'shoulder': 0.1}, 0.05)
    assert ri.is_interpolating()
    assert ri.wait_interpolation() == [False]
    assert ri.measured_joint_angles()['shoulder'] == pytest.approx(0.1)


def test_servo_hooks(robot):
    ri = FakeRobotInterface(robot)
    ri.servo_off(['shoulder'])
    assert ri.servo_states['shoulder'] is False
    ri.servo_on()
    assert all(ri.servo_states.values())

    no_servo = FakeRobotInterface(robot, servo=False)
    with pytest.raises(NotImplementedError):
        no_servo.servo_on()
    with pytest.raises(NotImplementedError):
        no_servo.servo_off()


def test_unmeasured_joint_is_nan_and_cannot_start_a_motion(robot):

    class PartialState(FakeRobotInterface):
        def _read_joint_state(self):
            state = super(PartialState, self)._read_joint_state()
            state.pop('slider')
            return state

    ri = PartialState(robot)
    assert np.isnan(ri.potentio_vector()[_index(robot, 'slider')])
    with pytest.raises(ValueError):
        ri.angle_vector({'slider': 0.1}, 1.0)
    ri.angle_vector({'shoulder': 0.1}, 1.0)


def test_interface_never_writes_the_model(ri, robot):
    before = robot.angle_vector().copy()
    ri.angle_vector({'shoulder': 0.7}, 1.0)
    ri.wait_interpolation()
    np.testing.assert_array_equal(robot.angle_vector(), before)

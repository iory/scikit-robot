from skrobot.coordinates import Coordinates
from skrobot.models import PR2


def _solve(robot, pos, caplog, **kwargs):
    caplog.clear()
    with caplog.at_level('WARNING', logger='skrobot.model.robot_model'):
        result = robot.inverse_kinematics(
            Coordinates(pos=pos), move_target=robot.rarm_end_coords,
            link_list=robot.rarm.link_list, **kwargs)
    return result, [r.getMessage() for r in caplog.records
                    if 'inverse_kinematics failed' in r.getMessage()]


def test_unreachable_target_warns_with_remaining_error(caplog):
    robot = PR2()
    robot.reset_manip_pose()
    av0 = robot.angle_vector().copy()
    result, messages = _solve(robot, [2.0, -0.2, 1.0], caplog)
    assert result is False
    assert len(messages) == 1
    assert 'rarm_end_coords: position error' in messages[0]
    assert 'reverted to the initial pose' in messages[0]
    assert (robot.angle_vector() == av0).all()


def test_failure_warning_can_be_silenced(caplog):
    robot = PR2()
    robot.reset_manip_pose()
    result, messages = _solve(robot, [2.0, -0.2, 1.0], caplog,
                              warn_if_fail=False)
    assert result is False
    assert messages == []


def test_reachable_target_does_not_warn(caplog):
    robot = PR2()
    robot.reset_manip_pose()
    result, messages = _solve(robot, [0.6, -0.2, 0.8], caplog)
    assert result is not False
    assert messages == []

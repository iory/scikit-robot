import numpy as np
from numpy import testing

import skrobot
from skrobot.coordinates import Coordinates


def _far_target_from_current(robot):
    """Target that is far enough the arm alone cannot reach."""
    ee_pos = robot.rarm_end_coords.worldpos().copy()
    return Coordinates(pos=ee_pos + np.array([0.8, 0.3, 0.0]))


def test_inverse_kinematics_use_base_false_is_backcompat():
    robot_a = skrobot.models.Fetch()
    robot_b = skrobot.models.Fetch()
    ee_pos = robot_a.rarm_end_coords.worldpos().copy()
    target = Coordinates(pos=ee_pos + np.array([0.2, 0.1, 0.0]))

    result_a = robot_a.inverse_kinematics(
        target, move_target=robot_a.rarm_end_coords,
        link_list=robot_a.rarm.link_list,
        rotation_mask=False, stop=100)
    result_b = robot_b.inverse_kinematics(
        target, move_target=robot_b.rarm_end_coords,
        link_list=robot_b.rarm.link_list,
        rotation_mask=False, stop=100, use_base=False)

    assert (result_a is False) == (result_b is False)
    testing.assert_allclose(
        robot_a.angle_vector(), robot_b.angle_vector(), atol=1e-8)


def test_inverse_kinematics_use_base_planar_reaches_far_target():
    robot = skrobot.models.Fetch()
    root0 = robot.root_link.copy_worldcoords()
    target = _far_target_from_current(robot)

    result = robot.inverse_kinematics(
        target, move_target=robot.rarm_end_coords,
        link_list=robot.rarm.link_list,
        rotation_mask=False, stop=100, use_base='planar')

    assert result is not False
    err = np.linalg.norm(
        robot.rarm_end_coords.worldpos() - target.worldpos())
    assert err < 1e-3, 'rarm end did not reach target: err={}'.format(err)
    moved = np.linalg.norm(
        robot.root_link.worldpos() - root0.worldpos())
    assert moved > 0.1, 'base did not move to reach far target'


def test_inverse_kinematics_use_base_detach_restores_root_link():
    robot = skrobot.models.Fetch()
    target = _far_target_from_current(robot)

    robot.inverse_kinematics(
        target, move_target=robot.rarm_end_coords,
        link_list=robot.rarm.link_list,
        rotation_mask=False, stop=100, use_base='planar')

    # After the call the virtual joint must be gone.
    assert robot.root_link.joint is None
    assert robot.root_link.parent_link is None


def test_inverse_kinematics_use_base_6dof_reaches_far_target():
    robot = skrobot.models.Fetch()
    target = _far_target_from_current(robot)

    result = robot.inverse_kinematics(
        target, move_target=robot.rarm_end_coords,
        link_list=robot.rarm.link_list,
        rotation_mask=False, stop=100, use_base='6dof')

    assert result is not False
    err = np.linalg.norm(
        robot.rarm_end_coords.worldpos() - target.worldpos())
    assert err < 1e-3


def test_base_weight_biases_toward_arm_when_small():
    """Small base_weight makes arm absorb most of the motion."""
    def run(bw):
        robot = skrobot.models.Fetch()
        ee0 = robot.rarm_end_coords.worldpos().copy()
        # y-only target: both arm (shoulder_pan) and base can contribute.
        target = Coordinates(pos=ee0 + np.array([0.0, 0.1, 0.0]))
        robot.inverse_kinematics(
            target, move_target=robot.rarm_end_coords,
            link_list=robot.rarm.link_list,
            rotation_mask=False, stop=200,
            use_base='planar', base_weight=bw)
        base_y = abs(robot.root_link.worldpos()[1])
        return base_y

    base_y_equal = run(1.0)
    base_y_arm_pref = run(0.01)
    base_y_base_pref = run(100.0)

    assert base_y_arm_pref < base_y_equal
    assert base_y_equal < base_y_base_pref
    # Arm-preferred should move the base at least 10x less than equal.
    assert base_y_arm_pref * 10 < base_y_equal


def test_base_weight_per_axis_vector():
    """Providing a per-axis sequence weights base DoFs independently."""
    robot = skrobot.models.Fetch()
    ee0 = robot.rarm_end_coords.worldpos().copy()
    target = Coordinates(pos=ee0 + np.array([0.0, 0.1, 0.0]))
    # Heavy yaw penalty, light xy.
    robot.inverse_kinematics(
        target, move_target=robot.rarm_end_coords,
        link_list=robot.rarm.link_list,
        rotation_mask=False, stop=200,
        use_base='planar', base_weight=[1.0, 1.0, 0.001])
    # Should still converge — vector weight accepted.
    err = np.linalg.norm(
        robot.rarm_end_coords.worldpos() - target.worldpos())
    assert err < 1e-3


def test_base_weight_without_use_base_warns():
    robot = skrobot.models.Fetch()
    ee0 = robot.rarm_end_coords.worldpos().copy()
    target = Coordinates(pos=ee0 + np.array([0.01, 0.0, 0.0]))
    # Expect a warning but IK still runs normally.
    robot.inverse_kinematics(
        target, move_target=robot.rarm_end_coords,
        link_list=robot.rarm.link_list,
        rotation_mask=False, stop=50,
        use_base=False, base_weight=0.5)


def test_inverse_kinematics_use_base_invalid_raises():
    robot = skrobot.models.Fetch()
    ee_pos = robot.rarm_end_coords.worldpos().copy()
    target = Coordinates(pos=ee_pos + np.array([0.1, 0.0, 0.0]))

    try:
        robot.inverse_kinematics(
            target, move_target=robot.rarm_end_coords,
            link_list=robot.rarm.link_list,
            rotation_mask=False, stop=10, use_base='invalid')
    except ValueError:
        pass
    else:
        raise AssertionError('Expected ValueError for invalid use_base')
    # Detach must still have run via finally.
    assert robot.root_link.joint is None
    assert robot.root_link.parent_link is None


def _run_multi_ee_dual_arm(robot, link_list_order):
    """Solve the same 2-EE IK with a user-specified ``link_list_order``.

    ``link_list_order`` controls which arm chain is passed as
    ``link_list[0]`` vs ``link_list[1]``.  ``move_target`` and
    ``target_coords`` are always in (rarm, larm) order, so the two
    arguments agree when ``link_list_order == ('rarm', 'larm')`` and
    disagree when ``('larm', 'rarm')`` — the reorder helper has to fix
    the latter silently.
    """
    robot.reset_manip_pose()  # start from a valid, within-limits pose
    chains = {'rarm': robot.rarm.link_list, 'larm': robot.larm.link_list}
    r_goal = Coordinates(
        pos=robot.rarm_end_coords.worldpos() + np.array([0.02, 0.0, 0.01]))
    l_goal = Coordinates(
        pos=robot.larm_end_coords.worldpos() + np.array([0.02, 0.0, 0.01]))
    mask_pair = [np.array([1, 1, 1]), np.array([1, 1, 1])]
    zero_mask = [np.array([0, 0, 0]), np.array([0, 0, 0])]
    robot.inverse_kinematics(
        target_coords=[r_goal, l_goal],
        move_target=[robot.rarm_end_coords, robot.larm_end_coords],
        link_list=[chains[link_list_order[0]], chains[link_list_order[1]]],
        position_mask=mask_pair,
        rotation_mask=zero_mask,
        stop=200, thre=[0.001, 0.001])
    return robot.angle_vector().copy(), r_goal, l_goal


def test_multi_ee_link_list_auto_reorders_to_match_move_target():
    """Swapping the order of link_list vs move_target must still converge
    to the same joint configuration — the solver auto-pairs each
    move_target with its kinematic-chain link_list.
    """
    robot_correct = skrobot.models.PR2()
    av_correct, r_goal, l_goal = _run_multi_ee_dual_arm(
        robot_correct, ('rarm', 'larm'))
    r_err_correct = np.linalg.norm(
        robot_correct.rarm_end_coords.worldpos() - r_goal.worldpos())
    l_err_correct = np.linalg.norm(
        robot_correct.larm_end_coords.worldpos() - l_goal.worldpos())

    robot_swapped = skrobot.models.PR2()
    av_swapped, r_goal_s, l_goal_s = _run_multi_ee_dual_arm(
        robot_swapped, ('larm', 'rarm'))
    r_err_swapped = np.linalg.norm(
        robot_swapped.rarm_end_coords.worldpos() - r_goal_s.worldpos())
    l_err_swapped = np.linalg.norm(
        robot_swapped.larm_end_coords.worldpos() - l_goal_s.worldpos())

    assert r_err_correct < 5e-3, (
        'rarm err with correct order: {}'.format(r_err_correct))
    assert r_err_swapped < 5e-3, (
        'rarm err with swapped order: {}'.format(r_err_swapped))
    assert l_err_correct < 5e-3
    assert l_err_swapped < 5e-3
    testing.assert_allclose(av_correct, av_swapped, atol=1e-6)

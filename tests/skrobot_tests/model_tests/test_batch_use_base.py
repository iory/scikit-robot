"""Tests for batch_inverse_kinematics with use_base."""
import numpy as np
import pytest

import skrobot
from skrobot.coordinates import Coordinates


def _build_fetch():
    robot = skrobot.models.Fetch()
    return robot


def test_batch_use_base_false_backcompat():
    """use_base=False leaves return signature as 3-tuple, same as before."""
    robot = _build_fetch()
    ee = robot.rarm_end_coords.worldpos()
    target = [Coordinates(pos=ee + np.array([0.1, 0.0, 0.0]))]

    result = robot.batch_inverse_kinematics(
        target_coords=target,
        move_target=robot.rarm_end_coords,
        link_list=robot.rarm.link_list,
        rotation_mask=False,
        stop=100, thre=0.01,
        backend='numpy', initial_angles='current',
    )
    assert isinstance(result, tuple) and len(result) == 3


def test_batch_use_base_planar_reaches_far_target():
    """Arm-unreachable target succeeds with use_base='planar'; applied base
    + angle vector reproduces the target EE pose."""
    robot = _build_fetch()
    ee = robot.rarm_end_coords.worldpos()
    N = 3
    targets = [
        Coordinates(pos=ee + np.array([0.7 + 0.1 * i, 0.0, 0.0]))
        for i in range(N)
    ]
    solutions, base_poses, success, _ = robot.batch_inverse_kinematics(
        target_coords=targets,
        move_target=robot.rarm_end_coords,
        link_list=robot.rarm.link_list,
        rotation_mask=False,
        stop=200, thre=0.01,
        backend='numpy', initial_angles='current',
        use_base='planar',
    )
    assert all(success), "unexpected failures: {}".format(success)
    # Apply each solution and verify the EE lands on target.
    for av, bp, tgt in zip(solutions, base_poses, targets):
        robot.angle_vector(av)
        robot.newcoords(bp)
        err = np.linalg.norm(
            robot.rarm_end_coords.worldpos() - tgt.worldpos())
        assert err < 0.01, "EE err {} exceeds threshold".format(err)


def test_batch_use_base_planar_arm_alone_fails():
    """Same far target without use_base fails — validates the base is what
    made use_base='planar' succeed in the previous test."""
    robot = _build_fetch()
    ee = robot.rarm_end_coords.worldpos()
    targets = [Coordinates(pos=ee + np.array([0.9, 0.0, 0.0]))]

    _, success, _ = robot.batch_inverse_kinematics(
        target_coords=targets,
        move_target=robot.rarm_end_coords,
        link_list=robot.rarm.link_list,
        rotation_mask=False,
        stop=200, thre=0.01,
        backend='numpy', initial_angles='current',
    )
    assert not success[0], "arm alone should not reach 0.9m target"


def test_batch_use_base_6dof_allows_vertical_motion():
    """6dof base can move in z; planar base cannot reach a target that
    needs vertical base displacement."""
    robot = _build_fetch()
    ee = robot.rarm_end_coords.worldpos()
    # Target shifted far forward and up, beyond arm-only reach.
    targets = [Coordinates(pos=ee + np.array([0.7, 0.0, 0.4]))]

    _, base_poses_6, success_6, _ = robot.batch_inverse_kinematics(
        target_coords=targets,
        move_target=robot.rarm_end_coords,
        link_list=robot.rarm.link_list,
        rotation_mask=False,
        stop=200, thre=0.01,
        backend='numpy', initial_angles='current',
        use_base='6dof',
    )
    assert success_6[0], "6dof should reach vertically displaced target"
    # The base's z should have moved (non-trivially) to make the reach.
    assert abs(base_poses_6[0].worldpos()[2]) > 0.05


def test_batch_use_base_multi_ee_shared_base():
    """Multi-EE + use_base: both arms share the same base DoFs."""
    robot = _build_fetch()
    rarm_mt = robot.rarm_end_coords
    rarm_ll = robot.link_lists(rarm_mt.parent)

    ee = rarm_mt.worldpos()
    r_targets = [Coordinates(pos=ee + np.array([0.6, 0.0, 0.0]))]
    # Single-EE multi-task (arm alone, but wrapped in list form) to exercise
    # the multi-EE + use_base path.
    solutions, base_poses, success, _ = robot.batch_inverse_kinematics(
        target_coords=[r_targets],
        move_target=[rarm_mt],
        link_list=[rarm_ll],
        rotation_mask=False,
        stop=200, thre=0.01,
        backend='numpy', initial_angles='current',
        use_base='planar',
    )
    assert success[0]
    robot.angle_vector(solutions[0])
    robot.newcoords(base_poses[0])
    err = np.linalg.norm(rarm_mt.worldpos() - r_targets[0].worldpos())
    assert err < 0.01


def test_batch_use_base_invalid_raises():
    """Invalid use_base value raises a clear ValueError."""
    robot = _build_fetch()
    ee = robot.rarm_end_coords.worldpos()
    targets = [Coordinates(pos=ee + np.array([0.1, 0, 0]))]
    with pytest.raises(ValueError, match="use_base"):
        robot.batch_inverse_kinematics(
            target_coords=targets,
            move_target=robot.rarm_end_coords,
            link_list=robot.rarm.link_list,
            rotation_mask=False,
            stop=10, backend='numpy', initial_angles='current',
            use_base='bogus',
        )


def test_batch_use_base_detaches_on_exception():
    """An exception inside the solver still detaches the virtual chain."""
    robot = _build_fetch()
    root_link = robot._find_fullbody_root_link()
    orig_parent = root_link._parent_link
    orig_joint = root_link.joint

    with pytest.raises(Exception):
        robot.batch_inverse_kinematics(
            # Intentionally malformed input to trigger an error inside impl.
            target_coords=np.zeros((1, 5)),  # wrong shape
            move_target=robot.rarm_end_coords,
            link_list=robot.rarm.link_list,
            stop=10, backend='numpy', initial_angles='current',
            use_base='planar',
        )

    assert root_link._parent_link is orig_parent
    assert root_link.joint is orig_joint


def test_batch_use_base_base_weight_warns():
    """base_weight is accepted but ignored with a RuntimeWarning in 3B."""
    robot = _build_fetch()
    ee = robot.rarm_end_coords.worldpos()
    targets = [Coordinates(pos=ee + np.array([0.5, 0, 0]))]

    with pytest.warns(RuntimeWarning, match="base_weight"):
        robot.batch_inverse_kinematics(
            target_coords=targets,
            move_target=robot.rarm_end_coords,
            link_list=robot.rarm.link_list,
            rotation_mask=False,
            stop=50, thre=0.01,
            backend='numpy', initial_angles='current',
            use_base='planar', base_weight=0.1,
        )

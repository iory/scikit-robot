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


def test_batch_use_base_does_not_grow_solver_cache():
    """Repeated use_base calls must not leave stale entries in the solver
    caches; every call attaches fresh virtual Link/Joint objects whose
    id()s are unique and would otherwise accumulate forever."""
    robot = _build_fetch()
    ee = robot.rarm_end_coords.worldpos()
    targets = [Coordinates(pos=ee + np.array([0.3, 0.0, 0.0]))]

    def _cache_sizes():
        return (
            len(getattr(robot, '_batch_ik_solver_cache', {}) or {}),
            len(getattr(
                robot, '_batch_ik_multi_ee_solver_cache', {}) or {}),
        )

    before = _cache_sizes()
    for _ in range(3):
        robot.batch_inverse_kinematics(
            target_coords=targets,
            move_target=robot.rarm_end_coords,
            link_list=robot.rarm.link_list,
            rotation_mask=False,
            stop=30, thre=0.02,
            backend='numpy', initial_angles='current',
            use_base='planar',
        )
    after = _cache_sizes()
    assert after == before, (
        "use_base batch IK leaked solver cache entries: {} -> {}".format(
            before, after))


def _pr2_base_displacement(base_weight, dx=0.3):
    """Solve a +x target on PR2 rarm with the given base_weight and
    report how much the base moved. PR2's rarm is flexible enough from
    its reset pose that arm vs. base is genuinely weight-sensitive;
    Fetch's default pose tucks the arm and skews this test."""
    robot = skrobot.models.PR2()
    robot.reset_pose()
    ee = robot.rarm.end_coords.worldpos()
    targets = [Coordinates(pos=ee + np.array([dx, 0.0, 0.0]))]
    _, base_poses, success, _ = robot.batch_inverse_kinematics(
        target_coords=targets,
        move_target=robot.rarm.end_coords,
        link_list=robot.link_lists(robot.rarm.end_coords.parent),
        rotation_mask=False,
        stop=500, thre=0.005,
        backend='numpy', initial_angles='current',
        use_base='planar', base_weight=base_weight,
    )
    return base_poses[0].worldpos(), bool(success[0])


def test_batch_base_weight_shifts_effort():
    """Higher base_weight => base moves more; lower base_weight => less."""
    pos_heavy, ok_heavy = _pr2_base_displacement(base_weight=0.01)
    pos_light, ok_light = _pr2_base_displacement(base_weight=10.0)
    assert ok_heavy and ok_light, (
        "both solves should succeed; heavy={}, light={}".format(
            ok_heavy, ok_light))
    # A "heavy" base (weight << 1) makes the base barely move; a "light"
    # base (weight >> 1) makes the base do most of the work. Expect a
    # sizeable gap on the same 30 cm target.
    assert pos_light[0] > pos_heavy[0] + 0.1, (
        "expected light-base to travel further than heavy-base; "
        "heavy_x={:.3f}, light_x={:.3f}".format(pos_heavy[0], pos_light[0]))


def test_batch_base_weight_per_axis_discourages_yaw():
    """Per-axis base_weight [1, 1, 0.01] discourages base yaw rotation."""
    robot = _build_fetch()
    ee = robot.rarm_end_coords.worldpos()
    # Off-axis target makes yaw attractive to the solver.
    targets = [Coordinates(pos=ee + np.array([0.4, 0.3, 0.0]))]
    solutions, base_poses, success, _ = robot.batch_inverse_kinematics(
        target_coords=targets,
        move_target=robot.rarm_end_coords,
        link_list=robot.rarm.link_list,
        rotation_mask=False,
        stop=200, thre=0.01,
        backend='numpy', initial_angles='current',
        use_base='planar', base_weight=[1.0, 1.0, 0.01],
    )
    assert success[0]
    # Extract yaw from base pose rotation (planar → rotation is Rz(yaw)).
    rot = base_poses[0].worldrot()
    yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    assert abs(yaw) < 0.05, (
        "expected near-zero yaw under heavy yaw weighting, got {}".format(
            yaw))


def test_batch_base_weight_jax_parity_with_numpy():
    """JAX backend produces equivalent base_weight behavior to NumPy."""
    try:
        import jax  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("JAX not installed")

    def _run(backend):
        robot = skrobot.models.PR2()
        robot.reset_pose()
        ee = robot.rarm.end_coords.worldpos()
        targets = [Coordinates(pos=ee + np.array([0.3, 0.0, 0.0]))]
        _, base_poses, success, _ = robot.batch_inverse_kinematics(
            target_coords=targets,
            move_target=robot.rarm.end_coords,
            link_list=robot.link_lists(robot.rarm.end_coords.parent),
            rotation_mask=False,
            stop=500, thre=0.005,
            backend=backend, initial_angles='current',
            use_base='planar', base_weight=0.1,
        )
        return base_poses[0].worldpos(), bool(success[0])

    pos_np, ok_np = _run('numpy')
    pos_jx, ok_jx = _run('jax')
    assert ok_np and ok_jx
    # JAX defaults to float32 so expect mm-scale drift at most.
    np.testing.assert_allclose(pos_jx, pos_np, atol=0.01)


def test_batch_base_weight_rejects_nonpositive():
    """base_weight <= 0 raises a clear ValueError."""
    robot = _build_fetch()
    ee = robot.rarm_end_coords.worldpos()
    targets = [Coordinates(pos=ee + np.array([0.4, 0, 0]))]
    with pytest.raises(ValueError, match="base_weight"):
        robot.batch_inverse_kinematics(
            target_coords=targets,
            move_target=robot.rarm_end_coords,
            link_list=robot.rarm.link_list,
            rotation_mask=False,
            stop=30, backend='numpy', initial_angles='current',
            use_base='planar', base_weight=0.0,
        )

"""Tests for multi-end-effector batch inverse kinematics."""
import numpy as np
import pytest

import skrobot
from skrobot.coordinates import Coordinates


def _build_pr2():
    robot = skrobot.models.PR2()
    robot.reset_pose()
    return robot


def test_multi_ee_single_task_matches_single_ee():
    """Multi-EE with a single task is numerically identical to single-EE."""
    robot_a = _build_pr2()
    robot_b = _build_pr2()

    rarm_mt_a = robot_a.rarm.end_coords
    rarm_mt_b = robot_b.rarm.end_coords
    rarm_ll_a = robot_a.link_lists(rarm_mt_a.parent)
    rarm_ll_b = robot_b.link_lists(rarm_mt_b.parent)

    pos0 = rarm_mt_a.worldpos()
    rot0 = rarm_mt_a.worldrot()

    N = 3
    targets_single = [
        Coordinates(pos=pos0 + np.array([0.01 * i, 0.0, 0.0]), rot=rot0)
        for i in range(1, N + 1)
    ]
    # Same targets, wrapped for multi-EE (1 task).
    targets_multi = [list(targets_single)]

    sol_a, suc_a, _ = robot_a.batch_inverse_kinematics(
        target_coords=targets_single,
        move_target=rarm_mt_a, link_list=rarm_ll_a,
        stop=100, thre=0.003, rthre=np.deg2rad(3.0),
        backend='numpy', initial_angles='current',
    )
    sol_b, suc_b, _ = robot_b.batch_inverse_kinematics(
        target_coords=targets_multi,
        move_target=[rarm_mt_b], link_list=[rarm_ll_b],
        stop=100, thre=0.003, rthre=np.deg2rad(3.0),
        backend='numpy', initial_angles='current',
    )

    assert suc_a == suc_b
    for sa, sb in zip(sol_a, sol_b):
        np.testing.assert_allclose(sa, sb, atol=1e-10)


def test_multi_ee_pr2_dual_arm_reachable_targets():
    """PR2 dual-arm batch with small symmetric targets converges."""
    robot = _build_pr2()
    rarm_mt = robot.rarm.end_coords
    larm_mt = robot.larm.end_coords
    rarm_ll = robot.link_lists(rarm_mt.parent)
    larm_ll = robot.link_lists(larm_mt.parent)

    r_pos0 = rarm_mt.worldpos()
    r_rot0 = rarm_mt.worldrot()
    l_pos0 = larm_mt.worldpos()
    l_rot0 = larm_mt.worldrot()

    N = 3
    rarm_targets = [
        Coordinates(pos=r_pos0 + np.array([0.005 * (i + 1), 0.0, 0.0]),
                    rot=r_rot0) for i in range(N)
    ]
    larm_targets = [
        Coordinates(pos=l_pos0 - np.array([0.005 * (i + 1), 0.0, 0.0]),
                    rot=l_rot0) for i in range(N)
    ]

    solutions, success_flags, _ = robot.batch_inverse_kinematics(
        target_coords=[rarm_targets, larm_targets],
        move_target=[rarm_mt, larm_mt], link_list=[rarm_ll, larm_ll],
        stop=200, thre=0.005, rthre=np.deg2rad(3.0),
        backend='numpy', initial_angles='current',
    )

    # All should succeed for these conservative targets.
    assert all(success_flags), "expected all successes, got {}".format(
        success_flags)

    # Verify by applying each solution and measuring EE error.
    for i, sol in enumerate(solutions):
        robot.angle_vector(sol)
        r_err = np.linalg.norm(
            rarm_mt.worldpos() - rarm_targets[i].worldpos())
        l_err = np.linalg.norm(
            larm_mt.worldpos() - larm_targets[i].worldpos())
        assert r_err < 0.005, "rarm pose {} err {}".format(i, r_err)
        assert l_err < 0.005, "larm pose {} err {}".format(i, l_err)


def test_multi_ee_respects_shared_joints():
    """Shared torso joint is solved jointly across tasks (union space)."""
    robot = _build_pr2()
    rarm_mt = robot.rarm.end_coords
    larm_mt = robot.larm.end_coords
    rarm_ll = robot.link_lists(rarm_mt.parent)
    larm_ll = robot.link_lists(larm_mt.parent)

    # Construct the solver directly to inspect the union joint set.
    from skrobot.kinematics.differentiable import create_batch_ik_solver
    solver = create_batch_ik_solver(
        robot, [rarm_ll, larm_ll], [rarm_mt, larm_mt],
        backend_name='numpy',
    )

    # The two tasks share the torso_lift_joint (and any joint above it).
    rarm_mapping = set(solver.union_info['task_mappings'][0].tolist())
    larm_mapping = set(solver.union_info['task_mappings'][1].tolist())
    shared = rarm_mapping & larm_mapping
    assert len(shared) > 0, "expected at least one shared joint"
    # Total union size < sum of per-task sizes when there is overlap.
    total = (len(solver.union_info['task_mappings'][0])
             + len(solver.union_info['task_mappings'][1]))
    assert solver.union_n_opt < total


def test_multi_ee_task_weights_bias_effort():
    """Task weights shift effort distribution between tasks."""
    robot_a = _build_pr2()
    robot_b = _build_pr2()

    def _run(robot, weights):
        rarm_mt = robot.rarm.end_coords
        larm_mt = robot.larm.end_coords
        rarm_ll = robot.link_lists(rarm_mt.parent)
        larm_ll = robot.link_lists(larm_mt.parent)
        r_pos0 = rarm_mt.worldpos()
        r_rot0 = rarm_mt.worldrot()
        l_pos0 = larm_mt.worldpos()
        l_rot0 = larm_mt.worldrot()

        # One batch element. Easy rarm target, infeasible larm target (too
        # far). Under very different weights the easy task's accuracy vs.
        # the hard task's accuracy tradeoff changes.
        r_targets = [Coordinates(pos=r_pos0 + np.array([0.01, 0.0, 0.0]),
                                 rot=r_rot0)]
        l_targets = [Coordinates(pos=l_pos0 + np.array([-0.30, 0.0, 0.0]),
                                 rot=l_rot0)]
        solutions, _, _ = robot.batch_inverse_kinematics(
            target_coords=[r_targets, l_targets],
            move_target=[rarm_mt, larm_mt], link_list=[rarm_ll, larm_ll],
            stop=200, thre=0.001, rthre=np.deg2rad(3.0),
            backend='numpy', initial_angles='current',
            task_weights=weights,
        )
        robot.angle_vector(solutions[0])
        r_err = np.linalg.norm(rarm_mt.worldpos() - r_targets[0].worldpos())
        l_err = np.linalg.norm(larm_mt.worldpos() - l_targets[0].worldpos())
        return r_err, l_err

    # rarm-heavy: rarm error should be smaller than with balanced weights.
    r_err_heavy, _ = _run(robot_a, [100.0, 1.0])
    r_err_balanced, _ = _run(robot_b, [1.0, 1.0])
    assert r_err_heavy <= r_err_balanced + 1e-6, (
        "expected rarm-heavy weighting to reduce rarm error; "
        "heavy={}, balanced={}".format(r_err_heavy, r_err_balanced))


def test_multi_ee_input_validation():
    """Shape mismatches raise clear ValueErrors."""
    robot = _build_pr2()
    rarm_mt = robot.rarm.end_coords
    larm_mt = robot.larm.end_coords
    rarm_ll = robot.link_lists(rarm_mt.parent)
    larm_ll = robot.link_lists(larm_mt.parent)

    rt = [Coordinates(pos=rarm_mt.worldpos() + np.array([0.01, 0, 0]),
                      rot=rarm_mt.worldrot())]
    lt = [Coordinates(pos=larm_mt.worldpos() - np.array([0.01, 0, 0]),
                      rot=larm_mt.worldrot()),
          Coordinates(pos=larm_mt.worldpos() - np.array([0.02, 0, 0]),
                      rot=larm_mt.worldrot())]

    # Inconsistent batch sizes across tasks.
    with pytest.raises(ValueError, match="Inconsistent batch sizes"):
        robot.batch_inverse_kinematics(
            target_coords=[rt, lt],
            move_target=[rarm_mt, larm_mt], link_list=[rarm_ll, larm_ll],
            stop=10, backend='numpy',
        )

    # Wrong number of move_targets for n_tasks > 1: detected as ambiguous.
    with pytest.raises(ValueError, match="Ambiguous batch_inverse_kinematics"):
        robot.batch_inverse_kinematics(
            target_coords=[rt, rt],
            move_target=rarm_mt, link_list=[rarm_ll, larm_ll],
            stop=10, backend='numpy',
        )


def test_multi_ee_legacy_single_ee_nested_link_list():
    """Legacy pattern ``link_list=[[...]]`` + scalar move_target stays single-EE."""
    robot = _build_pr2()
    rarm_mt = robot.rarm.end_coords
    rarm_ll = robot.link_lists(rarm_mt.parent)

    pos0 = rarm_mt.worldpos()
    rot0 = rarm_mt.worldrot()
    targets = [Coordinates(pos=pos0 + np.array([0.01, 0.0, 0.0]), rot=rot0)]

    # Should not raise "Ambiguous" and should solve as single-EE.
    solutions, success, _ = robot.batch_inverse_kinematics(
        target_coords=targets,
        move_target=rarm_mt, link_list=[rarm_ll],
        stop=100, thre=0.005, rthre=np.deg2rad(3.0),
        backend='numpy', initial_angles='current',
    )
    assert len(solutions) == 1


def test_multi_ee_inverted_shared_limits_errors():
    """Shared joint with inconsistent limits across tasks raises ValueError."""
    from skrobot.kinematics.differentiable import _build_union_joint_mapping
    from skrobot.kinematics.differentiable import extract_fk_parameters

    robot = _build_pr2()
    rarm_mt = robot.rarm.end_coords
    larm_mt = robot.larm.end_coords
    rarm_ll = robot.link_lists(rarm_mt.parent)
    larm_ll = robot.link_lists(larm_mt.parent)

    fk_rarm = extract_fk_parameters(robot, rarm_ll, rarm_mt)
    fk_larm = extract_fk_parameters(robot, larm_ll, larm_mt)
    # Corrupt shared torso limits so they invert under intersection.
    torso_idx_in_rarm = 1
    torso_idx_in_larm = 1
    fk_rarm['joint_limits_lower'][torso_idx_in_rarm] = 0.5
    fk_rarm['joint_limits_upper'][torso_idx_in_rarm] = 0.6
    fk_larm['joint_limits_lower'][torso_idx_in_larm] = 0.0
    fk_larm['joint_limits_upper'][torso_idx_in_larm] = 0.2

    with pytest.raises(ValueError, match="Inconsistent joint limits"):
        _build_union_joint_mapping(
            [rarm_ll, larm_ll], [fk_rarm, fk_larm])


def test_multi_ee_zero_weight_task_ignored():
    """A zero-weighted task does not gate success and does not steer solve."""
    robot = _build_pr2()
    rarm_mt = robot.rarm.end_coords
    larm_mt = robot.larm.end_coords
    rarm_ll = robot.link_lists(rarm_mt.parent)
    larm_ll = robot.link_lists(larm_mt.parent)

    # Reachable rarm target, infeasible larm target.
    rt = [Coordinates(pos=rarm_mt.worldpos() + np.array([0.01, 0, 0]),
                      rot=rarm_mt.worldrot())]
    lt = [Coordinates(pos=larm_mt.worldpos() + np.array([-0.80, 0, 0]),
                      rot=larm_mt.worldrot())]

    solutions, success_flags, _ = robot.batch_inverse_kinematics(
        target_coords=[rt, lt],
        move_target=[rarm_mt, larm_mt], link_list=[rarm_ll, larm_ll],
        stop=200, thre=0.003, rthre=np.deg2rad(3.0),
        backend='numpy', initial_angles='current',
        task_weights=[1.0, 0.0],
    )
    assert success_flags[0], (
        "rarm task should succeed when larm is zero-weighted, "
        "got success={}".format(success_flags))


def test_multi_ee_all_zero_weights_raises():
    """All-zero task weights raises a clear ValueError."""
    robot = _build_pr2()
    rarm_mt = robot.rarm.end_coords
    larm_mt = robot.larm.end_coords
    rarm_ll = robot.link_lists(rarm_mt.parent)
    larm_ll = robot.link_lists(larm_mt.parent)
    rt = [Coordinates(pos=rarm_mt.worldpos(), rot=rarm_mt.worldrot())]
    lt = [Coordinates(pos=larm_mt.worldpos(), rot=larm_mt.worldrot())]

    with pytest.raises(
            ValueError, match="no active, non-zero-weight tasks"):
        robot.batch_inverse_kinematics(
            target_coords=[rt, lt],
            move_target=[rarm_mt, larm_mt], link_list=[rarm_ll, larm_ll],
            stop=10, backend='numpy', initial_angles='current',
            task_weights=[0.0, 0.0],
        )


def test_multi_ee_per_task_bool_masks_internal():
    """Internal solver treats ``[True, False]`` as per-task when n_tasks=2.

    This exercises the _to_per_task rule directly on the solver since the
    public batch_inverse_kinematics path normalizes masks to a single 3-axis
    form before multi-EE dispatch; per-task mask propagation through the
    public API is out of scope here.
    """
    from skrobot.kinematics.differentiable import create_batch_ik_solver

    robot = _build_pr2()
    rarm_mt = robot.rarm.end_coords
    larm_mt = robot.larm.end_coords
    rarm_ll = robot.link_lists(rarm_mt.parent)
    larm_ll = robot.link_lists(larm_mt.parent)

    solver = create_batch_ik_solver(
        robot, [rarm_ll, larm_ll], [rarm_mt, larm_mt],
        backend_name='numpy',
    )

    N = 1
    r_targets = np.tile(rarm_mt.worldpos(), (N, 1)) \
        + np.array([[0.01, 0.0, 0.0]])
    l_targets = np.tile(larm_mt.worldpos(), (N, 1)) \
        + np.array([[-0.01, 0.0, 0.0]])
    r_rots = np.tile(rarm_mt.worldrot(), (N, 1, 1))
    l_rots = np.tile(larm_mt.worldrot(), (N, 1, 1))

    union_refs = solver.union_info['union_joint_refs']
    init = np.tile(
        np.array([j.joint_angle() for j in union_refs]), (N, 1))

    # Must not raise: per-task rotation_masks (True for rarm, False for larm).
    sol, _, _ = solver(
        [r_targets, l_targets], [r_rots, l_rots],
        initial_angles=init, max_iterations=50,
        rotation_masks=[True, False],
    )
    assert sol.shape == (N, solver.union_n_opt)


def test_multi_ee_jax_raises_until_ported():
    """Multi-EE on JAX backend currently falls back to NumPy with a warning."""
    robot = _build_pr2()
    rarm_mt = robot.rarm.end_coords
    larm_mt = robot.larm.end_coords
    rarm_ll = robot.link_lists(rarm_mt.parent)
    larm_ll = robot.link_lists(larm_mt.parent)

    rt = [Coordinates(pos=rarm_mt.worldpos() + np.array([0.005, 0, 0]),
                      rot=rarm_mt.worldrot())]
    lt = [Coordinates(pos=larm_mt.worldpos() - np.array([0.005, 0, 0]),
                      rot=larm_mt.worldrot())]

    with pytest.warns(RuntimeWarning, match="Multi-EE batch IK"):
        solutions, success, _ = robot.batch_inverse_kinematics(
            target_coords=[rt, lt],
            move_target=[rarm_mt, larm_mt], link_list=[rarm_ll, larm_ll],
            stop=100, thre=0.005, rthre=np.deg2rad(3.0),
            backend='jax', initial_angles='current',
        )
    assert len(solutions) == 1

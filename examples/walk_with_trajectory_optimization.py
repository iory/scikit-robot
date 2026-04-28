#!/usr/bin/env python
"""Whole-body walking on JAXON solved as a single trajectory optimisation.

Builds a ``TrajectoryProblem`` with both legs as parallel kinematic chains,
floating-base DoF, per-waypoint foot pose targets, and a CoG target, then
hands it to the JAXls solver. The optimiser jointly chooses the 12 leg
joint angles and the 6 base DoF at each waypoint without an external IK
loop.

Multiple walking segments can be chained with ``--segments``. Each
segment continues seamlessly from the pose where the previous one ended,
so the robot keeps walking instead of teleporting back to the initial
stance. Each segment specifies ``(n_cycles, dx, dy, dtheta_deg)`` where
``dx`` / ``dy`` are body-frame stride lengths per swing and
``dtheta_deg`` is the body-frame yaw advance per swing.

Dependencies: ``jax`` and ``jaxls`` (jaxls is not on PyPI). Install with::

    pip install jax
    pip install "git+https://github.com/brentyi/jaxls.git"

Usage::

    python examples/walk_with_trajectory_optimization.py
    python examples/walk_with_trajectory_optimization.py \\
        --segments 2,0.25,0,0 2,0.20,0,15 2,0.18,0.05,-15
    python examples/walk_with_trajectory_optimization.py --viewer trimesh
    python examples/walk_with_trajectory_optimization.py --no-interactive

Default viewer is ``pyrender``; ``trimesh`` and ``viser`` (browser) are
also supported. ``--no-interactive`` skips visualisation and only prints
timings and tracking errors.
"""

import argparse
import importlib.util
import os
import sys
import time


_INSTALL_HINT = (
    'This example requires the JAX and JAXls packages.\n'
    'JAXls is not on PyPI, so the usual ``pip install jaxls`` will not '
    'work.\n'
    'Install both with:\n'
    '    pip install jax\n'
    '    pip install "git+https://github.com/brentyi/jaxls.git"'
)


def _require_jax_and_jaxls():
    """Exit cleanly with install instructions if jax / jaxls is missing.

    Without this, an environment that lacks either package crashes with
    a bare ``ImportError`` from the first ``import jax`` below — not
    actionable for users running the example for the first time.
    """
    missing = [m for m in ('jax', 'jaxls')
               if importlib.util.find_spec(m) is None]
    if missing:
        print('Missing dependency: {}'.format(', '.join(missing)),
              file=sys.stderr)
        print(_INSTALL_HINT, file=sys.stderr)
        sys.exit(0)


_require_jax_and_jaxls()

import jax  # noqa: E402


jax.config.update('jax_enable_x64', True)

import numpy as np  # noqa: E402

import skrobot  # noqa: E402
from skrobot.coordinates import CascadedCoords  # noqa: E402
from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.models import JaxonJVRC  # noqa: E402
from skrobot.planner.trajectory_optimization import TrajectoryProblem  # noqa: E402
from skrobot.planner.trajectory_optimization.solvers import create_solver  # noqa: E402


def set_initial_stance(robot, knee_bend_deg=20.0):
    bend = np.deg2rad(knee_bend_deg)
    for side in ('LLEG', 'RLEG'):
        getattr(robot, '{}_JOINT2'.format(side)).joint_angle(-bend)
        getattr(robot, '{}_JOINT3'.format(side)).joint_angle(2.0 * bend)
        getattr(robot, '{}_JOINT4'.format(side)).joint_angle(-bend)


def floor_anchor(robot):
    """Translate the robot so the lowest foot vertex sits at z = 0."""
    foot_links = [robot.RLEG_LINK5, robot.LLEG_LINK5]
    lz = float('inf')
    for link in foot_links:
        meshes = link.visual_mesh
        if meshes is None:
            continue
        if not isinstance(meshes, (list, tuple)):
            meshes = [meshes]
        T = link.worldcoords().T()
        for m in meshes:
            if not hasattr(m, 'vertices') or len(m.vertices) == 0:
                continue
            wv = (T[:3, :3] @ np.asarray(m.vertices).T).T + T[:3, 3]
            lz = min(lz, float(wv[:, 2].min()))
    if not np.isfinite(lz):
        lz = -0.95
    robot.translate(np.array([0.0, 0.0, -lz]))


def rpy_to_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def yaw_from_rotation(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rpy_from_matrix(R):
    """Inverse of ``rpy_to_matrix`` (R = Rx Ry Rz, body-XYZ Euler)."""
    sy = float(R[0, 2])
    sy = max(-1.0, min(1.0, sy))
    ry = np.arcsin(sy)
    cy = np.cos(ry)
    if abs(cy) > 1e-6:
        rx = np.arctan2(-R[1, 2], R[2, 2])
        rz = np.arctan2(-R[0, 1], R[0, 0])
    else:
        rx = np.arctan2(R[2, 1], R[1, 1])
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def Rz_yaw(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def parse_segments(strings):
    """Parse 'n_cycles,dx,dy,dtheta_deg' segment strings."""
    segments = []
    for s in strings:
        parts = s.split(',')
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                "segment must be 'n_cycles,dx,dy,dtheta_deg' (got '{}')"
                .format(s))
        segments.append((int(parts[0]), float(parts[1]),
                         float(parts[2]), float(parts[3])))
    return segments


def plan_walking_targets(L_world, R_world, body_yaw0, foot_y_offset,
                         step_length, step_lateral, step_yaw,
                         swing_height, com_height, body_lateral_amp,
                         n_per_swing, n_per_transition, n_walk_cycles):
    """Build foot/CoG targets for a multi-cycle walk in arbitrary direction.

    ``step_length``/``step_lateral`` are the body-frame stride per swing
    and ``step_yaw`` is the body-frame yaw advance per swing (radians).
    Returns ``(lfoot_pos, rfoot_pos, lfoot_rot, rfoot_rot, com_pos, L_final, R_final, yaw_final)``.
    """
    lfoot_pos, rfoot_pos, com_pos = [], [], []
    yaw_l_hist, yaw_r_hist, yaw_body_hist = [], [], []
    L = np.asarray(L_world, dtype=float).copy()
    R = np.asarray(R_world, dtype=float).copy()
    body_yaw = float(body_yaw0)
    yaw_l = body_yaw
    yaw_r = body_yaw

    def _swing(swing_side):
        nonlocal body_yaw, yaw_l, yaw_r
        yaw_start = body_yaw
        yaw_end = body_yaw + step_yaw
        if swing_side == 'R':
            stance = L
            swing_start = R.copy()
            local_offset = np.array(
                [step_length, -2.0 * foot_y_offset + step_lateral, 0.0])
            com_y_dir = +1.0
        else:
            stance = R
            swing_start = L.copy()
            local_offset = np.array(
                [step_length, 2.0 * foot_y_offset + step_lateral, 0.0])
            com_y_dir = -1.0
        swing_end = stance + Rz_yaw(yaw_end) @ local_offset

        body_xy_start = 0.5 * (L[:2] + R[:2])
        body_xy_end = (0.5 * (L[:2] + swing_end[:2]) if swing_side == 'R'
                       else 0.5 * (R[:2] + swing_end[:2]))

        for k in range(n_per_swing):
            s = k / max(1, n_per_swing - 1)
            body_yaw_now = yaw_start + s * (yaw_end - yaw_start)
            sw_pos = swing_start + s * (swing_end - swing_start)
            sw_pos = sw_pos.copy()
            sw_pos[2] = swing_start[2] + swing_height * np.sin(np.pi * s)
            if swing_side == 'R':
                lfoot_pos.append(stance.copy())
                rfoot_pos.append(sw_pos)
                yaw_l_hist.append(yaw_l)            # stance: locked
                yaw_r_hist.append(body_yaw_now)     # swing: tracks body
            else:
                lfoot_pos.append(sw_pos)
                rfoot_pos.append(stance.copy())
                yaw_l_hist.append(body_yaw_now)
                yaw_r_hist.append(yaw_r)
            body_xy = body_xy_start + s * (body_xy_end - body_xy_start)
            sway_local = np.array(
                [0.0, com_y_dir * foot_y_offset * body_lateral_amp])
            sway_world = Rz_yaw(body_yaw_now)[:2, :2] @ sway_local
            com_pos.append(np.array([body_xy[0] + sway_world[0],
                                     body_xy[1] + sway_world[1],
                                     com_height]))
            yaw_body_hist.append(body_yaw_now)

        if swing_side == 'R':
            R[:] = swing_end
            yaw_r = yaw_end
        else:
            L[:] = swing_end
            yaw_l = yaw_end
        body_yaw = yaw_end

    def _transition(com_y_start_dir, com_y_end_dir):
        body_xy = 0.5 * (L[:2] + R[:2])
        for k in range(n_per_transition):
            s = (k + 1) / n_per_transition
            lfoot_pos.append(L.copy())
            rfoot_pos.append(R.copy())
            yaw_l_hist.append(yaw_l)
            yaw_r_hist.append(yaw_r)
            sway_dir = com_y_start_dir + (com_y_end_dir - com_y_start_dir) * s
            sway_local = np.array(
                [0.0, sway_dir * foot_y_offset * body_lateral_amp])
            sway_world = Rz_yaw(body_yaw)[:2, :2] @ sway_local
            com_pos.append(np.array([body_xy[0] + sway_world[0],
                                     body_xy[1] + sway_world[1],
                                     com_height]))
            yaw_body_hist.append(body_yaw)

    for _ in range(n_walk_cycles):
        _swing('R')
        _transition(+1.0, -1.0)
        _swing('L')
        _transition(-1.0, +1.0)

    lfoot_pos = np.asarray(lfoot_pos)
    rfoot_pos = np.asarray(rfoot_pos)
    com_pos = np.asarray(com_pos)
    yaw_l_hist = np.asarray(yaw_l_hist)
    yaw_r_hist = np.asarray(yaw_r_hist)
    yaw_body_hist = np.asarray(yaw_body_hist)
    T = lfoot_pos.shape[0]
    lfoot_rot = np.zeros((T, 3, 3))
    rfoot_rot = np.zeros((T, 3, 3))
    for i in range(T):
        lfoot_rot[i] = Rz_yaw(yaw_l_hist[i])
        rfoot_rot[i] = Rz_yaw(yaw_r_hist[i])
    return (lfoot_pos, rfoot_pos, lfoot_rot, rfoot_rot, com_pos, yaw_body_hist,
            L.copy(), R.copy(), float(body_yaw))


def apply_aug_to_robot(robot, problem, aug, base_world_initial):
    """Set joint angles + base pose on the robot from an augmented vector."""
    for j, val in zip(problem.joint_list, aug[:problem.n_joints]):
        j.joint_angle(float(val))
    base_xyz = aug[problem.n_joints:problem.n_joints + 3]
    base_rpy = aug[problem.n_joints + 3:problem.n_joints + 6]
    pose = Coordinates(
        pos=base_world_initial + base_xyz, rot=rpy_to_matrix(*base_rpy))
    robot.root_link.newcoords(pose, relative_coords='world')


def build_viewer(viewer_name):
    if viewer_name == 'pyrender':
        return skrobot.viewers.PyrenderViewer(resolution=(960, 720))
    if viewer_name == 'trimesh':
        return skrobot.viewers.TrimeshSceneViewer(resolution=(960, 720))
    if viewer_name == 'viser':
        return skrobot.viewers.ViserViewer()
    raise ValueError('unknown viewer: {}'.format(viewer_name))


def add_ground_with_stripes(viewer, segments_data, viewer_name,
                            stripe_spacing=0.25, margin=1.0):
    """Add a ground plane plus stripes for progress cue, and the planned
    CoG path as a line so the walked distance is easy to read off."""
    all_xy = np.concatenate(
        [np.concatenate(
            [s['lfoot_pos'][:, :2], s['rfoot_pos'][:, :2], s['com_pos'][:, :2]], axis=0)
         for s in segments_data], axis=0)
    x_min = min(float(all_xy[:, 0].min()) - margin, -1.0)
    x_max = max(float(all_xy[:, 0].max()) + margin, 1.0)
    y_min = min(float(all_xy[:, 1].min()) - margin, -1.5)
    y_max = max(float(all_xy[:, 1].max()) + margin, 1.5)
    ex = x_max - x_min
    ey = y_max - y_min
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    ground = skrobot.model.Box(
        extents=(ex, ey, 0.01), face_colors=(0.72, 0.74, 0.78))
    ground.translate([cx, cy, -0.005])
    viewer.add(ground)

    x_start = np.floor(x_min / stripe_spacing) * stripe_spacing
    for x in np.arange(x_start, x_max + 1e-6, stripe_spacing):
        stripe = skrobot.model.Box(
            extents=(0.02, ey, 0.003), face_colors=(0.32, 0.32, 0.4))
        stripe.translate([float(x), cy, 0.001])
        viewer.add(stripe)
    y_start = np.floor(y_min / stripe_spacing) * stripe_spacing
    for y in np.arange(y_start, y_max + 1e-6, stripe_spacing):
        stripe = skrobot.model.Box(
            extents=(ex, 0.01, 0.003), face_colors=(0.45, 0.45, 0.55))
        stripe.translate([cx, float(y), 0.001])
        viewer.add(stripe)

    # Planned CoG trail as a polyline plus per-segment endpoint markers
    # so the walk's progress is easy to read off. ViserViewer does not
    # implement LineString rendering, so we skip the polyline there.
    if viewer_name != 'viser':
        all_pts = np.concatenate(
            [np.column_stack([s['com_pos'][:, 0], s['com_pos'][:, 1],
                              np.full(s['com_pos'].shape[0], 0.005)])
             for s in segments_data], axis=0)
        if all_pts.shape[0] > 1:
            viewer.add(skrobot.model.LineString(points=all_pts))
    palette = [(0.85, 0.30, 0.20),
               (0.20, 0.55, 0.95),
               (0.25, 0.80, 0.35),
               (0.90, 0.80, 0.20),
               (0.80, 0.40, 0.90)]
    for i, seg in enumerate(segments_data):
        end_xy = seg['com_pos'][-1, :2]
        marker = skrobot.model.Sphere(
            radius=0.04, color=palette[i % len(palette)])
        marker.translate([float(end_xy[0]), float(end_xy[1]), 0.04])
        viewer.add(marker)


def animate_segments(viewer, robot, segments_data, viewer_name,
                     dt=0.05, loop=True):
    """Step joint angles + base pose through the viewer for all segments."""
    print('Press [q] (pyrender/trimesh) or Ctrl+C (viser) to exit.')
    try:
        while True:
            for seg in segments_data:
                problem = seg['problem']
                traj = seg['traj']
                base_world = seg['base_world_initial']
                for idx in range(traj.shape[0]):
                    apply_aug_to_robot(robot, problem, traj[idx], base_world)
                    viewer.redraw()
                    time.sleep(dt)
                    if viewer_name != 'viser' and not viewer.is_active:
                        return
            if not loop:
                break
            if viewer_name != 'viser' and not viewer.is_active:
                return
    except KeyboardInterrupt:
        print('\nInterrupted.')


def chain_segment_plans(robot, segments_spec, n_per_swing, n_per_transition,
                        body_lateral_amp, swing_height):
    """Chain ``plan_walking_targets`` deterministically across all
    segments so the entire walk becomes one continuous target stream.
    """
    base_world_initial = robot.root_link.worldcoords().worldpos().copy()
    current_rpy = rpy_from_matrix(robot.root_link.worldcoords().worldrot())
    body_yaw = float(current_rpy[2])

    left_foot_ee = CascadedCoords(parent=robot.LLEG_LINK5, name='l_foot')
    right_foot_ee = CascadedCoords(parent=robot.RLEG_LINK5, name='r_foot')
    L = left_foot_ee.worldpos().copy()
    R = right_foot_ee.worldpos().copy()
    com_height = float(robot.update_mass_properties()['total_centroid'][2])
    body_e_y = Rz_yaw(body_yaw)[:, 1]
    foot_y_offset = float(abs((L - R) @ body_e_y) / 2.0)

    lfoot_pos_segs, rfoot_pos_segs, lfoot_rot_segs, rfoot_rot_segs = [], [], [], []
    com_pos_segs, yaw_segs = [], []
    seg_offsets = [0]

    for n_cycles, dx, dy, dtheta_deg in segments_spec:
        (lfoot_pos, rfoot_pos, lfoot_rot, rfoot_rot, com_pos, yaw_body_hist,
         L_new, R_new, body_yaw_new) = plan_walking_targets(
            L, R, body_yaw, foot_y_offset,
            step_length=dx, step_lateral=dy,
            step_yaw=np.deg2rad(dtheta_deg),
            swing_height=swing_height, com_height=com_height,
            body_lateral_amp=body_lateral_amp,
            n_per_swing=n_per_swing,
            n_per_transition=n_per_transition,
            n_walk_cycles=n_cycles,
        )
        lfoot_pos_segs.append(lfoot_pos)
        rfoot_pos_segs.append(rfoot_pos)
        lfoot_rot_segs.append(lfoot_rot)
        rfoot_rot_segs.append(rfoot_rot)
        com_pos_segs.append(com_pos)
        yaw_segs.append(yaw_body_hist)
        seg_offsets.append(seg_offsets[-1] + lfoot_pos.shape[0])
        L = L_new
        R = R_new
        body_yaw = body_yaw_new

    return {
        'lfoot_pos': np.concatenate(lfoot_pos_segs, axis=0),
        'rfoot_pos': np.concatenate(rfoot_pos_segs, axis=0),
        'lfoot_rot': np.concatenate(lfoot_rot_segs, axis=0),
        'rfoot_rot': np.concatenate(rfoot_rot_segs, axis=0),
        'com_pos': np.concatenate(com_pos_segs, axis=0),
        'yaw_body_hist': np.concatenate(yaw_segs, axis=0),
        'seg_offsets': seg_offsets,
        'left_foot_ee': left_foot_ee,
        'right_foot_ee': right_foot_ee,
        'base_world_initial': base_world_initial,
        'current_rpy': current_rpy,
    }


def solve_full_walk(robot, plan, max_iter, body_lateral_amp,
                    position_weight, rotation_weight, com_weight):
    """Solve every segment as a single trajectory so smoothness and
    acceleration costs span the whole walk and the optimiser never
    sees a segment boundary."""
    lfoot_pos = plan['lfoot_pos']
    rfoot_pos = plan['rfoot_pos']
    lfoot_rot = plan['lfoot_rot']
    rfoot_rot = plan['rfoot_rot']
    com_pos = plan['com_pos']
    yaw_body_hist = plan['yaw_body_hist']
    base_world_initial = plan['base_world_initial']
    current_rpy = plan['current_rpy']
    n_waypoints = lfoot_pos.shape[0]

    lleg = [getattr(robot, 'LLEG_LINK{}'.format(i)) for i in range(6)]
    rleg = [getattr(robot, 'RLEG_LINK{}'.format(i)) for i in range(6)]
    problem = TrajectoryProblem(
        robot_model=robot,
        link_list=[lleg, rleg],
        n_waypoints=n_waypoints,
        dt=0.05,
        move_target=[plan['left_foot_ee'], plan['right_foot_ee']],
        n_base_dof=6,
    )
    problem.add_multi_ee_waypoint_cost(
        target_positions_per_chain=[lfoot_pos, rfoot_pos],
        target_rotations_per_chain=[lfoot_rot, rfoot_rot],
        position_weight=position_weight,
        rotation_weight=rotation_weight,
    )
    problem.add_com_cost(
        target_positions=com_pos, translation_axis=True, weight=com_weight)
    problem.add_smoothness_cost(weight=0.1)
    problem.add_acceleration_cost(weight=0.05)

    nominal = np.array([j.joint_angle() for j in problem.joint_list])
    base_xy_mid = np.mean(com_pos[:, :2], axis=0) - base_world_initial[:2]
    yaw_mid = float(yaw_body_hist[len(yaw_body_hist) // 2])
    nominal_aug = np.concatenate(
        [nominal,
         [base_xy_mid[0], base_xy_mid[1], 0.0, 0.0, 0.0, yaw_mid]])
    problem.add_posture_cost(nominal_angles=nominal_aug, weight=0.05)

    # Pin only the very first waypoint to the actual current robot pose
    # — every other waypoint stays free so smoothness/acceleration costs
    # propagate continuously across what used to be segment boundaries.
    problem.set_fixed_endpoints(start=True, end=False)

    n_aug = problem.n_joints + 6
    initial_traj = np.zeros((n_waypoints, n_aug))
    initial_traj[:, :problem.n_joints] = nominal[None, :]
    initial_traj[:, problem.n_joints + 0] = (
        com_pos[:, 0] - base_world_initial[0])
    initial_traj[:, problem.n_joints + 1] = (
        com_pos[:, 1] - base_world_initial[1])
    initial_traj[:, problem.n_joints + 5] = yaw_body_hist
    initial_traj[0, :problem.n_joints] = nominal
    initial_traj[0, problem.n_joints:problem.n_joints + 3] = 0.0
    initial_traj[0, problem.n_joints + 3:problem.n_joints + 6] = current_rpy

    solver = create_solver(
        'jaxls', max_iterations=max_iter, verbose=False)

    t0 = time.perf_counter()
    result = solver.solve(problem, initial_traj)
    elapsed = time.perf_counter() - t0
    print('Solved {} waypoints in {:.2f}s success={}'.format(
        n_waypoints, elapsed, result.success))

    apply_aug_to_robot(
        robot, problem, result.trajectory[-1], base_world_initial)

    return {
        'problem': problem,
        'traj': result.trajectory,
        'base_world_initial': base_world_initial,
        'lfoot_pos': lfoot_pos, 'rfoot_pos': rfoot_pos, 'com_pos': com_pos,
        'seg_offsets': plan['seg_offsets'],
        'left_foot_ee': plan['left_foot_ee'],
        'right_foot_ee': plan['right_foot_ee'],
    }


def build_segments_data_for_viz(walk):
    """Split the unified result back into per-segment dicts for the
    visualiser (so each segment gets its own coloured marker)."""
    offsets = walk['seg_offsets']
    out = []
    for i in range(len(offsets) - 1):
        a, b = offsets[i], offsets[i + 1]
        out.append({
            'problem': walk['problem'],
            'traj': walk['traj'][a:b],
            'base_world_initial': walk['base_world_initial'],
            'lfoot_pos': walk['lfoot_pos'][a:b],
            'rfoot_pos': walk['rfoot_pos'][a:b],
            'com_pos': walk['com_pos'][a:b],
            'left_foot_ee': walk['left_foot_ee'],
            'right_foot_ee': walk['right_foot_ee'],
        })
    return out


def report_tracking_errors(robot, segments_data):
    print('--- Tracking error per segment endpoint ---')
    for i, seg in enumerate(segments_data):
        problem = seg['problem']
        traj = seg['traj']
        base_world = seg['base_world_initial']
        idx = traj.shape[0] - 1
        apply_aug_to_robot(robot, problem, traj[idx], base_world)
        cog = robot.update_mass_properties()['total_centroid']
        print('  [seg {} end] CoG xy err={:.4f} m  '
              'L err={:.4f}  R err={:.4f}'.format(
                  i + 1,
                  float(np.linalg.norm(seg['com_pos'][idx, :2] - cog[:2])),
                  float(np.linalg.norm(
                      seg['lfoot_pos'][idx] - seg['left_foot_ee'].worldpos())),
                  float(np.linalg.norm(
                      seg['rfoot_pos'][idx] - seg['right_foot_ee'].worldpos()))))


def _is_ci():
    """True when running under GitHub Actions or RUN_EXAMPLE_TESTS=true."""
    return (os.environ.get('GITHUB_ACTIONS', '').lower() == 'true'
            or os.environ.get('RUN_EXAMPLE_TESTS', '').lower() == 'true')


def main():
    ci = _is_ci()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # CI runs the example as a smoke test (see tests/test_examples.py),
    # so we shrink the trajectory and iteration cap there to stay well
    # under the 120s per-example budget. Pass the flags explicitly to
    # override.
    default_segments = (['1,0.20,0,0'] if ci
                        else ['2,0.25,0,0', '2,0.20,0,8', '2,0.22,0,-4'])
    default_max_iter = 15 if ci else 120
    default_n_per_swing = 8 if ci else 30
    default_n_per_transition = 4 if ci else 15
    parser.add_argument(
        '--segments', nargs='+',
        default=default_segments,
        help="Walking segments as 'n_cycles,dx,dy,dtheta_deg'. "
             "Each segment continues from where the previous ended. "
             "dx/dy are body-frame stride per swing in metres, "
             "dtheta_deg is yaw change per swing in degrees.")
    parser.add_argument('--max-iter', type=int, default=default_max_iter,
                        help='Levenberg-Marquardt iteration cap.')
    parser.add_argument('--n-per-swing', type=int,
                        default=default_n_per_swing,
                        help='Waypoints per single foot swing.')
    parser.add_argument('--n-per-transition', type=int,
                        default=default_n_per_transition,
                        help='Waypoints per double-support transition.')
    parser.add_argument('--viewer', type=str, default='pyrender',
                        choices=['pyrender', 'trimesh', 'viser'],
                        help='Viewer backend for live visualisation.')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Skip visualisation; only print timings/errors.')
    parser.add_argument('--no-loop', action='store_true',
                        help='Play the walk once instead of looping.')
    args = parser.parse_args()

    segments = parse_segments(args.segments)
    print('Planned {} segment(s){}:'.format(
        len(segments), ' [CI smoke-test settings]' if ci else ''))
    for i, (n, dx, dy, dt) in enumerate(segments):
        print('  {}: n_cycles={} dx={} dy={} dtheta={}deg'.format(
            i + 1, n, dx, dy, dt))

    t_total = time.perf_counter()

    robot = JaxonJVRC()
    set_initial_stance(robot, knee_bend_deg=20.0)
    floor_anchor(robot)

    plan = chain_segment_plans(
        robot, segments,
        n_per_swing=args.n_per_swing,
        n_per_transition=args.n_per_transition,
        body_lateral_amp=0.7, swing_height=0.08)

    walk = solve_full_walk(
        robot, plan, args.max_iter, body_lateral_amp=0.7,
        position_weight=2000.0, rotation_weight=500.0,
        com_weight=300.0)

    segments_data = build_segments_data_for_viz(walk)

    report_tracking_errors(robot, segments_data)

    final_pos = robot.root_link.worldcoords().worldpos()
    final_yaw_deg = np.rad2deg(yaw_from_rotation(
        robot.root_link.worldcoords().worldrot()))
    print('Final base pose: x={:.3f} y={:.3f} z={:.3f} yaw={:.1f}deg'.format(
        final_pos[0], final_pos[1], final_pos[2], final_yaw_deg))
    print('Total wall time: {:.2f} s'.format(time.perf_counter() - t_total))

    if args.no_interactive:
        return

    apply_aug_to_robot(
        robot, segments_data[0]['problem'], segments_data[0]['traj'][0],
        segments_data[0]['base_world_initial'])
    viewer = build_viewer(args.viewer)
    viewer.add(robot)
    add_ground_with_stripes(viewer, segments_data, args.viewer)
    viewer.show()

    animate_segments(viewer, robot, segments_data, args.viewer,
                     dt=0.05, loop=not args.no_loop)

    if args.viewer != 'viser':
        viewer.close()


if __name__ == '__main__':
    main()

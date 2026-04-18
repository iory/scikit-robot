#!/usr/bin/env python
"""Griphis alternating wall-climbing gait using fullbody IK.

Griphis is a two-gripper climbing robot with suction-pad ends.  This
example drives it up a vertical wall with an alternating gait: one
gripper stays stuck while the other swings along an arc to a higher
contact point, then the roles swap.  The whole thing is one repeated
multi-end-effector IK call with ``use_base='6dof'`` so that ``base_link``
is free to translate and rotate while both hands are constrained.

Assumptions
-----------
- Wall is a vertical plane (normal = +Y, surface = xz-plane) at y = WALL_Y.
- Robot climbs toward +Z (up) along the wall.
- Suction pads on both grippers; the nail-closing joints are kept open
  (worm_rotate = 15 rad) throughout, so adhesion is modelled as pure
  pose contact.
- Fullbody IK (``use_base='6dof'``) is used so ``base_link`` can
  translate/rotate freely in space while stance/swing constraints pin the
  supporting hand and move the swing hand to its next wall target.

The URDF + meshes are fetched on first use via
``skrobot.models.Griphis`` (downloaded to ``~/.skrobot/griphis_description/``),
which also attaches ``gripper_1_end_coords`` / ``gripper_2_end_coords`` at
the centroid of each gripper's four nail tips with +X pointing forward.
"""
import argparse
import time

import numpy as np

from skrobot.collision import RobotCollisionChecker
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.models import Griphis


# Collision-proxy tuning.  The default swept-sphere radii for Griphis's
# stubby links are too conservative and report "collisions" at known-safe
# straight-extended poses.  Scaling radii to 0.4 gives the two-column
# gait ~1cm margin without missing real cross-arm contacts; intra-arm
# pairs are ignored because the 4-DoF arm chain can never physically
# fold into itself.
COLLISION_RADIUS_SCALE = 0.4
ARM_LINK_TEMPLATES = ('mid_pitch_{gid}_link', 'pitch_{gid}_link',
                      'yaw_{gid}_link', 'roll_{gid}_link',
                      'worm_{gid}_link')

# Wall geometry: xz-plane at y = WALL_Y, normal pointing back toward the robot.
WALL_Y = 0.20
WALL_NORMAL = np.array([0.0, -1.0, 0.0])   # from wall surface toward robot
WALL_UP = np.array([0.0, 0.0, 1.0])        # "up" along the wall

# Worm open angle: nails retracted, suction pad active.
WORM_OPEN_RAD = 15.0

# Per-step climb distance (swing hand moves this far along WALL_UP each step).
STEP_LENGTH = 0.10

# Arc trajectory: the hand lifts off the wall by ARC_HEIGHT along
# +WALL_NORMAL in a half-sine peaking at the midpoint, split into
# ARC_SUBSTEPS IK-solved waypoints so the viewer can animate the swing.
ARC_HEIGHT = 0.06
ARC_SUBSTEPS = 20

# Initial wall-contact xz positions (symmetric left/right).
END1_START_XZ = np.array([0.10, 0.00])   # (x, z)
END2_START_XZ = np.array([-0.10, 0.00])

_FULL_MASK_PAIR = [np.array([1, 1, 1]), np.array([1, 1, 1])]


# ---------------------------------------------------------------------------
# Wall / link-list / IK helpers
# ---------------------------------------------------------------------------

def wall_contact_coords(xz):
    """Coordinates on the wall at (xz[0], WALL_Y, xz[1]) with end_coords'
    +X into the wall (along -WALL_NORMAL) and +Z along world +Z.
    """
    ex = -WALL_NORMAL
    ez = WALL_UP
    ey = np.cross(ez, ex)
    rot = np.column_stack([ex, ey, ez])
    return Coordinates(pos=np.array([xz[0], WALL_Y, xz[1]]), rot=rot)


def arm_link_list(robot, gripper_id):
    """Joints from base_link to roll_{gid}_link (worm/nail excluded)."""
    return [getattr(robot, f'mid_pitch_{gripper_id}_link'),
            getattr(robot, f'pitch_{gripper_id}_link'),
            getattr(robot, f'yaw_{gripper_id}_link'),
            getattr(robot, f'roll_{gripper_id}_link')]


def set_suction_open(robot):
    """Fix both worm_rotate joints at WORM_OPEN_RAD (suction-pad mode)."""
    robot.worm_rotate_1.joint_angle(WORM_OPEN_RAD)
    robot.worm_rotate_2.joint_angle(WORM_OPEN_RAD)


def build_self_collision_checker(robot):
    """Self-collision checker covering cross-arm and arm-vs-base pairs only.

    Intra-arm pairs are excluded because the 4-DoF arm chain cannot
    physically self-overlap; the swept-sphere proxies would report false
    positives otherwise.
    """
    checker = RobotCollisionChecker(robot)
    groups = {1: [], 2: []}
    for gid in (1, 2):
        for tmpl in ARM_LINK_TEMPLATES:
            name = tmpl.format(gid=gid)
            link = getattr(robot, name, None)
            if link is None:
                continue
            checker.add_link(link, radius_scale=COLLISION_RADIUS_SCALE)
            groups[gid].append(name)
    checker.add_link(robot.base_link, radius_scale=COLLISION_RADIUS_SCALE)
    ignore = []
    for grp in groups.values():
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                ignore.append((grp[i], grp[j]))
    checker.setup_self_collision_pairs(min_link_distance=2,
                                       ignore_pairs=ignore)
    return checker


def self_collision_summary(checker):
    d = checker.compute_self_collision_distances()
    if len(d) == 0:
        return 0, float('inf')
    return int(np.sum(d < 0.0)), float(np.min(d))


def solve_initial_pose(robot, end1, end2):
    """Place both grippers on the wall at their starting xz positions.

    Seeds the yaw joints with ±90° so each gripper already faces the wall
    (+X → +Y) before the solver runs; without this seed the solver tends
    to stall in a local minimum with both arms extended sideways (±X).
    """
    robot.yaw_1.joint_angle(-np.pi / 2)
    robot.yaw_2.joint_angle(np.pi / 2)

    t1 = wall_contact_coords(END1_START_XZ)
    t2 = wall_contact_coords(END2_START_XZ)
    link_list = [arm_link_list(robot, 1), arm_link_list(robot, 2)]
    return robot.inverse_kinematics(
        target_coords=[t1, t2],
        move_target=[end1, end2],
        link_list=link_list,
        use_base='6dof',
        base_weight=1.0,
        position_mask=_FULL_MASK_PAIR,
        rotation_mask=_FULL_MASK_PAIR,
        stop=200,
    )


def _ik_once(robot, end_swing, end_stance, swing_target, link_list,
             thre, rthre, stop):
    stance_hold = end_stance.copy_worldcoords()
    return robot.inverse_kinematics(
        target_coords=[swing_target, stance_hold],
        move_target=[end_swing, end_stance],
        link_list=link_list,
        use_base='6dof',
        # base_weight > 1 lets the base absorb the step so the stance arm
        # stays near its held configuration.
        base_weight=3.0,
        position_mask=_FULL_MASK_PAIR,
        rotation_mask=_FULL_MASK_PAIR,
        thre=[thre, thre],
        rthre=[rthre, rthre],
        revert_if_fail=False,
        stop=stop,
    )


def arc_waypoints(start_pos, goal_pos, goal_rot,
                  n=ARC_SUBSTEPS, height=ARC_HEIGHT):
    """N (position, rotation) waypoints from start_pos to goal_pos.

    xz interpolates linearly; the hand lifts off the wall by a half-sine
    of amplitude ``height`` along +WALL_NORMAL (robot-side of the wall)
    so the swing arc never penetrates.  Rotation stays at goal_rot so
    the gripper keeps facing the wall, ready for the next suction contact.
    Skips t=0 (== current pose) and includes t=1 (final goal pose).
    """
    ts = np.linspace(0.0, 1.0, n + 1)[1:]
    pts = []
    for t in ts:
        xz = start_pos + t * (goal_pos - start_pos)
        lift = WALL_NORMAL * height * np.sin(np.pi * t)
        pts.append((xz + lift, goal_rot))
    return pts


def step_swing(robot, end_swing, end_stance, swing_target, link_list,
               thre=0.0005, rthre_deg=0.5, stop=3000, on_substep=None,
               substep_thre=0.003, substep_rthre_deg=2.0,
               substep_stop=400):
    """Move swing to swing_target while holding stance, animating through
    an arc that lifts off the wall and lands back on it at the goal.
    ``on_substep(sub_target)`` is invoked after every sub-IK solve.

    Intermediate arc waypoints use the looser (substep_thre,
    substep_rthre_deg, substep_stop) tolerance so the animation runs
    fast; only the final waypoint (the actual wall-contact goal) is
    solved with the tight (thre, rthre_deg, stop) tolerance so the
    gripper lands precisely.

    Returns (final_swing_pos_err, stance_pos_drift) in metres.
    """
    rthre = np.deg2rad(rthre_deg)
    sub_rthre = np.deg2rad(substep_rthre_deg)
    start_pos = end_swing.worldpos().copy()
    goal_pos = swing_target.worldpos().copy()
    goal_rot = swing_target.worldrot()
    stance_pos_start = end_stance.worldpos().copy()

    waypoints = list(arc_waypoints(start_pos, goal_pos, goal_rot))
    n = len(waypoints)
    for i, (sub_pos, sub_rot) in enumerate(waypoints):
        is_final = (i == n - 1)
        sub_target = Coordinates(pos=sub_pos, rot=sub_rot)
        _ik_once(robot, end_swing, end_stance, sub_target, link_list,
                 thre if is_final else substep_thre,
                 rthre if is_final else sub_rthre,
                 stop if is_final else substep_stop)
        if on_substep is not None:
            on_substep(sub_target)

    swing_err = float(np.linalg.norm(end_swing.worldpos() - goal_pos))
    stance_drift = float(
        np.linalg.norm(end_stance.worldpos() - stance_pos_start))
    return swing_err, stance_drift


def next_wall_target_above(end_swing):
    """Swing climbs its own x-column by STEP_LENGTH along WALL_UP."""
    p = end_swing.worldpos()
    xz_on_wall = np.array([p[0], p[2] + STEP_LENGTH * WALL_UP[2]])
    return wall_contact_coords(xz_on_wall)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(num_steps=40, interactive=True, step_pause=0.05,
         substep_pause=0.0):
    robot = Griphis()
    end1 = robot.gripper_1_end_coords
    end2 = robot.gripper_2_end_coords
    set_suction_open(robot)

    print('--- solving initial pose ---')
    ok = solve_initial_pose(robot, end1, end2)
    print(f'initial IK: {"OK" if ok is not False else "FAIL"}')
    print(f'  end1 @ {end1.worldpos().round(4)}  end2 @ {end2.worldpos().round(4)}')
    print(f'  base  @ {robot.base_link.worldpos().round(4)}')

    checker = build_self_collision_checker(robot)
    n0, md0 = self_collision_summary(checker)
    print(f'  self_collision: n={n0} min_dist={md0 * 1000:+.1f}mm')
    assert n0 == 0, (
        f'initial pose is self-colliding ({n0} pairs, min={md0 * 1000:.1f}mm) '
        'before any step — refuse to start the gait')

    viewer = None
    axis_swing = None
    if interactive:
        try:
            from skrobot.viewers import PyrenderViewer
            viewer = PyrenderViewer()
            viewer.add(robot)
            # Ground at z = 0, spanning the robot side of the wall so it
            # visually anchors where the climb starts from.
            ground = Box(extents=[2.0, 2.0, 0.01],
                         face_colors=[180, 180, 180, 200])
            ground.translate([0.0, WALL_Y - 1.0, -0.005])
            viewer.add(ground)
            wall_box = Box(extents=[2.0, 0.005, 4.0],
                           face_colors=[150, 200, 255, 120])
            wall_box.translate([0, WALL_Y, 1.5])
            viewer.add(wall_box)
            axis_swing = Axis(axis_radius=0.003, axis_length=0.05)
            viewer.add(axis_swing)
            viewer.show()
        except Exception as e:
            print(f'viewer skipped: {e}')
            viewer = None

    # link_list is a constant [arm1, arm2]; the IK solver auto-pairs each
    # chain with its move_target via kinematic-chain membership, so we do
    # not have to reorder even when swing/stance swap each iteration.
    link_list_both = [arm_link_list(robot, 1), arm_link_list(robot, 2)]
    stance_id = 2
    for step in range(num_steps):
        swing_id = 1 if stance_id == 2 else 2
        end_swing = end1 if swing_id == 1 else end2
        end_stance = end2 if stance_id == 2 else end1
        target = next_wall_target_above(end_swing)

        if axis_swing is not None:
            axis_swing.newcoords(target)
            viewer.redraw()

        def _on_substep(sub_target):
            if viewer is None:
                return
            if axis_swing is not None:
                axis_swing.newcoords(sub_target)
            viewer.redraw()
            if substep_pause > 0:
                time.sleep(substep_pause)

        swing_err, stance_drift = step_swing(
            robot, end_swing, end_stance, target, link_list_both,
            on_substep=_on_substep)
        n_coll, min_dist = self_collision_summary(checker)
        tag = 'COLLIDE' if n_coll > 0 else 'ok'
        print(f'step {step:02d}  swing=G{swing_id}  '
              f'target_z={target.worldpos()[2]:.3f}  '
              f'swing_err={swing_err * 1000:.1f}mm  '
              f'stance_drift={stance_drift * 1000:.1f}mm  '
              f'coll={n_coll:d} min_d={min_dist * 1000:+.1f}mm {tag}  '
              f'end{swing_id} @ {end_swing.worldpos().round(4)}')

        if viewer is not None:
            viewer.redraw()
            time.sleep(step_pause)

        stance_id, swing_id = swing_id, stance_id

    if viewer is not None:
        print('==> Press [q] to close window')
        while viewer.is_active:
            time.sleep(0.1)
            viewer.redraw()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--steps', type=int, default=None,
        help='number of alternating swing steps (default: 40 interactive, '
             '3 with --no-interactive)')
    parser.add_argument(
        '--no-interactive', action='store_true',
        help='skip the PyrenderViewer and exit as soon as the gait finishes; '
             'also reduces --steps default to 3 so smoke tests finish fast')
    parser.add_argument('--pause', type=float, default=0.05,
                        help='seconds to pause between steps in viewer')
    parser.add_argument('--substep-pause', type=float, default=0.0,
                        help='seconds to pause between arc waypoints')
    args = parser.parse_args()
    steps = args.steps if args.steps is not None else (
        3 if args.no_interactive else 40)
    main(num_steps=steps,
         interactive=not args.no_interactive,
         step_pause=args.pause,
         substep_pause=args.substep_pause)

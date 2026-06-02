#!/usr/bin/env python
"""Hydrus planar end-effector tracing a circle with joint IK.

Hydrus is a planar multilink aerial robot.  Here its root (body center)
is pinned in space and the tip frame ``leg5`` is driven around a circle
in the horizontal plane.  Two things move together each step:

- the body yaw is rotated so the arm points at the current target
  (``aim``), which keeps the required joint motion small, and
- ``joint1`` / ``joint3`` are solved with planar IK to land ``leg5`` on
  the target, while ``joint2`` is held fixed.

Because the root is fixed and only ``leg5`` xy is constrained, the IK
call uses ``position_mask='xy'`` and ``rotation_mask=False``.

The URDF + meshes are fetched on first use via ``skrobot.models.Hydrus``
(cached under ``~/.skrobot/hydrus_description/``).
"""
import argparse

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.model import Axis
from skrobot.model.primitives import Sphere
from skrobot.models import Hydrus


# Body center is pinned at the world origin.
ROOT_POS = np.zeros(3)

# Center configuration of the arm joints; the circle is traced around the
# arm_end_coords (leg5) position reached at this configuration.
JOINT1_CENTER = 1.0
JOINT2_CENTER = 0.6
JOINT3_CENTER = 1.0

# Travel limits for joint1 / joint3 [rad].
JOINT_MIN = 0.3
JOINT_MAX = 1.47


def aim_yaw(target_xy, alpha0):
    """Body yaw that points the arm bearing at ``target_xy``.

    ``alpha0`` is the bearing of leg5 seen from the root at zero yaw, so
    subtracting it makes the arm face the target with minimal joint
    motion.  The root is at ``ROOT_POS``.
    """
    phi = np.arctan2(target_xy[1] - ROOT_POS[1],
                     target_xy[0] - ROOT_POS[0])
    return phi - alpha0


def set_center_pose(robot):
    """Put the robot in its center configuration with zero body yaw."""
    robot.root_link.newcoords(Coordinates(pos=ROOT_POS))
    robot.joint1.joint_angle(JOINT1_CENTER)
    robot.joint2.joint_angle(JOINT2_CENTER)
    robot.joint3.joint_angle(JOINT3_CENTER)


def solve_step(robot, target_xy, alpha0, ik_stop):
    """Aim the body at the target and IK the arm onto it in the xy plane.

    Uses the ``arm`` limb (joint1..joint3, tip at ``arm_end_coords``):
    only the in-plane position is constrained, so ``rotation_axis=False``
    and ``translation_axis='z'`` (z is left free).

    Returns (success, xy_error).
    """
    robot.root_link.newcoords(
        Coordinates(pos=ROOT_POS).rotate(aim_yaw(target_xy, alpha0), 'z'))
    result = robot.arm.inverse_kinematics(
        Coordinates(pos=[target_xy[0], target_xy[1], 0.0]),
        rotation_axis=False,
        translation_axis='z',
        stop=ik_stop,
        revert_if_fail=False,
    )
    err = float(
        np.linalg.norm(robot.arm_end_coords.worldpos()[:2] - target_xy))
    return result is not False and result is not None, err


def main(steps=720, radius=0.10, ik_stop=20, interactive=True, pause=0.0,
         viewer_name='pyrender'):
    robot = Hydrus()
    robot.joint1.min_angle = JOINT_MIN
    robot.joint1.max_angle = JOINT_MAX
    robot.joint3.min_angle = JOINT_MIN
    robot.joint3.max_angle = JOINT_MAX

    set_center_pose(robot)
    center = robot.arm_end_coords.worldpos().copy()
    bearing = center[:2] - ROOT_POS[:2]
    alpha0 = float(np.arctan2(bearing[1], bearing[0]))
    print('circle center (arm_end_coords) = {}'.format(center.round(4)))
    print('arm bearing alpha0 = {:.3f} rad'.format(alpha0))
    print('arm joints = {}'.format([j.name for j in robot.arm.joint_list]))

    viewer = None
    target_axis = None
    if interactive:
        try:
            from skrobot.viewers import create_viewer
            viewer = create_viewer(viewer_name)
            viewer.add(robot)
            for j in range(80):
                phi = 2.0 * np.pi * j / 80
                mark = Sphere(radius=0.008,
                              pos=center + np.array([radius * np.cos(phi),
                                                     radius * np.sin(phi),
                                                     0.0]))
                mark.set_color([60, 120, 220, 255])
                viewer.add(mark)
            target_axis = Axis(axis_radius=0.012, axis_length=0.15,
                               pos=center.copy())
            viewer.add(target_axis)
            viewer.show()
            print('==> viewer running; blue circle is the reference path, '
                  'the axis is the live target.')
        except Exception as e:
            print('viewer skipped: {}'.format(e))
            viewer = None

    errors = []
    log_every = max(1, steps // 12)
    k = 0
    while True:
        theta = 2.0 * np.pi * (k % steps) / steps
        target_xy = center[:2] + radius * np.array([np.cos(theta),
                                                    np.sin(theta)])
        ok, err = solve_step(robot, target_xy, alpha0, ik_stop)
        errors.append(err)

        if k % log_every == 0 or not ok:
            print('step {:>4d}  target=({:+.3f},{:+.3f})  '
                  'err={:.4f}m  ok={}  joints=({:+.3f},{:+.3f},{:+.3f})'.format(
                      k, target_xy[0], target_xy[1], err, ok,
                      robot.joint1.joint_angle(), robot.joint2.joint_angle(),
                      robot.joint3.joint_angle()))

        if viewer is not None:
            target_axis.newcoords(
                Coordinates(pos=[target_xy[0], target_xy[1], 0.0]))
            viewer.pause(pause)
            if not viewer.is_active:
                break
            k += 1
        else:
            k += 1
            if k >= steps:
                break

    print('mean xy error = {:.4f} m, max = {:.4f} m  (over {} steps)'.format(
        float(np.mean(errors)), float(np.max(errors)), len(errors)))

    if viewer is not None:
        viewer.wait_until_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=720,
                        help='number of samples per revolution')
    parser.add_argument('--radius', type=float, default=0.10,
                        help='circle radius [m]')
    parser.add_argument('--ik-stop', type=int, default=20,
                        help='max IK iterations per step')
    parser.add_argument('--no-interactive', action='store_true',
                        help='run one revolution headless and exit')
    parser.add_argument('--viewer', type=str,
                        choices=['trimesh', 'pyrender', 'viser'],
                        default='pyrender',
                        help='Choose the viewer type: trimesh, pyrender or '
                             'viser')
    parser.add_argument('--pause', type=float, default=0.0,
                        help='seconds to pause between steps in the viewer')
    args = parser.parse_args()
    main(steps=args.steps,
         radius=args.radius,
         ik_stop=args.ik_stop,
         interactive=not args.no_interactive,
         pause=args.pause,
         viewer_name=args.viewer)

#!/usr/bin/env python
"""Find one mobile-base pose from which a robot reaches several targets.

Fetch has to grasp two objects on a table from above and two on a shelf
from the side, all without driving in between. Shared-base IK from the
current pose alone settles in a local optimum; an inverse reachability
map proposes base poses from which every target is within reach, and the
same IK started there reaches all four.

Usage:
    python examples/base_placement_demo.py
    python examples/base_placement_demo.py --no-interactive
"""

import argparse
import time

import numpy as np

import skrobot
from skrobot.coordinates import Coordinates
from skrobot.kinematics import find_base_placement
from skrobot.kinematics import InverseReachabilityMap
from skrobot.kinematics import set_base_pose
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.models import Fetch


# (center, size) of the furniture standing on the floor
FURNITURE = {
    'table': ([3.2, 0.0, 0.36], [0.7, 1.2, 0.72]),
    'shelf': ([2.3, 1.05, 0.6], [0.9, 0.35, 1.2]),
}
ROBOT_RADIUS = 0.35


def make_targets():
    def top_grasp(pos, yaw):
        return Coordinates(pos=pos).rotate(yaw, 'z').rotate(np.pi / 2, 'y')

    def side_grasp(pos):
        return Coordinates(pos=pos).rotate(np.pi / 2, 'z')

    return [
        top_grasp([2.95, 0.25, 0.80], 0.0),
        top_grasp([3.05, -0.20, 0.82], np.pi / 4),
        side_grasp([2.15, 0.80, 0.95]),
        side_grasp([2.45, 0.80, 1.10]),
    ]


def is_valid_base(pose):
    """Keep the base footprint clear of the furniture."""
    x, y = pose.worldpos()[:2]
    for center, size in FURNITURE.values():
        dx = max(abs(x - center[0]) - size[0] / 2.0, 0.0)
        dy = max(abs(y - center[1]) - size[1] / 2.0, 0.0)
        if np.hypot(dx, dy) < ROBOT_RADIUS:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=500000,
                        help='arm samples in the inverse reachability map')
    parser.add_argument('--no-interactive', action='store_true',
                        help='solve and print, without the viewer')
    args = parser.parse_args()

    robot = Fetch()
    robot.reset_pose()
    move_target = robot.rarm.end_coords
    targets = make_targets()

    start = time.time()
    irm = InverseReachabilityMap(robot, move_target).compute(
        n_samples=args.samples)
    print('Inverse reachability map: {} samples in {:.2f} s'.format(
        irm.n_samples, time.time() - start))

    start = time.time()
    local = find_base_placement(
        robot, targets, move_target=move_target, is_valid_base=is_valid_base)
    print('IK from the current base:   {}/{} reached ({:.2f} s)'.format(
        local.n_reached, len(targets), time.time() - start))

    start = time.time()
    result = find_base_placement(
        robot, targets, move_target=move_target, irm=irm,
        is_valid_base=is_valid_base)
    print('IK from the map candidates: {}/{} reached ({:.2f} s)'.format(
        result.n_reached, len(targets), time.time() - start))
    print('  base pose: {}'.format(result))

    if args.no_interactive:
        return

    viewer = skrobot.viewers.ViserViewer()
    for name, (center, size) in FURNITURE.items():
        color = [176, 140, 105] if name == 'table' else [140, 150, 165]
        viewer.add(Box(size, face_colors=color, pos=center))
    for target in targets:
        axis = Axis(axis_radius=0.008, axis_length=0.12)
        axis.newcoords(target)
        viewer.add(axis)
    set_base_pose(robot, result.base_pose)
    viewer.add(robot)
    viewer.show()
    print('Cycling through the targets. Press Ctrl+C to exit.')
    try:
        while True:
            for av in result.angle_vectors:
                robot.angle_vector(av)
                viewer.redraw()
                time.sleep(1.5)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

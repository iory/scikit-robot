#!/usr/bin/env python

import argparse
import importlib.util
import os
import sys
import time

import skrobot
from skrobot.viewers import MitsubaViewer


_INSTALL_HINT = """\
This example needs the mitsuba package.
Install it with:
    pip install mitsuba
Wheels are published for CPython 3.9 and newer; there may be none yet for the
very latest interpreter release."""


def _require_mitsuba():
    """Exit cleanly when mitsuba is missing.

    skrobot only depends on mitsuba through the ``[all]`` extra, and there are
    interpreter versions it has no wheels for yet. Reporting that is more use
    than the bare ImportError MitsubaViewer would raise, and it keeps
    tests/skrobot_tests/test_examples.py -- which runs every example -- from
    failing on an environment that simply cannot install it.
    """
    if importlib.util.find_spec('mitsuba') is None:
        print('Missing dependency: mitsuba', file=sys.stderr)
        print(_INSTALL_HINT, file=sys.stderr)
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--output', type=str, default='mitsuba_render.png',
        help='Output image path.')
    parser.add_argument(
        '--resolution', type=int, nargs=2, metavar=('W', 'H'),
        default=(640, 480),
        help='Output resolution as width height.')
    parser.add_argument(
        '--spp', type=int, default=None,
        help='Samples per pixel (None uses variant-dependent default).')
    parser.add_argument(
        '--variant', type=str, default=None,
        help='Mitsuba variant (None enables auto-selection).')
    parser.add_argument(
        '--no-interactive', action='store_true',
        help='Accepted for consistency with the other examples and with '
             'tests/skrobot_tests/test_examples.py, which runs every example '
             'with it. This one never opens a window or waits for input, so '
             'it makes no difference here.')
    args = parser.parse_args()
    _require_mitsuba()

    robot = skrobot.models.Panda()
    robot.reset_manip_pose()
    robot.rarm.joint_list[1].joint_angle(-0.6)
    robot.rarm.joint_list[3].joint_angle(-1.9)

    viewer = MitsubaViewer(
        resolution=tuple(args.resolution),
        spp=args.spp,
        variant=args.variant,
        fov=(58.0, 45.0),     # Control framing (same tuple convention as pyrender).
        light_intensity=4.0,  # Increase key light for a brighter still render.
    )
    viewer.add(robot)

    # Named markers are useful for targets/obstacles in rendered reports.
    viewer.add_box(
        center=[0.48, 0.18, 0.16],
        extents=[0.14, 0.10, 0.03],
        color=(0.55, 0.35, 0.18),
        name='tray')
    viewer.add_sphere(
        center=[0.48, 0.18, 0.27],
        radius=0.035,
        color=(0.90, 0.15, 0.15),
        name='target')

    viewer.set_camera(
        eye=[1.45, -1.55, 1.05],
        target=[0.32, 0.05, 0.38],
        up=(0, 0, 1),
    )

    output = os.path.abspath(args.output)
    start = time.time()
    viewer.save_image(output)
    elapsed = time.time() - start

    print('Saved image: {}'.format(output))
    print('Mitsuba variant: {}'.format(viewer.mi.variant()))
    print('Render time: {:.3f} s'.format(elapsed))


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Make a description-only URDF loadable by Gazebo (gz-sim / sdformat).

sdformat's URDF parser drops every link without mass, and a massless root
link drops the whole model.  This gives each link that has no inertial a
token one (or one estimated from its geometry with --mass-from-geometry),
leaving existing inertials untouched, and can strip collisions and mimic
tags.  The result is still a URDF; load it with ``gz sdf -p`` or spawn it.
"""

import argparse
from pathlib import Path
import sys

from skrobot.urdf import prepare_urdf_for_gazebo_file
from skrobot.urdf.prepare_for_gazebo import DEFAULT_TOKEN_INERTIA
from skrobot.urdf.prepare_for_gazebo import DEFAULT_TOKEN_MASS
from skrobot.utils.inertia import DEFAULT_DENSITY


def main():
    parser = argparse.ArgumentParser(
        description='Make a description-only URDF loadable by Gazebo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  convert-urdf-for-gazebo robot.urdf
  convert-urdf-for-gazebo robot.urdf -o robot_gz.urdf --strip-collision
  convert-urdf-for-gazebo robot.urdf --mass-from-geometry --density 500
        """
    )
    parser.add_argument('urdf_file', type=str, help='Path to the input URDF file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output URDF path (default: <urdf_stem>_gazebo.urdf)')
    parser.add_argument('--token-mass', type=float, default=DEFAULT_TOKEN_MASS,
                        help='Mass [kg] given to a link without an inertial '
                             '(default: %(default)s)')
    parser.add_argument('--token-inertia', type=float,
                        default=DEFAULT_TOKEN_INERTIA,
                        help='Diagonal inertia [kg m^2] of the token inertial '
                             '(default: %(default)s)')
    parser.add_argument('--mass-from-geometry', action='store_true',
                        help="Estimate a missing inertial from the link's "
                             'collision (else visual) geometry as a solid of '
                             '--density; falls back to the token inertial')
    parser.add_argument('--density', type=float, default=DEFAULT_DENSITY,
                        help='Density [kg/m^3] for --mass-from-geometry '
                             '(default: %(default)s)')
    parser.add_argument('--strip-collision', action='store_true',
                        help='Remove every <collision> element')
    parser.add_argument('--strip-mimic', action='store_true',
                        help='Remove every <mimic> element (by default they '
                             'are kept; sdformat 14 turns them into SDF '
                             'mimic constraints)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='List every link that got an inertial')
    args = parser.parse_args()

    urdf_path = Path(args.urdf_file)
    if not urdf_path.exists():
        print('error: URDF not found: {}'.format(urdf_path), file=sys.stderr)
        return 1
    out_path = args.output or str(
        urdf_path.with_name(urdf_path.stem + '_gazebo.urdf'))

    try:
        report = prepare_urdf_for_gazebo_file(
            str(urdf_path), out_path,
            token_mass=args.token_mass,
            token_inertia=args.token_inertia,
            mass_from_geometry=args.mass_from_geometry,
            density=args.density,
            strip_collision=args.strip_collision,
            strip_mimic=args.strip_mimic,
        )
    except ValueError as e:
        print('error: {}'.format(e), file=sys.stderr)
        return 1

    added = report['added']
    token = sorted(name for name, method in added.items() if method == 'token')
    print('wrote URDF: {}'.format(out_path))
    print('inertials: {} added ({} token, {} from geometry), {} kept'.format(
        len(added), len(token), len(added) - len(token), len(report['kept'])))
    if report['replaced']:
        print('replaced non-positive-mass inertials: {}'.format(
            ', '.join(report['replaced'])))
    if args.strip_collision:
        print('collisions removed: {}'.format(report['collisions_removed']))
    if report['mimic_joints']:
        print('mimic joints ({}): {}'.format(
            'stripped' if report['mimic_stripped'] else 'kept',
            len(report['mimic_joints'])))
    if args.verbose:
        for name in sorted(added):
            print('  {}: {}'.format(name, added[name]))
    return 0


if __name__ == '__main__':
    sys.exit(main())

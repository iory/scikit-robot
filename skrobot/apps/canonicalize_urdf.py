#!/usr/bin/env python

import argparse
import sys

from skrobot.urdf.canonicalize import canonicalize_urdf_file


def main():
    """Rewrite a URDF in canonical, diff-friendly form."""
    parser = argparse.ArgumentParser(
        description='Rewrite a URDF in canonical form (links/joints sorted '
                    'by name, fixed attribute order, uniform number '
                    'formatting) so two equivalent URDFs diff cleanly.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'urdf_file',
        type=str,
        help='Path to the URDF file')
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path (default: print to stdout)')
    parser.add_argument(
        '--kinematics-only',
        action='store_true',
        help='Drop visual/collision/inertial/material/transmission/gazebo '
             'elements, keeping only the kinematic skeleton')

    args = parser.parse_args()

    try:
        text = canonicalize_urdf_file(
            args.urdf_file, output_path=args.output,
            kinematics_only=args.kinematics_only)
    except (ValueError, OSError) as e:
        print("Error: {}".format(e), file=sys.stderr)
        sys.exit(1)
    if args.output is None:
        sys.stdout.write(text)


if __name__ == '__main__':
    main()

#!/usr/bin/env python

import argparse
import sys

from skrobot.urdf.hash import get_urdf_hash


def main():
    """Calculate hash of URDF file including all referenced assets."""
    parser = argparse.ArgumentParser(
        description='Calculate comprehensive hash of URDF including all '
                    'referenced meshes and textures.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'urdf_file',
        type=str,
        help='Path to the URDF file')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose output')

    args = parser.parse_args()

    try:
        hash_value = get_urdf_hash(args.urdf_file)
        if args.verbose:
            print(f"URDF file: {args.urdf_file}")
            print(f"Hash: {hash_value}")
        else:
            print(hash_value)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

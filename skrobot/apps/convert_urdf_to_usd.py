#!/usr/bin/env python
"""
Convert a URDF into a USD physics articulation (Isaac Sim / PhysX).

One rigid body per link, visual meshes embedded with their colours (and a
``UsdPreviewSurface`` per glTF PBR material), convex collision meshes, and
revolute/prismatic/fixed joints with position drives. The robot is the stage's
defaultPrim, so the file can be referenced from another stage as is. Needs
``usd-core`` (``pip install usd-core``); ``--decompose`` also needs ``coacd``.
"""

import argparse
from pathlib import Path
import sys


def _parse_home(items):
    """``['joint=0.5', ...]`` -> ``{'joint': 0.5}``."""
    home = {}
    for item in items or []:
        name, sep, value = item.partition('=')
        if not sep or not name:
            raise ValueError('--home takes JOINT=VALUE, not {!r}'.format(item))
        home[name] = float(value)
    return home


def main():
    parser = argparse.ArgumentParser(
        description='Convert a URDF into a USD physics articulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  convert-urdf-to-usd robot.urdf
  convert-urdf-to-usd robot.urdf -o out/robot.usda --floating-base
  convert-urdf-to-usd robot.urdf --decompose --coacd-quality fine
  convert-urdf-to-usd robot.urdf --home elbow_joint=1.2 --home wrist_joint=-0.3
        """
    )
    parser.add_argument('urdf_file', type=str, help='Path to the input URDF file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output .usd/.usda/.usdc path (default: <urdf_stem>.usd)')
    parser.add_argument('--floating-base', action='store_true',
                        help='Leave the base free (default: welded to the world)')
    parser.add_argument('--mobile-base', action='store_true',
                        help='Give the base planar x / y / yaw joints instead of '
                             'welding it')
    parser.add_argument('--self-collision', action='store_true',
                        help='Keep self-collision between the links (default: off)')
    parser.add_argument('--drive-stiffness', type=float, default=1000.0,
                        help='Position-drive stiffness of the joints')
    parser.add_argument('--drive-damping', type=float, default=200.0,
                        help='Position-drive damping of the joints')
    parser.add_argument('--no-auto-gain', action='store_true',
                        help='Use the same drive gains on every joint instead of '
                             'scaling them by each joint\'s gravity load')
    parser.add_argument('--home', action='append', default=[],
                        metavar='JOINT=VALUE',
                        help='Drive target of a joint (rad or m); repeatable')
    decompose = parser.add_mutually_exclusive_group()
    decompose.add_argument('--decompose', action='store_true',
                           help='CoACD-decompose every concave link\'s collision '
                                'mesh (default: one convex hull per link)')
    decompose.add_argument('--decompose-links', nargs='+', default=None,
                           metavar='LINK',
                           help='CoACD-decompose exactly these links (names or '
                                'substrings)')
    parser.add_argument('--coacd-quality', choices=['balanced', 'fine'],
                        default=None,
                        help='CoACD preset for --decompose / --decompose-links '
                             '(default: a tight fit)')
    args = parser.parse_args()

    urdf_path = Path(args.urdf_file)
    if not urdf_path.exists():
        print('error: URDF not found: {}'.format(urdf_path), file=sys.stderr)
        return 1
    try:
        home = _parse_home(args.home)
    except ValueError as e:
        print('error: {}'.format(e), file=sys.stderr)
        return 1
    try:
        import pxr  # noqa: F401

        from skrobot.export.usd import urdf_to_usd
    except ImportError:
        print('error: usd-core is required: pip install usd-core',
              file=sys.stderr)
        return 1
    out_path = args.output or str(urdf_path.with_suffix('.usd'))
    decompose_links = True if args.decompose else args.decompose_links
    coacd_params = None
    if args.coacd_quality is not None:
        coacd_params = {'quality': args.coacd_quality}

    stage = urdf_to_usd(
        str(urdf_path), out_path,
        floating_base=args.floating_base,
        mobile_base=args.mobile_base,
        self_collision=args.self_collision,
        drive_stiffness=args.drive_stiffness,
        drive_damping=args.drive_damping,
        auto_gain=not args.no_auto_gain,
        home_positions=home or None,
        decompose_links=decompose_links,
        coacd_params=coacd_params,
    )
    print('wrote USD: {} (defaultPrim {})'.format(
        out_path, stage.GetDefaultPrim().GetPath()))
    return 0


if __name__ == '__main__':
    sys.exit(main())

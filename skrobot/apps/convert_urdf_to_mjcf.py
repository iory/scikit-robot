#!/usr/bin/env python
"""
Convert a URDF into a MuJoCo MJCF model.

Walks the raw parsed URDF and emits an MJCF that MuJoCo can load directly --
handling fixed joints (weld), mimic joints (equality constraints), primitive and
mesh geometry (meshes exported to binary STL and decimated under MuJoCo's face
limit), inertials (regularized to be positive-definite), and per-joint position
actuators. Radians are pinned so angles match ROS.
"""

import argparse
from pathlib import Path
import sys

from skrobot.urdf import urdf_to_mjcf


def main():
    parser = argparse.ArgumentParser(
        description='Convert a URDF into a MuJoCo MJCF model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  convert-urdf-to-mjcf robot.urdf
  convert-urdf-to-mjcf robot.urdf -o out/robot.xml
  convert-urdf-to-mjcf robot.urdf --floating-base --no-actuators
        """
    )
    parser.add_argument('urdf_file', type=str, help='Path to the input URDF file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output .xml path (default: <urdf_stem>.xml)')
    parser.add_argument('--mesh-dir', type=str, default=None,
                        help='Directory for exported mesh assets '
                             '(default: <output_dir>/assets)')
    parser.add_argument('--floating-base', action='store_true',
                        help='Give the base link a free joint (default: welded)')
    parser.add_argument('--no-actuators', action='store_true',
                        help='Do not emit position actuators')
    parser.add_argument('--actuator-kp', type=float, default=50.0,
                        help='Proportional gain for position actuators')
    parser.add_argument('--no-ground', action='store_true',
                        help='Do not add a ground plane / light')
    parser.add_argument('--self-collision', action='store_true',
                        help='Keep full self-collision (default: robot collides '
                             'with ground but not itself)')
    parser.add_argument('--no-actuator-forcerange', action='store_true',
                        help='Do not cap actuator torque to the URDF effort limit')
    parser.add_argument('--convex-decompose-collision', action='store_true',
                        help='CoACD-decompose collision meshes into convex parts '
                             '(accurate but slow; needs the coacd package)')
    parser.add_argument('--coacd-quality', choices=['balanced', 'fine'],
                        default='balanced',
                        help='CoACD preset for --convex-decompose-collision')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    urdf_path = Path(args.urdf_file)
    if not urdf_path.exists():
        print('error: URDF not found: {}'.format(urdf_path), file=sys.stderr)
        return 1
    out_path = args.output or str(urdf_path.with_suffix('.xml'))

    urdf_to_mjcf(
        str(urdf_path), out_path,
        mesh_dir=args.mesh_dir,
        floating_base=args.floating_base,
        add_position_actuators=not args.no_actuators,
        actuator_kp=args.actuator_kp,
        add_ground=not args.no_ground,
        self_collision=args.self_collision,
        add_actuator_forcerange=not args.no_actuator_forcerange,
        convex_decompose_collision=args.convex_decompose_collision,
        coacd_quality=args.coacd_quality,
    )
    print('wrote MJCF: {}'.format(out_path))
    if args.verbose:
        try:
            import mujoco
            m = mujoco.MjModel.from_xml_path(out_path)
            print('MuJoCo loads OK: nq={} njnt={} nu={} nbody={} ngeom={}'.format(
                m.nq, m.njnt, m.nu, m.nbody, m.ngeom))
        except Exception as e:  # noqa: BLE001
            print('note: MuJoCo verification skipped/failed: {}'.format(e),
                  file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())

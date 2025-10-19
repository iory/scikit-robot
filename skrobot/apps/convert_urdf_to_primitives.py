#!/usr/bin/env python
"""
Convert URDF mesh geometries to primitive shapes.

This tool analyzes mesh geometries in a URDF file and converts them to
primitive shapes (box, cylinder, sphere) for more efficient physics
simulation and simplified robot models.
"""

import argparse
import logging
from pathlib import Path
import sys

from skrobot.urdf import convert_meshes_to_primitives


def main():
    parser = argparse.ArgumentParser(
        description='Convert URDF mesh geometries to primitive shapes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert both visual and collision meshes to primitives
  convert-urdf-to-primitives robot.urdf

  # Specify output file
  convert-urdf-to-primitives robot.urdf -o robot_primitives.urdf

  # Preserve visual meshes, only convert collision
  convert-urdf-to-primitives robot.urdf --preserve-visual

  # Force all meshes to be boxes
  convert-urdf-to-primitives robot.urdf --primitive-type box

  # Modify the input file in place
  convert-urdf-to-primitives robot.urdf --inplace

  # Process with verbose output
  convert-urdf-to-primitives robot.urdf -v
        """
    )

    parser.add_argument(
        'urdf_file',
        type=str,
        help='Path to the input URDF file'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to the output URDF file (default: input_primitives.urdf)'
    )

    parser.add_argument(
        '--inplace',
        action='store_true',
        help='Modify the input URDF file in place'
    )

    parser.add_argument(
        '--preserve-visual',
        action='store_true',
        help='Preserve visual meshes, only convert collision meshes'
    )

    parser.add_argument(
        '--preserve-collision',
        action='store_true',
        help='Preserve collision meshes, only convert visual meshes'
    )

    parser.add_argument(
        '--primitive-type',
        type=str,
        choices=['box', 'cylinder', 'sphere', 'auto'],
        default='auto',
        help='Force a specific primitive type for all meshes (default: auto-detect)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite output file if it exists'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(message)s')

    input_path = Path(args.urdf_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.urdf_file}' does not exist", file=sys.stderr)
        return 1

    if args.inplace and args.output:
        print("Error: Cannot use both --output and --inplace options", file=sys.stderr)
        return 1

    if args.preserve_visual and args.preserve_collision:
        print("Error: Cannot use both --preserve-visual and --preserve-collision", file=sys.stderr)
        return 1

    if args.inplace:
        output_path = None
    elif args.output:
        output_path = Path(args.output)
        if output_path.exists() and not args.force:
            print(f"Error: Output file '{output_path}' already exists. Use --force to overwrite.", file=sys.stderr)
            return 1
    else:
        output_path = input_path.parent / f"{input_path.stem}_primitives.urdf"
        if output_path.exists() and not args.force:
            print(f"Error: Output file '{output_path}' already exists. Use --force to overwrite.", file=sys.stderr)
            return 1

    convert_visual = not args.preserve_visual
    convert_collision = not args.preserve_collision

    primitive_type = None if args.primitive_type == 'auto' else args.primitive_type

    try:
        print(f"Loading URDF from: {input_path}")
        if args.preserve_visual:
            print("Preserving visual meshes, converting collision meshes to primitives...")
        elif args.preserve_collision:
            print("Preserving collision meshes, converting visual meshes to primitives...")
        else:
            print("Converting both visual and collision meshes to primitives...")

        if primitive_type:
            print(f"Forcing primitive type: {primitive_type}")
        else:
            print("Auto-detecting best primitive type for each mesh")

        modified_count = convert_meshes_to_primitives(
            str(input_path),
            str(output_path) if output_path else None,
            convert_visual=convert_visual,
            convert_collision=convert_collision,
            primitive_type=primitive_type
        )

        if modified_count > 0:
            print(f"Converted {modified_count} geometry elements to primitives")
        else:
            print("No mesh geometries found to convert")

        if args.inplace:
            print(f"Modified URDF saved in place: {input_path}")
        elif output_path:
            print(f"Modified URDF saved to: {output_path}")

        print("Conversion completed successfully!")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

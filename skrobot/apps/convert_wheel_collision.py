#!/usr/bin/env python
"""
Convert continuous joint child link collision meshes to cylinders.

This tool identifies continuous joints in a URDF file and converts
their child links' collision meshes to cylinder primitives. This is
particularly useful for wheel representations where cylinder collisions
are more efficient for physics simulations.
"""

import argparse
from pathlib import Path
import sys

from skrobot.urdf import convert_wheel_collisions_to_cylinders


def main():
    parser = argparse.ArgumentParser(
        description='Convert continuous joint child link collision meshes to cylinders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert wheel collisions in a URDF file
  convert-wheel-collision robot.urdf

  # Specify output file
  convert-wheel-collision robot.urdf -o robot_cylinder.urdf

  # Modify the input file in place
  convert-wheel-collision robot.urdf --inplace

  # Process with verbose output
  convert-wheel-collision robot.urdf -v
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
        help='Path to the output URDF file (default: input_cylinder_collision.urdf)'
    )

    parser.add_argument(
        '--inplace',
        action='store_true',
        help='Modify the input URDF file in place'
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

    # Configure logging
    import logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(message)s')

    # Check input file
    input_path = Path(args.urdf_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.urdf_file}' does not exist", file=sys.stderr)
        return 1

    # Check for conflicting options
    if args.output and args.inplace:
        print("Error: Cannot use both --output and --inplace options", file=sys.stderr)
        return 1

    # Determine output file
    if args.inplace:
        output_path = None  # Will modify in place
    elif args.output:
        output_path = Path(args.output)
        # Check if output exists
        if output_path.exists() and not args.force:
            print(f"Error: Output file '{output_path}' already exists. Use --force to overwrite.", file=sys.stderr)
            return 1
    else:
        output_path = input_path.parent / f"{input_path.stem}_cylinder_collision.urdf"
        # Check if output exists
        if output_path.exists() and not args.force:
            print(f"Error: Output file '{output_path}' already exists. Use --force to overwrite.", file=sys.stderr)
            return 1

    try:
        # Load and convert URDF
        print(f"Loading URDF from: {input_path}")
        print("Converting wheel collisions to cylinders...")

        # Call the conversion function
        modified_links = convert_wheel_collisions_to_cylinders(
            str(input_path),
            str(output_path) if output_path else None
        )

        if modified_links:
            print(f"Modified {len(modified_links)} links:")
            for link_name in modified_links:
                print(f"  - {link_name}")
        else:
            print("No continuous joint child links found to convert")

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

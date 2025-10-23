#!/usr/bin/env python3

import argparse
import os
import sys

from skrobot.urdf.scale_urdf import scale_urdf


def main():
    """Main entry point for the scale_urdf command.

    This command line tool allows users to scale a URDF model by a given
    factor, creating a miniature (scale < 1.0) or enlarged (scale > 1.0)
    version of the robot model.
    """
    parser = argparse.ArgumentParser(
        description='Scale a URDF file by a given factor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a half-scale miniature model
  scale-urdf robot.urdf robot_half.urdf --scale 0.5

  # Create a double-size model
  scale-urdf robot.urdf robot_2x.urdf --scale 2.0

  # Create a 10cm miniature (10% of original size)
  scale-urdf robot.urdf robot_mini.urdf --scale 0.1

  # Modify file in place
  scale-urdf robot.urdf --inplace --scale 0.5

Notes:
  This tool scales all geometric and physical properties:
  - Joint and link positions (xyz coordinates)
  - Mesh geometries (via scale attribute)
  - Primitive geometries (box, cylinder, sphere dimensions)
  - Mass (scaled by scale^3, assuming constant density)
  - Inertia tensors (scaled by scale^5)
        """)

    parser.add_argument(
        'input_urdf',
        help='Path to the input URDF file'
    )

    parser.add_argument(
        'output_urdf',
        nargs='?',
        help='Path for the output URDF file (default: <input>_scaled.urdf)'
    )

    parser.add_argument(
        '--scale',
        type=float,
        required=True,
        help='Scale factor (e.g., 0.5 for half-size, 2.0 for double-size)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--inplace', '-i',
        action='store_true',
        help='Modify the input file in place (ignores output_urdf argument)'
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_urdf):
        print(f"Error: Input file '{args.input_urdf}' not found.", file=sys.stderr)
        sys.exit(1)

    # Validate scale
    if args.scale <= 0:
        print(f"Error: Scale must be positive, got: {args.scale}", file=sys.stderr)
        sys.exit(1)

    # Determine output filename
    if args.inplace:
        output_urdf = args.input_urdf
        if args.verbose:
            print(f"Using inplace mode: will modify '{output_urdf}' directly")
    else:
        output_urdf = args.output_urdf
        if not output_urdf:
            base, ext = os.path.splitext(args.input_urdf)
            output_urdf = f"{base}_scaled{ext}"

        # Check if output file already exists
        if os.path.exists(output_urdf):
            print(f"Error: Output file '{output_urdf}' already exists.", file=sys.stderr)
            print("Please specify a different output file or remove the existing file.", file=sys.stderr)
            sys.exit(1)

    if args.verbose:
        print(f"Input URDF: {args.input_urdf}")
        print(f"Output URDF: {output_urdf}")
        print(f"Scale factor: {args.scale}")
        if args.scale < 1.0:
            percentage = args.scale * 100
            print(f"Creating miniature model at {percentage}% of original size")
        elif args.scale > 1.0:
            print(f"Creating enlarged model at {args.scale}x original size")
        else:
            print("Scale is 1.0 - creating identical copy")

    try:
        scale_urdf(
            input_file=args.input_urdf,
            output_file=output_urdf,
            scale=args.scale
        )

        if args.verbose:
            print(f"Successfully created scaled URDF: {output_urdf}")
        else:
            print(f"Scaled URDF saved to: {output_urdf}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

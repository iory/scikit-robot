#!/usr/bin/env python3

import argparse
import os
import sys

from skrobot.urdf import transform_urdf_with_world_link


def main():
    """Main entry point for the transform_urdf command.

    This command line tool allows users to add a world link with transform
    to a URDF file by directly manipulating the XML structure and saving
    the result to a new file.
    """
    parser = argparse.ArgumentParser(
        description='Add a transformed world link to a URDF file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a world link with default transform
  transform_urdf robot.urdf output.urdf

  # Add world link with translation
  transform_urdf robot.urdf output.urdf --x 1.0 --y 0.5 --z 0.2

  # Add world link with rotation (in degrees)
  transform_urdf robot.urdf output.urdf --roll 10 --pitch 20 --yaw 30

  # Add world link with custom name
  transform_urdf robot.urdf output.urdf --world-link-name my_world

  # Combined translation and rotation
  transform_urdf robot.urdf output.urdf --x 1.0 --z 0.5 --yaw 45

  # Modify file in place
  transform_urdf robot.urdf --inplace --x 1.0 --yaw 30
        """)

    parser.add_argument(
        'input_urdf',
        help='Path to the input URDF file'
    )

    parser.add_argument(
        'output_urdf',
        nargs='?',
        help='Path for the output URDF file (default: <input>_transformed.urdf)'
    )

    # Translation arguments
    parser.add_argument(
        '--x',
        type=float, default=0.0,
        help='Translation in X (meters). Default: 0.0'
    )
    parser.add_argument(
        '--y',
        type=float, default=0.0,
        help='Translation in Y (meters). Default: 0.0'
    )
    parser.add_argument(
        '--z',
        type=float, default=0.0,
        help='Translation in Z (meters). Default: 0.0'
    )

    # Rotation arguments (in degrees)
    parser.add_argument(
        '--roll',
        type=float, default=0.0,
        help='Rotation around X-axis (degrees). Default: 0.0'
    )
    parser.add_argument(
        '--pitch',
        type=float, default=0.0,
        help='Rotation around Y-axis (degrees). Default: 0.0'
    )
    parser.add_argument(
        '--yaw',
        type=float, default=0.0,
        help='Rotation around Z-axis (degrees). Default: 0.0'
    )

    # World link name
    parser.add_argument(
        '--world-link-name',
        default='world',
        help='Name for the new world link. Default: "world"'
    )

    # Verbose output
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    # Inplace option
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

    # Determine output filename
    if args.inplace:
        output_urdf = args.input_urdf
        if args.verbose:
            print(f"Using inplace mode: will modify '{output_urdf}' directly")
    else:
        output_urdf = args.output_urdf
        if not output_urdf:
            base, ext = os.path.splitext(args.input_urdf)
            output_urdf = f"{base}_transformed{ext}"

        # Check if output file already exists
        if os.path.exists(output_urdf):
            print(f"Error: Output file '{output_urdf}' already exists.", file=sys.stderr)
            print("Please specify a different output file or remove the existing file.", file=sys.stderr)
            sys.exit(1)

    if args.verbose:
        print(f"Input URDF: {args.input_urdf}")
        print(f"Output URDF: {output_urdf}")
        print(f"World link name: {args.world_link_name}")
        print(f"Transform: x={args.x}, y={args.y}, z={args.z}")
        print(f"Rotation: roll={args.roll}°, pitch={args.pitch}°, yaw={args.yaw}°")

    try:
        transform_urdf_with_world_link(
            input_file=args.input_urdf,
            output_file=output_urdf,
            x=args.x, y=args.y, z=args.z,
            roll=args.roll, pitch=args.pitch, yaw=args.yaw,
            world_link_name=args.world_link_name
        )

        if args.verbose:
            print(f"Successfully created transformed URDF: {output_urdf}")
        else:
            print(f"Transformed URDF saved to: {output_urdf}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

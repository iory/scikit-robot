#!/usr/bin/env python

import argparse
import os
import sys

from skrobot.urdf import extract_sub_urdf
from skrobot.utils.urdf import URDF


def main():
    """Main entry point for the extract_sub_urdf command.

    This command line tool allows users to extract a sub-URDF from a larger URDF
    by specifying a root link and optionally a target link.

    """
    parser = argparse.ArgumentParser(
        description='Extract a sub-URDF from a root link',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all links from 'r_shoulder_pan_link' downward
  extract_sub_urdf robot.urdf r_shoulder_pan_link output.urdf

  # Extract only the path from 'torso_lift_link' to 'head_tilt_link'
  extract_sub_urdf robot.urdf torso_lift_link output.urdf --to head_tilt_link

  # List all available links first
  extract_sub_urdf robot.urdf --list

  # Extract sub-URDF without fixed joints
  extract_sub_urdf robot.urdf base_link output.urdf --no-fixed-joints

  # Extract with verbose output
  extract_sub_urdf robot.urdf root_link output.urdf --verbose
        """)

    parser.add_argument(
        'input_urdf',
        type=str,
        help='Path to the input URDF file')

    parser.add_argument(
        'root_link',
        type=str,
        nargs='?',
        help='Name of the root link for the sub-URDF')

    parser.add_argument(
        'output_urdf',
        type=str,
        nargs='?',
        help='Path to the output URDF file')

    parser.add_argument(
        '--to',
        type=str,
        dest='to_link',
        help='Target link name. If specified, extract only the path from root_link to this link')

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available links in the URDF and exit')

    parser.add_argument(
        '--no-fixed-joints',
        action='store_true',
        help='Exclude fixed joints from the sub-URDF')

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output')

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Overwrite output file if it exists')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_urdf):
        print("Error: Input URDF file '{}' not found".format(
            args.input_urdf), file=sys.stderr)
        sys.exit(1)

    try:
        # Load URDF for listing
        if args.verbose or args.list:
            print("Loading URDF: {}".format(args.input_urdf))

        urdf = URDF.load(args.input_urdf)

        # If --list is specified, show all links and exit
        if args.list:
            robot_name = urdf.name
            links = list(urdf.link_map.keys())
            base_link_name = urdf._base_link.name if urdf._base_link else "Unknown"

            print("Robot name: {}".format(robot_name))
            print("Current root link: {}".format(base_link_name))
            print("Total links: {}".format(len(links)))
            print("\nAll links:")
            for i, link_name in enumerate(links, 1):
                marker = " (current root)" if link_name == base_link_name else ""
                print("  {:2d}. {}{}".format(i, link_name, marker))
            return

        # Validate required arguments
        if not args.root_link:
            print("Error: root_link is required when not using --list",
                  file=sys.stderr)
            parser.print_help()
            sys.exit(1)

        if not args.output_urdf:
            print("Error: output_urdf is required when not using --list",
                  file=sys.stderr)
            parser.print_help()
            sys.exit(1)

        # Check if output file already exists
        args.output_urdf = os.path.abspath(args.output_urdf)
        if os.path.exists(args.output_urdf) and not args.force:
            print("Error: Output file '{}' already exists. "
                  "Use --force to overwrite.".format(args.output_urdf),
                  file=sys.stderr)
            sys.exit(1)

        # Validate the root link
        if args.root_link not in urdf.link_map:
            print("Error: Link '{}' not found in URDF".format(
                args.root_link), file=sys.stderr)
            print("Available links: {}".format(
                ', '.join(urdf.link_map.keys())), file=sys.stderr)
            sys.exit(1)

        # Validate the to_link if specified
        if args.to_link and args.to_link not in urdf.link_map:
            print("Error: Target link '{}' not found in URDF".format(
                args.to_link), file=sys.stderr)
            print("Available links: {}".format(
                ', '.join(urdf.link_map.keys())), file=sys.stderr)
            sys.exit(1)

        # Show extraction parameters
        if args.verbose:
            print("Root link: {}".format(args.root_link))
            if args.to_link:
                print("Target link: {}".format(args.to_link))
                print("Mode: Extract path from root to target")
            else:
                print("Mode: Extract all descendants of root")
            print("Keep fixed joints: {}".format(not args.no_fixed_joints))
            print("Output file: {}".format(args.output_urdf))

        # Perform the extraction
        if args.verbose:
            print("Extracting sub-URDF...")

        extract_sub_urdf(
            input_urdf_path=args.input_urdf,
            root_link_name=args.root_link,
            to_link_name=args.to_link,
            output_urdf_path=args.output_urdf,
            keep_fixed_joints=not args.no_fixed_joints
        )

        # Verify the result
        if args.verbose:
            print("Verifying result...")

        result_urdf = URDF.load(args.output_urdf)
        num_links = len(result_urdf.links)
        num_joints = len(result_urdf.joints)

        if args.to_link:
            print("Successfully extracted path from '{}' to '{}'".format(
                args.root_link, args.to_link))
        else:
            print("Successfully extracted sub-URDF from root link '{}'".format(
                args.root_link))

        print("Sub-URDF contains {} links and {} joints".format(
            num_links, num_joints))
        print("Output saved to: {}".format(args.output_urdf))

        if args.verbose:
            print("\nIncluded links:")
            for i, link in enumerate(result_urdf.links, 1):
                print("  {:2d}. {}".format(i, link.name))

    except FileNotFoundError as e:
        print("Error: {}".format(e), file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print("Error: {}".format(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("Unexpected error: {}".format(e), file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

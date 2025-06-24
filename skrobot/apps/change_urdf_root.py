#!/usr/bin/env python3

import argparse
import os
import sys

from skrobot.urdf import URDFXMLRootLinkChanger


def main():
    """Main entry point for the change_urdf_root command.

    This command line tool allows users to change the root link of a URDF file
    by directly manipulating the XML structure and saving the result
    to a new file.

    """
    parser = argparse.ArgumentParser(
        description='Change the root link of a URDF file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Change root link to 'base_link' and save to new file
  change_urdf_root robot.urdf base_link output.urdf

  # List all available links first
  change_urdf_root robot.urdf --list

  # Change root link with verbose output
  change_urdf_root robot.urdf new_root output.urdf --verbose
        """)

    parser.add_argument(
        'input_urdf',
        type=str,
        help='Path to the input URDF file')

    parser.add_argument(
        'new_root_link',
        type=str,
        nargs='?',
        help='Name of the new root link')

    parser.add_argument(
        'output_urdf',
        type=str,
        nargs='?',
        help='Path to the output URDF file')

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available links in the URDF and exit')

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
        # Create the root link changer
        if args.verbose:
            print("Loading URDF: {}".format(args.input_urdf))

        changer = URDFXMLRootLinkChanger(args.input_urdf)

        # If --list is specified, show all links and exit
        if args.list:
            current_root = changer.get_current_root_link()
            links = changer.list_links()

            print("Current root link: {}".format(current_root))
            print("Total links: {}".format(len(links)))
            print("\nAll links:")
            for i, link_name in enumerate(links, 1):
                marker = " (current root)" if link_name == current_root else ""
                print("  {:2d}. {}{}".format(i, link_name, marker))
            return

        # Validate required arguments
        if not args.new_root_link:
            print("Error: new_root_link is required when not using --list",
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

        # Validate the new root link
        available_links = changer.list_links()
        if args.new_root_link not in available_links:
            print("Error: Link '{}' not found in URDF".format(
                args.new_root_link), file=sys.stderr)
            print("Available links: {}".format(', '.join(available_links)),
                  file=sys.stderr)
            sys.exit(1)

        # Show current state
        current_root = changer.get_current_root_link()
        if args.verbose:
            print("Current root link: {}".format(current_root))
            print("New root link: {}".format(args.new_root_link))
            print("Output file: {}".format(args.output_urdf))

        # Check if the new root is the same as current
        if args.new_root_link == current_root:
            if args.verbose:
                print("Warning: New root link is "
                      + "the same as current root link")

        # Perform the root link change
        if args.verbose:
            print("Changing root link...")

        changer.change_root_link(args.new_root_link, args.output_urdf)

        # Verify the result
        if args.verbose:
            print("Verifying result...")

        result_changer = URDFXMLRootLinkChanger(args.output_urdf)
        actual_root = result_changer.get_current_root_link()

        if actual_root == args.new_root_link:
            print("Successfully changed root link from '{}' "
                  "to '{}'".format(current_root, args.new_root_link))
            print("Modified URDF saved to: {}".format(args.output_urdf))
        else:
            print("Failed to change root link. Expected "
                  "'{}', got '{}'".format(args.new_root_link, actual_root),
                  file=sys.stderr)
            sys.exit(1)

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

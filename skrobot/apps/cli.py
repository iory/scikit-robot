#!/usr/bin/env python

import argparse
import os
import sys


def get_available_apps():
    """Dynamically discover available apps in the apps directory."""
    apps_dir = os.path.dirname(__file__)
    apps = {}

    # App metadata: (help_text, version_requirement)
    app_metadata = {
        'visualize_urdf': ('Visualize URDF model', None),
        'convert_urdf_mesh': ('Convert URDF mesh files', (3, 6)),
        'convert_urdf_to_primitives': ('Convert URDF meshes to primitive shapes', None),
        'modularize_urdf': ('Modularize URDF files', None),
        'change_urdf_root': ('Change URDF root link', None),
        'transform_urdf': ('Add world link with transform to URDF', None),
        'visualize_mesh': ('Visualize mesh file', None),
        'urdf_hash': ('Calculate URDF hash', None),
        'convert_wheel_collision': ('Convert wheel collision model', None),
        'extract_sub_urdf': ('Extract sub-URDF from a root link', None),
        'generate_robot_class': ('Generate robot class from URDF geometry', None),
    }

    for filename in os.listdir(apps_dir):
        if filename.endswith('.py') and filename != '__init__.py' and filename != 'cli.py':
            app_name = filename[:-3]  # Remove .py extension

            # Check if app exists and has main function
            try:
                module_path = f'skrobot.apps.{app_name}'
                module = __import__(module_path, fromlist=['main'])
                if hasattr(module, 'main'):
                    # Convert underscore to hyphen for command name
                    command_name = app_name.replace('_', '-')

                    # Get metadata
                    help_text, version_req = app_metadata.get(app_name, (f'Run {command_name}', None))

                    # Check version requirement
                    if version_req is None or (sys.version_info.major, sys.version_info.minor) >= version_req:
                        apps[command_name] = {
                            'module': module_path,
                            'help': help_text
                        }
            except ImportError:
                continue

    return apps


def main():
    parser = argparse.ArgumentParser(
        prog='skr',
        description='scikit-robot CLI tool'
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands'
    )

    # Dynamically add subcommands
    available_apps = get_available_apps()
    for command_name, app_info in available_apps.items():
        app_parser = subparsers.add_parser(
            command_name,
            help=app_info['help'],
            add_help=False
        )
        app_parser.set_defaults(
            func=lambda args, module=app_info['module']: run_app(f'{module}:main', args)
        )

    # Parse arguments
    args, unknown = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Pass remaining arguments to the subcommand
    sys.argv = [args.command] + unknown
    args.func(args)


def run_app(module_path, args):
    module_name, func_name = module_path.split(':')
    module = __import__(module_name, fromlist=[func_name])
    func = getattr(module, func_name)
    func()


if __name__ == '__main__':
    main()

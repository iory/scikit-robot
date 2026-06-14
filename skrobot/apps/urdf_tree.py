#!/usr/bin/env python

import argparse
import json as json_module
import os
import sys

from skrobot.urdf import kinematic_tree
from skrobot.urdf import print_urdf_tree
from skrobot.urdf import validate_urdf_structure


def _color(text, code, enabled):
    if not enabled:
        return text
    return '\033[{}m{}\033[0m'.format(code, text)


def _colorize_summary(result, enabled):
    text = str(result)
    if not enabled:
        return text
    text = text.replace(
        'All validation checks passed!',
        '✓ All validation checks passed!')
    out = []
    for line in text.splitlines():
        if line.startswith('✓ All validation'):
            out.append(_color(line, '32', True))   # green
        elif line.startswith('Validation Errors:'):
            out.append(_color(line, '31', True))    # red
        elif line.startswith('Warnings:'):
            out.append(_color(line, '33', True))    # yellow
        else:
            out.append(line)
    return '\n'.join(out)


def main():
    """Print a URDF link tree and validate its structure.

    Given a URDF, prints a validation summary (base/end links, connected
    components, cycles, naming, zero-velocity joints) followed by the link
    tree. Exit code is non-zero when the URDF is structurally invalid, so it
    is usable in CI.
    """
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, TypeError):
        pass

    parser = argparse.ArgumentParser(
        description='Print a URDF link tree and validate its structure.',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'urdf_file', type=str,
        help="Path to the URDF file, or '-' to read URDF XML from stdin.")
    parser.add_argument(
        '--full', action='store_true',
        help='Show the full link tree (every link) instead of the collapsed\n'
             'kinematic (movable-skeleton) tree.')
    parser.add_argument(
        '--no-collapse', action='store_true',
        help='In the kinematic tree, keep fixed-frame subtrees expanded.')
    parser.add_argument(
        '--no-annotate', action='store_true',
        help='Do not annotate edges with joint type/axis in the kinematic '
             'tree.')
    parser.add_argument(
        '--world', action='store_true',
        help='Show joint axes in the world frame at the initial pose and\n'
             'suffix each link with its world position @[x, y, z]. Loads the\n'
             'URDF into a RobotModel (meshes skipped) to compute kinematics.')
    parser.add_argument(
        '--validate-only', action='store_true',
        help='Print only the validation summary, not the tree.')
    parser.add_argument(
        '--tree-only', action='store_true',
        help='Print only the tree, not the validation summary.')
    parser.add_argument(
        '--json', action='store_true',
        help='Print the validation result as JSON (implies --validate-only).')
    parser.add_argument(
        '--no-color', action='store_true',
        help='Disable colored output.')
    args = parser.parse_args()

    if args.urdf_file == '-':
        source = sys.stdin.read()
    else:
        if not os.path.exists(args.urdf_file):
            print('Error: URDF file not found: {}'.format(args.urdf_file),
                  file=sys.stderr)
            sys.exit(2)
        source = args.urdf_file

    try:
        result = validate_urdf_structure(source)
    except Exception as e:
        print('Error: failed to parse URDF: {}'.format(e), file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json_module.dumps({
            'is_valid': result.is_valid,
            'errors': result.errors,
            'warnings': result.warnings,
            'summary': result.summary,
        }, indent=2))
        sys.exit(0 if result.is_valid else 1)

    use_color = sys.stdout.isatty() and not args.no_color

    if not args.tree_only:
        print(_colorize_summary(result, use_color))
        if not args.validate_only:
            print()

    if not args.validate_only:
        if args.full:
            print(print_urdf_tree(source, world=args.world))
        else:
            print(kinematic_tree(
                source,
                collapse_fixed=not args.no_collapse,
                annotate=not args.no_annotate,
                world=args.world))

    sys.exit(0 if result.is_valid else 1)


if __name__ == '__main__':
    main()

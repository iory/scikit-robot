#!/usr/bin/env python
"""Minimal example: Generate robot class from URDF geometry.

This script demonstrates how to automatically generate a Python robot class
with kinematic chain properties (right_arm, left_arm, etc.) from any robot model.

No LLM or API keys required - uses only URDF structure and naming conventions.

Usage:
    python generate_robot_class.py
    python generate_robot_class.py --robot pr2
    python generate_robot_class.py --robot panda --output /tmp/MyPanda.py
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Generate robot class from URDF geometry')
    parser.add_argument(
        '--robot', '-r',
        default='panda',
        choices=['panda', 'pr2', 'fetch', 'kuka'],
        help='Robot model to use (default: panda)')
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output path for generated Python file')
    args = parser.parse_args()

    # Import robot model
    if args.robot == 'panda':
        from skrobot.models import Panda
        robot = Panda()
    elif args.robot == 'pr2':
        from skrobot.models import PR2
        robot = PR2()
    elif args.robot == 'fetch':
        from skrobot.models import Fetch
        robot = Fetch()
    elif args.robot == 'kuka':
        from skrobot.models import Kuka
        robot = Kuka()

    print(f"Robot: {robot.name}")
    print(f"Links: {len(robot.link_list)}")
    print(f"Joints: {len(robot.joint_list)}")
    print()

    # Generate class
    from skrobot.urdf.robot_class_generator import generate_robot_class_from_geometry

    code = generate_robot_class_from_geometry(
        robot,
        output_path=args.output,
        class_name=f"My{robot.name.title().replace('_', '')}"
    )

    if args.output:
        print(f"Generated class saved to: {args.output}")
    else:
        print("Generated code:")
        print("-" * 60)
        print(code)


if __name__ == '__main__':
    main()

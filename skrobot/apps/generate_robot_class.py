#!/usr/bin/env python

"""Generate robot class from URDF geometry.

This tool generates Python robot class code with kinematic chain properties
(arm, right_arm, left_arm, etc.) from any URDF robot model.

No LLM or API keys required - uses only URDF structure and geometry.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Generate robot class from URDF geometry')
    parser.add_argument(
        'input_urdfpath',
        type=str,
        help='Input URDF path')
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for generated Python file')
    parser.add_argument(
        '--class-name', '-c',
        type=str,
        default=None,
        help='Class name for the generated class (default: auto-generated)')
    parser.add_argument(
        '--show-groups',
        action='store_true',
        help='Show detected groups without generating code')
    args = parser.parse_args()

    from skrobot.models.urdf import RobotModelFromURDF
    from skrobot.urdf.robot_class_generator import generate_groups_from_geometry
    from skrobot.urdf.robot_class_generator import generate_robot_class_from_geometry

    robot = RobotModelFromURDF(urdf_file=args.input_urdfpath)

    print(f"Robot: {robot.name}")
    print(f"Links: {len(robot.link_list)}")
    print(f"Joints: {len(robot.joint_list)}")
    print()

    if args.show_groups:
        groups, end_effectors, end_coords_info, robot_name = \
            generate_groups_from_geometry(robot)
        print("Detected groups:")
        for group_name, group_data in groups.items():
            links = group_data.get('links', [])
            tip = group_data.get('tip_link', 'N/A')
            print(f"  {group_name}: {len(links)} links, tip={tip}")
        print()
        print("End coordinates:")
        for group_name, ec_info in end_coords_info.items():
            parent = ec_info.get('parent_link', 'N/A')
            pos = ec_info.get('pos', [0, 0, 0])
            print(f"  {group_name}: parent={parent}, pos={pos}")
        return

    code = generate_robot_class_from_geometry(
        robot,
        output_path=args.output,
        class_name=args.class_name,
        urdf_path=args.input_urdfpath,
    )

    if args.output:
        print(f"Generated class saved to: {args.output}")
    else:
        print("Generated code:")
        print("-" * 60)
        print(code)


if __name__ == '__main__':
    main()

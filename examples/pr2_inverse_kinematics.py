#!/usr/bin/env python

import argparse
import time

import numpy as np

import skrobot
from skrobot.utils.visualization import ik_visualization


def demonstrate_revert_if_fail(robot_model, target_coords, link_list, viewer,
                               no_ik_visualization):
    """Demonstrate the difference between revert_if_fail=True and False"""
    import numpy as np

    print("\n" + "=" * 60)
    print("Testing revert_if_fail=False with unreachable target")
    print("=" * 60)

    # Get the correct end-effector position after first IK
    actual_end_pos = robot_model.rarm_end_coords.worldpos()
    actual_end_rot = robot_model.rarm_end_coords.worldrot()

    # Visualize current end-effector position with green coordinate frame
    current_ee_axis = skrobot.model.Axis(
        axis_radius=0.008,
        axis_length=0.12,
        pos=actual_end_pos,
        rot=actual_end_rot
    )
    # Set green color for current end-effector
    current_ee_axis.set_color([0, 255, 0])  # RGB for green
    viewer.add(current_ee_axis)

    # Define an unreachable target (too far away)
    unreachable_pos = [0.8, -0.8, 1.2]  # Beyond robot's reach
    print("Attempting unreachable target: {}".format(unreachable_pos))
    print("Current end-effector position: {}".format(actual_end_pos))

    unreachable_coords = skrobot.coordinates.Coordinates(unreachable_pos, [0, 0, 0])

    # Visualize unreachable target with a different color (red)
    unreachable_axis = skrobot.model.Axis(
        axis_radius=0.012,
        axis_length=0.18,
        pos=unreachable_coords.translation,
        rot=unreachable_coords.rotation
    )
    # Set red color for unreachable target
    unreachable_axis.set_color([255, 0, 0])  # RGB for red
    viewer.add(unreachable_axis)
    viewer.redraw()
    print("Unreachable target visualized with red coordinate frame")

    # First attempt with default revert_if_fail=True
    print("\n1. Standard IK (revert_if_fail=True):")
    result_standard = robot_model.inverse_kinematics(
        unreachable_coords,
        link_list=link_list,
        move_target=robot_model.rarm_end_coords,
        rotation_axis=True,
        revert_if_fail=True  # Default behavior
    )

    standard_end_pos = robot_model.rarm_end_coords.worldpos()
    success_standard = result_standard is not False and result_standard is not None
    print("   Result: {}".format("Success" if success_standard else "Failed"))
    print("   End-effector position: {}".format(standard_end_pos))
    print("   Distance to target: {:.3f}m".format(np.linalg.norm(standard_end_pos - unreachable_pos)))

    # Reset to a known good position (the successful target from earlier)
    robot_model.inverse_kinematics(
        target_coords,  # The original successful target
        link_list=link_list,
        move_target=robot_model.rarm_end_coords,
        rotation_axis=True
    )

    viewer.redraw()

    # Second attempt with revert_if_fail=False
    print("\n2. Progressive IK (revert_if_fail=False):")
    if not no_ik_visualization:
        with ik_visualization(viewer, sleep_time=0.5):
            result_progressive = robot_model.inverse_kinematics(
                unreachable_coords,
                link_list=link_list,
                move_target=robot_model.rarm_end_coords,
                rotation_axis=True,
                revert_if_fail=False  # Keep partial progress
            )
    else:
        result_progressive = robot_model.inverse_kinematics(
            unreachable_coords,
            link_list=link_list,
            move_target=robot_model.rarm_end_coords,
            rotation_axis=True,
            revert_if_fail=False  # Keep partial progress
        )

    progressive_end_pos = robot_model.rarm_end_coords.worldpos()
    progressive_end_rot = robot_model.rarm_end_coords.worldrot()
    success_progressive = result_progressive is not False and result_progressive is not None
    print("   Result: {}".format("Success" if success_progressive else "Failed (but kept progress)"))
    print("   End-effector position: {}".format(progressive_end_pos))
    print("   Distance to target: {:.3f}m".format(np.linalg.norm(progressive_end_pos - unreachable_pos)))

    # Visualize progressive IK result with cyan coordinate frame
    progressive_result_axis = skrobot.model.Axis(
        axis_radius=0.006,
        axis_length=0.10,
        pos=progressive_end_pos,
        rot=progressive_end_rot
    )
    # Set cyan color for progressive IK result
    progressive_result_axis.set_color([0, 255, 255])  # RGB for cyan
    viewer.add(progressive_result_axis)

    # Compare distances
    standard_distance = np.linalg.norm(standard_end_pos - unreachable_pos)
    progressive_distance = np.linalg.norm(progressive_end_pos - unreachable_pos)

    print("\n" + "-" * 50)
    print("Comparison:")
    print("   Standard IK distance to target: {:.3f}m".format(standard_distance))
    print("   Progressive IK distance to target: {:.3f}m".format(progressive_distance))

    if progressive_distance < standard_distance:
        improvement = standard_distance - progressive_distance
        print("   [OK] Progressive IK got {:.3f}m closer to target!".format(improvement))
    else:
        print("   Standard IK performed better in this case")

    viewer.redraw()
    print("\nAll IK demonstrations completed!")
    print("\n" + "=" * 60)
    print("VISUALIZATION LEGEND:")
    print("=" * 60)
    print("Red coordinate frame   : Unreachable target")
    print("Cyan coordinate frame  : Progressive IK result (revert_if_fail=False)")
    print("   Frame axes: Red=X, Green=Y, Blue=Z")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Simple PR2 inverse kinematics example.')
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help="Run in non-interactive mode (do not wait for user input)"
    )
    parser.add_argument(
        '--viewer', type=str,
        choices=['trimesh', 'pyrender'], default='trimesh',
        help='Choose the viewer type: trimesh or pyrender')
    parser.add_argument(
        '--no-ik-visualization',
        action='store_true',
        help="Disable inverse kinematics visualization during solving"
    )
    parser.add_argument(
        '--skip-revert-demo',
        action='store_true',
        help="Skip the revert_if_fail demonstration"
    )
    parser.add_argument(
        '--translation-tolerance', '--translation_tolerance',
        type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'),
        help='Translation tolerance per axis in meters (e.g., 0.05 0.05 0.05)'
    )
    parser.add_argument(
        '--rotation-tolerance', '--rotation_tolerance',
        type=float, nargs=3, default=None, metavar=('R', 'P', 'Y'),
        help='Rotation tolerance per axis in degrees (e.g., 10 10 10)'
    )
    args = parser.parse_args()

    # Process tolerance arguments
    translation_tolerance = args.translation_tolerance
    rotation_tolerance = None
    if args.rotation_tolerance is not None:
        rotation_tolerance = [np.deg2rad(v) for v in args.rotation_tolerance]

    # Create robot model
    robot_model = skrobot.models.PR2()
    robot_model.reset_pose()

    # Define joint list for right arm
    link_list = [
        robot_model.r_shoulder_pan_link,
        robot_model.r_shoulder_lift_link,
        robot_model.r_upper_arm_roll_link,
        robot_model.r_elbow_flex_link,
        robot_model.r_forearm_roll_link,
        robot_model.r_wrist_flex_link,
        robot_model.r_wrist_roll_link
    ]

    # Create viewer
    if args.viewer == 'trimesh':
        viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
    elif args.viewer == 'pyrender':
        viewer = skrobot.viewers.PyrenderViewer(resolution=(640, 480))

    viewer.add(robot_model)
    viewer.show()
    viewer.set_camera([np.deg2rad(45), -np.deg2rad(0),
                       np.deg2rad(135)], distance=2.5)

    # Define target position
    target_pos = [0.7, -0.2, 0.8]
    print("Solving inverse kinematics for target: {}".format(target_pos))
    if translation_tolerance is not None:
        print("Translation tolerance: {}m".format(translation_tolerance))
    if args.rotation_tolerance is not None:
        print("Rotation tolerance: {} degrees".format(args.rotation_tolerance))

    # Create target coordinates
    target_coords = skrobot.coordinates.Coordinates(target_pos, [0, 0, 0])

    # Visualize target position with Axis model
    target_axis = skrobot.model.Axis(
        axis_radius=0.01,
        axis_length=0.15,
        pos=target_coords.translation,
        rot=target_coords.rotation
    )
    viewer.add(target_axis)
    viewer.redraw()
    print("Target position visualized with coordinate frame")

    # Build IK kwargs
    ik_kwargs = {
        'link_list': link_list,
        'move_target': robot_model.rarm_end_coords,
        'rotation_axis': True,
    }
    if translation_tolerance is not None:
        ik_kwargs['translation_tolerance'] = translation_tolerance
    if rotation_tolerance is not None:
        ik_kwargs['rotation_tolerance'] = rotation_tolerance

    # Solve inverse kinematics with optional visualization
    if not args.no_ik_visualization:
        with ik_visualization(viewer, sleep_time=0.5):
            result = robot_model.inverse_kinematics(target_coords, **ik_kwargs)
    else:
        result = robot_model.inverse_kinematics(target_coords, **ik_kwargs)

    # Check if result is successful (could be False or an array)
    success = result is not False and result is not None
    if success:
        print("Successfully reached target!")
    else:
        print("Failed to reach target")

    viewer.redraw()
    print("First IK solving completed!")

    # Demonstrate revert_if_fail=False with an unreachable target
    if not args.skip_revert_demo:
        demonstrate_revert_if_fail(robot_model, target_coords, link_list, viewer,
                                   args.no_ik_visualization)

    if not args.no_interactive:
        print('==> Press [q] to close window')
        while viewer.is_active:
            time.sleep(0.1)
            viewer.redraw()
    viewer.close()
    time.sleep(1.0)


if __name__ == '__main__':
    main()

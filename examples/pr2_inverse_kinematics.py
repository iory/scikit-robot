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
    actual_end_pos = robot_model.right_arm_end_coords.worldpos()
    actual_end_rot = robot_model.right_arm_end_coords.worldrot()

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
        move_target=robot_model.right_arm_end_coords,
        rotation_mask=True,
        revert_if_fail=True  # Default behavior
    )

    standard_end_pos = robot_model.right_arm_end_coords.worldpos()
    success_standard = result_standard is not False and result_standard is not None
    print("   Result: {}".format("Success" if success_standard else "Failed"))
    print("   End-effector position: {}".format(standard_end_pos))
    print("   Distance to target: {:.3f}m".format(np.linalg.norm(standard_end_pos - unreachable_pos)))

    # Reset to a known good position (the successful target from earlier)
    robot_model.inverse_kinematics(
        target_coords,  # The original successful target
        link_list=link_list,
        move_target=robot_model.right_arm_end_coords,
        rotation_mask=True
    )

    viewer.redraw()

    # Second attempt with revert_if_fail=False
    print("\n2. Progressive IK (revert_if_fail=False):")
    if not no_ik_visualization:
        with ik_visualization(viewer, sleep_time=0.5):
            result_progressive = robot_model.inverse_kinematics(
                unreachable_coords,
                link_list=link_list,
                move_target=robot_model.right_arm_end_coords,
                rotation_mask=True,
                revert_if_fail=False  # Keep partial progress
            )
    else:
        result_progressive = robot_model.inverse_kinematics(
            unreachable_coords,
            link_list=link_list,
            move_target=robot_model.right_arm_end_coords,
            rotation_mask=True,
            revert_if_fail=False  # Keep partial progress
        )

    progressive_end_pos = robot_model.right_arm_end_coords.worldpos()
    progressive_end_rot = robot_model.right_arm_end_coords.worldrot()
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


def demonstrate_fullbody_ik(robot_model, link_list, viewer,
                            no_ik_visualization):
    """Demonstrate fullbody IK (use_base) reaching a target that is
    out of arm range without moving the mobile base."""
    print("\n" + "=" * 60)
    print("Fullbody IK: reaching a far target by moving the base")
    print("=" * 60)

    # Reset to a clean pose so the demo starts deterministic.
    robot_model.reset_pose()
    viewer.redraw()

    ee_pos0 = robot_model.right_arm_end_coords.worldpos().copy()
    base_pos0 = robot_model.root_link.worldpos().copy()

    # Target 0.8 m forward from the current end-effector pose — beyond
    # the right arm's reach without base motion.
    far_target_pos = ee_pos0 + np.array([0.8, 0.0, 0.0])
    far_target = skrobot.coordinates.Coordinates(far_target_pos, [0, 0, 0])
    print("Far target position: {}".format(far_target_pos))
    print("Initial base position: {}".format(base_pos0))

    far_target_axis = skrobot.model.Axis(
        axis_radius=0.012,
        axis_length=0.18,
        pos=far_target.translation,
        rot=far_target.rotation,
    )
    far_target_axis.set_color([255, 128, 0])  # orange
    viewer.add(far_target_axis)
    viewer.redraw()

    # 1) Arm-only IK — expected to fail because target is out of reach.
    print("\n1. Arm-only IK (use_base=False):")
    result_armonly = robot_model.inverse_kinematics(
        far_target,
        link_list=link_list,
        move_target=robot_model.right_arm_end_coords,
        rotation_mask=False,
        stop=100,
    )
    success_armonly = result_armonly is not False \
        and result_armonly is not None
    ee_armonly = robot_model.right_arm_end_coords.worldpos()
    err_armonly = np.linalg.norm(ee_armonly - far_target_pos)
    print("   Result: {}".format(
        "Success" if success_armonly else "Failed"))
    print("   End-effector position: {}".format(ee_armonly))
    print("   Distance to target: {:.3f}m".format(err_armonly))

    # Reset between attempts.
    robot_model.reset_pose()
    viewer.redraw()

    # 2) Fullbody IK with planar base — base translates/rotates on
    # the ground so the arm can reach.
    print("\n2. Fullbody IK (use_base='planar'):")
    if not no_ik_visualization:
        with ik_visualization(viewer, sleep_time=0.5):
            result_fullbody = robot_model.inverse_kinematics(
                far_target,
                link_list=link_list,
                move_target=robot_model.right_arm_end_coords,
                rotation_mask=False,
                stop=100,
                use_base='planar',
            )
    else:
        result_fullbody = robot_model.inverse_kinematics(
            far_target,
            link_list=link_list,
            move_target=robot_model.right_arm_end_coords,
            rotation_mask=False,
            stop=100,
            use_base='planar',
        )
    success_fullbody = result_fullbody is not False \
        and result_fullbody is not None
    ee_fullbody = robot_model.right_arm_end_coords.worldpos()
    base_fullbody = robot_model.root_link.worldpos()
    err_fullbody = np.linalg.norm(ee_fullbody - far_target_pos)
    base_moved = np.linalg.norm(base_fullbody - base_pos0)
    print("   Result: {}".format(
        "Success" if success_fullbody else "Failed"))
    print("   End-effector position: {}".format(ee_fullbody))
    print("   Distance to target: {:.3f}m".format(err_fullbody))
    print("   Base moved by: {:.3f}m".format(base_moved))

    viewer.redraw()
    print("\n" + "-" * 50)
    print("Comparison:")
    print("   Arm-only distance to target: {:.3f}m".format(err_armonly))
    print("   Fullbody  distance to target: {:.3f}m".format(err_fullbody))
    if err_fullbody < err_armonly:
        print("   [OK] Fullbody IK reached the target by moving the base.")
    print("\n" + "=" * 60)
    print("VISUALIZATION LEGEND (fullbody demo):")
    print("=" * 60)
    print("Orange coordinate frame: Far target (out of arm-only reach)")
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
        '--skip-fullbody-demo',
        action='store_true',
        help="Skip the fullbody IK (use_base) demonstration"
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
        'move_target': robot_model.right_arm_end_coords,
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

    # Demonstrate fullbody IK with a far target
    if not args.skip_fullbody_demo:
        demonstrate_fullbody_ik(robot_model, link_list, viewer,
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

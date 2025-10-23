#!/usr/bin/env python

import argparse
import time

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.models import Fetch
from skrobot.models import Nextage
from skrobot.models import Panda
from skrobot.models import PR2
from skrobot.models import R8_6


def parse_axis_constraint(axis_str):
    """Parse axis constraint string to appropriate format."""
    if axis_str.lower() == 'true':
        return True
    elif axis_str.lower() == 'false':
        return False
    else:
        # String like 'xy', 'xyz', 'z', etc.
        return axis_str.lower()


def main():
    parser = argparse.ArgumentParser(description='Advanced Batch IK Demo with axis constraints')
    parser.add_argument('--robot', type=str, default='pr2',
                        choices=['fetch', 'pr2', 'panda', 'r8_6', 'nextage'],
                        help='Robot model to use. Default: fetch')
    parser.add_argument('--rotation-axis', '--rotation_axis', '-r',
                        default='True',
                        help='Rotation axis constraints (True/False/xyz/xy/z/etc). Default: True')
    parser.add_argument('--translation-axis', '--translation_axis', '-t',
                        default='True',
                        help='Translation axis constraints (True/False/xyz/xy/z/etc). Default: True')
    parser.add_argument('--attempts-per-pose', '--attempts_per_pose', '-a',
                        type=int, default=50,
                        help='Number of attempts per pose with different random initial poses. Default: 50')
    parser.add_argument('--stop', '-s', type=int, default=100,
                        help='Maximum iterations per attempt. Default: 100')
    parser.add_argument('--thre', type=float, default=0.001,
                        help='Position error threshold in meters. Default: 0.001')
    parser.add_argument('--rthre', type=float, default=1.0,
                        help='Rotation error threshold in degrees. Default: 1.0')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Disable interactive visualization')
    parser.add_argument('--viewer', type=str,
                        choices=['pyrender', 'trimesh'], default='trimesh',
                        help='Choose the viewer type: trimesh or pyrender. Default: trimesh')

    args = parser.parse_args()

    # Parse axis constraints
    rotation_axis = parse_axis_constraint(args.rotation_axis)
    translation_axis = parse_axis_constraint(args.translation_axis)

    print("ADVANCED BATCH IK - WITH AXIS CONSTRAINTS")
    print("=" * 55)
    print("Configuration:")
    print(f"   Robot: {args.robot.upper()}")
    print(f"   Rotation axis: {rotation_axis}")
    print(f"   Translation axis: {translation_axis}")
    print(f"   Attempts per pose: {args.attempts_per_pose}")
    print(f"   Max iterations: {args.stop}")
    print(f"   Position threshold: {args.thre}m")
    print(f"   Rotation threshold: {args.rthre} degrees")

    # Initialize robot based on selection
    if args.robot == 'fetch':
        robot = Fetch()
        arm = robot.rarm
    elif args.robot == 'pr2':
        robot = PR2()
        arm = robot.rarm
    elif args.robot == 'panda':
        robot = Panda()
        arm = robot.rarm
    elif args.robot == 'r8_6':
        robot = R8_6()
        arm = robot.rarm
    elif args.robot == 'nextage':
        robot = Nextage()
        arm = robot.rarm

    robot.reset_pose()

    # Define target poses based on robot type
    if args.robot == 'r8_6':
        # R8_6-specific target poses (within reachable workspace)
        # R8_6 has z-axis range ~0.18-1.4m and typical reach ~0.5-1.0m from base
        target_poses = [
            Coordinates(pos=(0.85, -0.2, 0.85)).rotate(np.deg2rad(15), 'y'),
            Coordinates(pos=(0.90, -0.25, 0.90)).rotate(np.deg2rad(-10), 'z'),
            Coordinates(pos=(0.80, -0.15, 0.95)).rotate(np.deg2rad(20), 'x'),
            Coordinates(pos=(0.75, -0.30, 0.88)).rotate(np.deg2rad(10), 'y').rotate(np.deg2rad(5), 'z'),
            Coordinates(pos=(0.88, -0.22, 0.92)).rotate(np.deg2rad(-15), 'x'),
            Coordinates(pos=(0.82, -0.18, 0.87)).rotate(np.deg2rad(12), 'y').rotate(np.deg2rad(-8), 'z'),
            Coordinates(pos=(0.78, -0.28, 0.93)).rotate(np.deg2rad(-5), 'y'),
            Coordinates(pos=(0.46, -0.24, 0.89)).rotate(np.deg2rad(8), 'z').rotate(np.deg2rad(3), 'x'),
            Coordinates(pos=(0.56, -0.24, 0.89)),
        ]
    elif args.robot == 'nextage':
        target_poses = [
            Coordinates(pos=(0.2, -0.3, 0.0)).rotate(np.deg2rad(-25), 'z'),
            Coordinates(pos=(0.2, -0.3, -0.05)).rotate(np.deg2rad(-30), 'z'),
            Coordinates(pos=(0.35, -0.25, 0.05)).rotate(np.deg2rad(35), 'y').rotate(np.deg2rad(-20), 'z'),
            Coordinates(pos=(0.1, -0.1, 0.1)).rotate(np.deg2rad(90), 'x'),
            Coordinates(pos=(0.2, -0.1, 0.03)).rotate(np.deg2rad(45), 'x'),
            Coordinates(pos=(0.35, -0.15, 0.1)).rotate(np.deg2rad(-80), 'y').rotate(np.deg2rad(15), 'z'),
        ]
    else:
        # Default target poses for Fetch/PR2/Panda
        target_poses = [
            Coordinates(pos=(0.7, -0.2, 0.9)).rotate(np.deg2rad(30), 'y'),
            Coordinates(pos=(0.6, -0.3, 1.0)).rotate(np.deg2rad(-25), 'z'),
            Coordinates(pos=(0.8, -0.1, 0.8)).rotate(np.deg2rad(45), 'x'),
            Coordinates(pos=(0.5, -0.4, 1.1)).rotate(np.deg2rad(20), 'y').rotate(np.deg2rad(15), 'z'),
            Coordinates(pos=(0.65, -0.45, 0.95)).rotate(np.deg2rad(-30), 'x'),
            Coordinates(pos=(0.75, -0.25, 1.05)).rotate(np.deg2rad(35), 'y').rotate(np.deg2rad(-20), 'z'),
            Coordinates(pos=(0.55, -0.35, 0.85)).rotate(np.deg2rad(-15), 'y'),
            Coordinates(pos=(0.68, -0.38, 1.08)).rotate(np.deg2rad(25), 'z').rotate(np.deg2rad(10), 'x'),
        ]

    for i, coord in enumerate(target_poses):
        pos = coord.worldpos()

    print("\nStarting batch IK solving...")

    overall_start = time.time()
    # Use inverse_kinematics_defaults if available, otherwise specify explicitly
    ik_kwargs = {
        'move_target': arm.end_coords,
        'rotation_axis': rotation_axis,
        'translation_axis': translation_axis,
        'stop': args.stop,
        'thre': args.thre,
        'rthre': np.deg2rad(args.rthre),
        'attempts_per_pose': args.attempts_per_pose,
    }

    # For robots with inverse_kinematics_defaults, use those settings
    if hasattr(arm, 'inverse_kinematics_defaults'):
        ik_defaults = arm.inverse_kinematics_defaults
        if 'link_list' in ik_defaults:
            ik_kwargs['link_list'] = ik_defaults['link_list']

    solutions, success_flags, attempt_counts = robot.batch_inverse_kinematics(
        target_poses,
        **ik_kwargs
    )
    overall_time = time.time() - overall_start

    success_count = sum(success_flags)

    print("\nOVERALL PERFORMANCE:")
    print(f"   Total time: {overall_time:.3f}s")
    print(f"   Success rate: {success_count}/{len(target_poses)} ({success_count / len(target_poses) * 100:.1f}%)")
    print(f"   Average time per solved pose: {overall_time / max(success_count, 1):.3f}s")

    print("\nATTEMPT BREAKDOWN:")
    for i, (success, attempts) in enumerate(zip(success_flags, attempt_counts)):
        status = "[SOLVED]" if success else "[FAILED]"
        print(f"   Pose {i}: {status} after {attempts} attempts")

    if success_count > 0:
        print("\nVerifying successful solutions...")

        successful_solutions = []
        successful_indices = []
        original_angles = robot.angle_vector()

        for i, (solution, success) in enumerate(zip(solutions, success_flags)):
            if success:
                successful_solutions.append(solution)
                successful_indices.append(i)

                # Test the solution
                robot.angle_vector(solution)
                achieved_coords = arm.end_coords.copy_worldcoords()
                achieved_pos = achieved_coords.worldpos()
                target_pos = target_poses[i].worldpos()

                # Calculate position error considering only the constrained axes
                pos_error = achieved_pos - target_pos
                constrained_pos_error = pos_error.copy()

                # Apply translation constraints to error calculation
                if translation_axis is False:
                    constrained_pos_error = np.array([0, 0, 0])
                elif isinstance(translation_axis, str):
                    # For mirror constraints (xm, ym, zm), don't zero out any translation errors
                    # since we want to see the full error after mirroring is applied
                    if translation_axis.lower() not in ['xm', 'ym', 'zm']:
                        # Standard axis constraints
                        if 'x' not in translation_axis.lower():
                            constrained_pos_error[0] = 0
                        if 'y' not in translation_axis.lower():
                            constrained_pos_error[1] = 0
                        if 'z' not in translation_axis.lower():
                            constrained_pos_error[2] = 0

                pos_error_norm = np.linalg.norm(constrained_pos_error)

                # Calculate rotation error considering rotation constraints
                rot_error = 0.0
                rot_error_details = ""

                if rotation_axis is not False:
                    # Calculate rotation error using difference_rotation method
                    dif_rot = achieved_coords.difference_rotation(target_poses[i], rotation_axis=rotation_axis)
                    rot_error = np.linalg.norm(dif_rot)

                    # Calculate individual axis errors for detailed analysis
                    achieved_x = achieved_coords.axis('x')
                    achieved_y = achieved_coords.axis('y')
                    achieved_z = achieved_coords.axis('z')
                    target_x = target_poses[i].axis('x')
                    target_y = target_poses[i].axis('y')
                    target_z = target_poses[i].axis('z')

                    x_angle_error = np.rad2deg(np.arccos(np.clip(np.dot(achieved_x, target_x), -1, 1)))
                    y_angle_error = np.rad2deg(np.arccos(np.clip(np.dot(achieved_y, target_y), -1, 1)))
                    z_angle_error = np.rad2deg(np.arccos(np.clip(np.dot(achieved_z, target_z), -1, 1)))

                    rot_error_details = (
                        f" | Rot: {np.rad2deg(rot_error):.2f}° "
                        f"(X:{x_angle_error:.1f}° Y:{y_angle_error:.1f}° Z:{z_angle_error:.1f}°)"
                    )

                print(f"  [OK] Pose {i}: Pos = {pos_error_norm:.4f}m{rot_error_details}")

                # Restore for next test
                robot.angle_vector(original_angles)

        if not args.no_interactive:
            print(f"\nAttempting visualization of {len(successful_solutions)} solutions...")

        if not args.no_interactive:
            try:
                if args.viewer == 'pyrender':
                    from skrobot.viewers import PyrenderViewer
                    viewer = PyrenderViewer(update_interval=1 / 30.0)
                else:  # trimesh
                    from skrobot.viewers import TrimeshSceneViewer
                    viewer = TrimeshSceneViewer(update_interval=1 / 30.0)

                print(f"Adding {len(target_poses)} target poses as coordinate frames...")
                axis_objects = []
                for i, target_coord in enumerate(target_poses):
                    is_solved = success_flags[i]

                    if is_solved:
                        axis = Axis.from_coords(
                            target_coord,
                            axis_radius=0.008,
                            axis_length=0.12,
                            alpha=0.3
                        )
                        status = "[SOLVED]"
                    else:
                        axis = Axis.from_coords(
                            target_coord,
                            axis_radius=0.008,
                            axis_length=0.12,
                            alpha=0.3
                        )
                        axis.set_color([1.0, 0.0, 0.0, 1.0])
                        status = "[UNSOLVED]"

                    axis_objects.append(axis)
                    viewer.add(axis)
                    pos = target_coord.worldpos()
                    print(f"  Target {i}: {status} - Axis at [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

                robot.reset_pose()
                viewer.add(robot)

                end_effector_axis = Axis.from_coords(
                    arm.end_coords,
                    axis_radius=0.006,
                    axis_length=0.10,
                    alpha=1.0
                )
                viewer.add(end_effector_axis)

                print(f"Added robot (will cycle through {len(successful_solutions)} solutions)")
                print("Added end-effector axis (RGB colors) to show actual robot pose")

                viewer.show()

                print("\n3D VISUALIZATION ACTIVE!")
                print("=" * 50)
                print("Mouse: Rotate and zoom")
                print("Close window to continue")
                print("Solved poses: Normal colored axes (RGB = XYZ)")
                print("Unsolved poses: Red axes")
                print("End-effector: RGB colored axis (shows actual robot pose)")
                print("Note: Axes shown regardless of rotation_axis setting for visual comparison")
                print(f"Robot cycles through {len(successful_solutions)} successful solutions only")
                print("Each solution displays for 0.5 seconds")
                print("Current target axis will be highlighted, others will be dimmed")

            # Animation loop - cycle through solutions
                try:
                    solution_idx = 0
                    last_change_time = time.time()

                    while viewer.is_active:
                        current_time = time.time()

                        if current_time - last_change_time > 0.5:
                            if len(successful_solutions) > 0:
                                robot.angle_vector(successful_solutions[solution_idx])
                                orig_idx = successful_indices[solution_idx]

                                end_effector_axis.newcoords(arm.end_coords)

                                for i, axis in enumerate(axis_objects):
                                    if success_flags[i]:
                                        if i == orig_idx:
                                            axis.set_alpha(1.0)
                                        else:
                                            axis.set_alpha(0.3)

                                target_pos = target_poses[orig_idx].worldpos()
                                achieved_pos = arm.end_coords.worldpos()
                                pos_error = np.linalg.norm(achieved_pos - target_pos)

                                print(f"Showing solution {solution_idx + 1}/{len(successful_solutions)} "
                                      f"(target pose {orig_idx}) | Error: {pos_error:.4f}m")

                                solution_idx = (solution_idx + 1) % len(successful_solutions)
                                last_change_time = current_time

                        viewer.redraw()
                        time.sleep(0.05)

                    print("\nVisualization completed")

                except KeyboardInterrupt:
                    print("\nVisualization interrupted")

            except ImportError:
                print("3D visualization not available")
                print("Install with: pip install pyrender trimesh")
            except Exception as e:
                print(f"Visualization error: {e}")

        else:
            print("No successful solutions to visualize")
    else:
        print("Interactive visualization skipped (--no-interactive)")


if __name__ == '__main__':
    main()

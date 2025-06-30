#!/usr/bin/env python

import argparse
import time

import numpy as np

import skrobot
from skrobot.coordinates.base import lerp_coordinates
from skrobot.coordinates.base import slerp_coordinates


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Demonstrate SLERP vs LERP trajectory interpolation'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='enter interactive shell'
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help="Run in non-interactive mode (do not wait for user input)"
    )
    parser.add_argument(
        '--method',
        choices=['slerp', 'lerp', 'both'],
        default='both',
        help='Interpolation method to demonstrate'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=20,
        help='Number of interpolation steps'
    )
    parser.add_argument(
        '--simple-rotation',
        action='store_true',
        help='Use simple rotation for comparison (less difference)'
    )
    args = parser.parse_args()

    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(800, 600))
    plane = skrobot.model.Box(
        extents=(2, 2, 0.01), face_colors=(0.75, 0.75, 0.75)
    )
    viewer.add(plane)

    start_coords = skrobot.coordinates.Coordinates(
        pos=[0.3, -0.5, 0.5],
        rot=[0, 0, 0]
    )
    end_coords = skrobot.coordinates.Coordinates(
        pos=[0.8, 0.5, 0.8]
    )
    if args.simple_rotation:
        # Simple rotation - less difference between SLERP and LERP
        end_coords.rotate(np.pi / 2, 'z')  # 90 degrees around Z
        end_coords.rotate(np.pi / 4, 'x')  # 45 degrees around X
    else:
        # Complex rotation - significant difference between SLERP and LERP
        end_coords.rotate(np.pi * 0.7, 'z')  # 126 degrees around Z
        end_coords.rotate(np.pi * 0.5, 'x')  # 90 degrees around X
        end_coords.rotate(np.pi * 0.3, 'y')  # 54 degrees around Y

    def create_coordinate_frame(coords, name="", scale=0.1):
        """Create visual representation of coordinate frame using Axis model"""
        axis = skrobot.model.Axis(axis_radius=0.005, axis_length=scale,
                                  pos=coords.translation, rot=coords.rotation)
        return [axis]

    print("=== SLERP vs LERP Trajectory Interpolation Demo ===")
    print("Rotation type: {}".format('Simple' if args.simple_rotation else 'Complex'))
    print("Start position: {}".format(start_coords.translation))
    print("End position: {}".format(end_coords.translation))
    print("Start quaternion: {}".format(start_coords.quaternion))
    print("End quaternion: {}".format(end_coords.quaternion))

    start_frame_objects = create_coordinate_frame(start_coords, "start", 0.15)
    end_frame_objects = create_coordinate_frame(end_coords, "end", 0.15)

    for obj in start_frame_objects + end_frame_objects:
        viewer.add(obj)

    n_steps = args.steps
    t_values = np.linspace(0, 1, n_steps)

    trajectory_objects = []
    slerp_trajectory = []
    lerp_trajectory = []

    if args.method in ['slerp', 'both']:
        print("\nGenerating SLERP trajectory with {} steps...".format(n_steps))
        for t in t_values:
            interp_coords = slerp_coordinates(start_coords, end_coords, t)
            slerp_trajectory.append(interp_coords)
            axis = skrobot.model.Axis(axis_radius=0.004, axis_length=0.06,
                                      pos=interp_coords.translation,
                                      rot=interp_coords.rotation)
            trajectory_objects.append(axis)
            viewer.add(axis)

    if args.method in ['lerp', 'both']:
        print("Generating LERP trajectory with {} steps...".format(n_steps))
        offset = np.array([0, 0, -0.15]) if args.method == 'both' else np.array([0, 0, 0])
        for t in t_values:
            interp_coords = lerp_coordinates(start_coords, end_coords, t)
            lerp_trajectory.append(interp_coords)

            # Create transparent axis using Axis class with alpha parameter
            transparent_axis = skrobot.model.Axis(
                axis_radius=0.002,
                axis_length=0.04,
                alpha=0.3,
                pos=interp_coords.translation + offset,
                rot=interp_coords.rotation
            )
            trajectory_objects.append(transparent_axis)
            viewer.add(transparent_axis)

    # Analyze trajectory smoothness for each method
    print("\n=== Trajectory Smoothness Analysis ===")

    def calculate_consecutive_differences(trajectory, name):
        """Calculate angular differences between consecutive frames"""
        if len(trajectory) < 2:
            return

        max_angle_diff = 0
        max_diff_idx = 0
        angle_diffs = []

        for i in range(len(trajectory) - 1):
            # Calculate rotation difference between consecutive frames
            curr = trajectory[i]
            next = trajectory[i + 1]

            # Compute relative rotation
            relative_rot = np.dot(curr.rotation.T, next.rotation)
            trace = np.clip(np.trace(relative_rot), -1, 3)
            angle_diff = np.arccos((trace - 1) / 2)
            angle_diff_deg = np.rad2deg(angle_diff)
            angle_diffs.append(angle_diff_deg)

            if angle_diff > max_angle_diff:
                max_angle_diff = angle_diff
                max_diff_idx = i

        # Statistics
        mean_diff = np.mean(angle_diffs)
        std_diff = np.std(angle_diffs)

        print("\n{} Trajectory:".format(name))
        print("  Mean angular step: {:.2f}°".format(mean_diff))
        print("  Std deviation: {:.2f}°".format(std_diff))
        print("  Max angular step: {:.2f}° (between t={:.2f} and t={:.2f})".format(
            np.rad2deg(max_angle_diff), t_values[max_diff_idx], t_values[max_diff_idx + 1]))

        # Visualize maximum step
        if max_angle_diff > 0.01:
            # Add a thick line at maximum step
            line_points = np.array([
                trajectory[max_diff_idx].translation + (offset if name == "LERP" else np.zeros(3)),
                trajectory[max_diff_idx + 1].translation + (offset if name == "LERP" else np.zeros(3))
            ])

            # Yellow line for maximum step
            max_line = skrobot.model.LineString(
                points=line_points,
                color=[1, 1, 0, 0.8]  # Yellow
            )
            viewer.add(max_line)

            # Add sphere marker at the midpoint
            midpoint = (line_points[0] + line_points[1]) / 2
            marker = skrobot.model.Sphere(
                radius=0.015,
                color=[1, 1, 0, 0.8]  # Yellow
            )
            marker.translate(midpoint + np.array([0, 0, 0.05]))
            viewer.add(marker)

        return angle_diffs

    if args.method in ['slerp', 'both'] and len(slerp_trajectory) > 1:
        calculate_consecutive_differences(slerp_trajectory, "SLERP")

    if args.method in ['lerp', 'both'] and len(lerp_trajectory) > 1:
        calculate_consecutive_differences(lerp_trajectory, "LERP")

    # Compare SLERP vs LERP if both are available
    if args.method == 'both':
        print("\n=== SLERP vs LERP Comparison ===")

        # Calculate differences between SLERP and LERP at each point
        max_diff_idx = 0
        max_angle_diff = 0

        for i, t in enumerate(t_values):
            slerp_result = slerp_trajectory[i]
            lerp_result = lerp_trajectory[i]

            # Calculate angular difference
            relative_rot = np.dot(slerp_result.rotation.T, lerp_result.rotation)
            trace = np.clip(np.trace(relative_rot), -1, 3)
            angle_diff = np.arccos((trace - 1) / 2)

            if angle_diff > max_angle_diff:
                max_angle_diff = angle_diff
                max_diff_idx = i

            # Visualize differences with connecting lines
            angle_diff_deg = np.rad2deg(angle_diff)
            if angle_diff_deg > 0.5:  # Show lines for noticeable differences
                line_points = np.array([
                    slerp_result.translation,
                    lerp_result.translation + offset
                ])

                # Color based on difference magnitude
                color_intensity = min(angle_diff_deg / 15.0, 1.0)
                line_color = [color_intensity, 0, 1 - color_intensity, 0.3]

                line = skrobot.model.LineString(
                    points=line_points,
                    color=line_color
                )
                viewer.add(line)

        print("  Maximum rotation difference: {:.1f}° at t={:.2f}".format(
            np.rad2deg(max_angle_diff), t_values[max_diff_idx]))

        # Visualize maximum difference point
        if max_angle_diff > 0.01:
            max_slerp = slerp_trajectory[max_diff_idx]
            max_lerp = lerp_trajectory[max_diff_idx]

            # Red indicator at maximum difference
            indicator = skrobot.model.Sphere(
                radius=0.02,
                color=[1, 0, 0, 0.7]
            )
            midpoint = (max_slerp.translation + max_lerp.translation + offset) / 2
            indicator.translate(midpoint + np.array([0, 0, 0.1]))
            viewer.add(indicator)

    print("\n=== Visualization Legend ===")
    print("  * Yellow lines/spheres: Maximum angular step within each trajectory")
    print("  * Blue-to-red lines: SLERP vs LERP differences (if both shown)")
    print("  * Red sphere: Maximum SLERP vs LERP difference point")

    # Set camera view
    viewer.set_camera(angles=[np.deg2rad(30), 0, np.deg2rad(45)], distance=2.5)
    viewer.show()

    if args.interactive:
        print('''>>> # Trajectory Interpolation Demo
>>> # Large coordinate frames: Start and end poses
>>> # Medium coordinate frames: SLERP trajectory (opaque axes)
>>> # Small transparent coordinate frames: LERP trajectory (offset)
>>> #
>>> # Available variables:
>>> # - start_coords: Starting coordinate frame
>>> # - end_coords: Ending coordinate frame
>>> # - slerp_trajectory: List of SLERP interpolated coordinates
>>> # - lerp_trajectory: List of LERP interpolated coordinates
>>> # - viewer: 3D viewer
>>> #
>>> # Example usage:
>>> start_coords.translation
>>> end_coords.rotation
>>> len(slerp_trajectory) if 'slerp_trajectory' in locals() else 0
>>> viewer.redraw()
''')

        # Make trajectories available in interactive mode
        if args.method in ['slerp', 'both']:
            globals()['slerp_trajectory'] = slerp_trajectory
        if args.method in ['lerp', 'both']:
            globals()['lerp_trajectory'] = lerp_trajectory
        globals()['start_coords'] = start_coords
        globals()['end_coords'] = end_coords
        globals()['viewer'] = viewer

        import IPython
        IPython.embed()
    else:
        print('\n=== Visualization Guide ===')
        print('  * Large axes: Start (lower) and End (upper) coordinate frames')
        print('  * Opaque axes: SLERP trajectory (spherical linear interpolation)')
        print('  * Transparent axes: LERP trajectory (linear interpolation, offset below)')
        print('  * Colors: Red=X, Green=Y, Blue=Z axes')

        if args.method == 'both' and not args.simple_rotation:
            print('\n️  Notice: With complex rotations, SLERP and LERP produce noticeably different paths!')
            print('   SLERP follows the shortest rotation path on the unit sphere.')
            print('   LERP linearly interpolates rotation matrices (less natural).')

        if not args.no_interactive:
            print('\n==> Press [q] to close window')
            while not viewer.has_exit:
                time.sleep(0.1)
                viewer.redraw()
        else:
            print('\n==> Demo complete')
            time.sleep(3)


if __name__ == '__main__':
    main()

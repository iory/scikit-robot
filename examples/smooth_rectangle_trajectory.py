#!/usr/bin/env python
"""Example: Generate smooth trajectory for rectangular path using PR2.

This example demonstrates how to use plan_smooth_trajectory_ik to generate
a smooth joint trajectory through 4 corners of a rectangle, avoiding
sudden joint angle jumps.
"""

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.models import PR2
from skrobot.planner import compute_trajectory_smoothness
from skrobot.planner import plan_smooth_trajectory_ik
from skrobot.viewers import PyrenderViewer


def main():
    # Initialize robot
    print("Loading PR2 model...")
    robot = PR2()
    robot.reset_manip_pose()

    # Define 4 corners of a rectangle in front of the robot
    # The rectangle is in the YZ plane at x=0.5
    center_x = 0.55
    y_range = 0.15
    z_center = 0.95
    z_range = 0.12

    # Set orientation for end-effector (pointing forward)
    orientation = Coordinates().rotate(np.pi / 2, 'y')

    corners = [
        Coordinates(pos=[center_x, y_range, z_center - z_range],
                    rot=orientation.rotation),
        Coordinates(pos=[center_x, -y_range, z_center - z_range],
                    rot=orientation.rotation),
        Coordinates(pos=[center_x, -y_range, z_center + z_range],
                    rot=orientation.rotation),
        Coordinates(pos=[center_x, y_range, z_center + z_range],
                    rot=orientation.rotation),
    ]

    print("\nRectangle corners:")
    for i, c in enumerate(corners):
        print(f"  Corner {i + 1}: pos={c.worldpos()}")

    # Generate smooth trajectory
    print("\nGenerating smooth trajectory...")
    trajectory, coords, success, info = plan_smooth_trajectory_ik(
        robot,
        robot.rarm_end_coords,
        corners,
        link_list=robot.rarm.link_list,
        n_divisions=8,
        closed_loop=True,
        rotation_axis=False,  # Position only, no rotation constraint
        position_tolerance=0.015,
        rotation_tolerance=np.deg2rad(30.0),
        slsqp_options={'ftol': 1e-5, 'disp': True, 'maxiter': 300},
        verbose=True
    )

    # Compute smoothness metrics
    print("\n=== Trajectory Analysis ===")
    initial_metrics = compute_trajectory_smoothness(info['initial_trajectory'])
    optimized_metrics = compute_trajectory_smoothness(trajectory)

    print("\nInitial trajectory (seeded IK):")
    print(f"  Max velocity: {np.rad2deg(initial_metrics['max_velocity']):.2f} deg/step")
    print(f"  Max acceleration: {np.rad2deg(initial_metrics['max_acceleration']):.2f} deg/step^2")

    print("\nOptimized trajectory (SQP):")
    print(f"  Max velocity: {np.rad2deg(optimized_metrics['max_velocity']):.2f} deg/step")
    print(f"  Max acceleration: {np.rad2deg(optimized_metrics['max_acceleration']):.2f} deg/step^2")

    if initial_metrics['max_acceleration'] > 0:
        improvement = (1 - optimized_metrics['max_acceleration'] /
                       initial_metrics['max_acceleration']) * 100
        print(f"\nAcceleration reduction: {improvement:.1f}%")

    # Position/rotation error analysis
    print("\n=== Task Space Errors ===")
    print(f"Max position error: {np.max(info['position_errors']) * 1000:.2f} mm")
    print(f"Max rotation error: {np.rad2deg(np.max(info['rotation_errors'])):.2f} deg")

    # Setup viewer
    print("\n=== Visualization ===")
    viewer = PyrenderViewer()
    viewer.add(robot)

    from skrobot.model.primitives import Sphere

    # Add spheres for corners (red)
    for i, c in enumerate(corners):
        corner_sphere = Sphere(radius=0.02, color=[1.0, 0.2, 0.2, 1.0])
        corner_sphere.translate(c.worldpos())
        viewer.add(corner_sphere)

    # Get end-effector positions for the trajectory
    joint_list = robot.joint_list_from_link_list(
        robot.rarm.link_list, ignore_fixed_joint=True)

    trajectory_points = []
    for joint_angles in trajectory:
        for j, joint in enumerate(joint_list):
            joint.joint_angle(joint_angles[j])
        trajectory_points.append(robot.rarm_end_coords.worldpos().copy())

    # Add small spheres to show the path
    for i, pos in enumerate(trajectory_points):
        sphere = Sphere(radius=0.008, color=[0.2, 0.6, 1.0, 0.7])
        sphere.translate(pos)
        viewer.add(sphere)

    viewer.show()

    # Reset to first pose
    for j, joint in enumerate(joint_list):
        joint.joint_angle(trajectory[0, j])
    viewer.redraw()

    # Animation loop
    import time
    delay = 0.05

    print("Playing trajectory...")
    print('Press q to exit the viewer window.')
    while viewer.is_active:
        for i, joint_angles in enumerate(trajectory):
            for j, joint in enumerate(joint_list):
                joint.joint_angle(joint_angles[j])
            viewer.redraw()
            time.sleep(delay)


if __name__ == '__main__':
    main()

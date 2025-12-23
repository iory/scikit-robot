#!/usr/bin/env python
"""Example demonstrating face and line constraint IK.

This example shows how to use FaceTarget and LineTarget to solve IK
where the end-effector reaches any point on a face (plane) or line segment.
"""

import time

import numpy as np

import skrobot
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.model.primitives import Cylinder
from skrobot.model.primitives import Sphere
from skrobot.planner import FaceTarget
from skrobot.planner import LineTarget
from skrobot.planner import solve_ik_with_constraint
from skrobot.viewers import PyrenderViewer


def add_target_marker(viewer, pos, color=[1.0, 0.2, 0.2, 1.0], radius=0.015):
    """Add a sphere marker at target position."""
    sphere = Sphere(radius=radius, color=color)
    sphere.translate(pos)
    viewer.add(sphere)
    return sphere


def add_ee_trail(viewer, positions, color=[0.2, 0.8, 0.2, 0.8], radius=0.008):
    """Add spheres showing end-effector trajectory."""
    spheres = []
    for pos in positions:
        sphere = Sphere(radius=radius, color=color)
        sphere.translate(pos)
        viewer.add(sphere)
        spheres.append(sphere)
    return spheres


def main():
    # Load robot model
    robot = skrobot.models.Panda()
    robot.reset_manip_pose()

    # Get end-effector and link list
    end_coords = robot.rarm_end_coords
    link_list = [
        robot.panda_link1,
        robot.panda_link2,
        robot.panda_link3,
        robot.panda_link4,
        robot.panda_link5,
        robot.panda_link6,
        robot.panda_link7,
    ]
    joint_list = robot.joint_list_from_link_list(link_list, ignore_fixed_joint=True)

    # Create viewer
    print("Initializing viewer...")
    viewer = PyrenderViewer()
    viewer.add(robot)

    # Add coordinate axes at origin
    origin_axis = Axis(axis_radius=0.005, axis_length=0.1)
    viewer.add(origin_axis)

    viewer.show()
    time.sleep(0.5)  # Let viewer initialize

    # ===================
    # Example 1: Face Target
    # ===================
    print("\n" + "=" * 50)
    print("Example 1: Face Target (reaching a wall)")
    print("=" * 50)

    # Check initial end-effector pose
    initial_ee_pos = end_coords.worldpos()
    initial_ee_rot = end_coords.worldrot()
    print(f"Initial end-effector position: {initial_ee_pos}")

    # Add marker for initial position
    add_target_marker(viewer, initial_ee_pos.copy(),
                      color=[0.5, 0.5, 0.5, 0.5], radius=0.012)

    # Create a wall (face) in front of the robot
    wall = Box(extents=[0.01, 0.4, 0.4], face_colors=[1.0, 0.6, 0.6, 0.5])
    wall_pos = np.array([0.45, 0.0, 0.5])
    wall.translate(wall_pos)
    viewer.add(wall)

    # Define face target
    face_target = FaceTarget(
        center=wall_pos - np.array([0.005, 0, 0]),
        normal=np.array([-1.0, 0.0, 0.0]),
        x_length=0.18,
        y_length=0.18,
        margin=0.02,
        approach_axis='-z'
    )

    # Add markers for face corners
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            corner = (face_target.center
                      + dx * face_target.x_length * face_target.x_axis
                      + dy * face_target.y_length * face_target.y_axis)
            add_target_marker(viewer, corner, color=[1.0, 0.3, 0.3, 0.8], radius=0.01)

    viewer.redraw()

    # Solve IK to reach the face
    print("\nSolving IK for face target...")
    success = solve_ik_with_constraint(
        robot,
        face_target,
        move_target=end_coords,
        link_list=link_list,
        thre=0.02,
        rthre=np.deg2rad(10.0),
        stop=200,
        verbose=True
    )

    if success:
        ee_pos = end_coords.worldpos()
        error = face_target.compute_error(ee_pos, end_coords.worldrot())
        print(f"\n  Success!")
        print(f"  End-effector position: {ee_pos}")
        print(f"  Distance from face plane: {abs(error[0]):.4f} m")

        # Add marker for final position (green)
        add_target_marker(viewer, ee_pos.copy(), color=[0.2, 1.0, 0.2, 1.0], radius=0.015)

        # Add axis at end-effector to show orientation
        ee_axis = Axis(axis_radius=0.003, axis_length=0.05)
        ee_axis.newcoords(end_coords.copy_worldcoords())
        viewer.add(ee_axis)
    else:
        print("  IK failed!")

    viewer.redraw()
    print("\nPress Enter to continue to Line Target example...")
    input()

    # ===================
    # Example 2: Line Target
    # ===================
    print("\n" + "=" * 50)
    print("Example 2: Line Target (reaching a rail)")
    print("=" * 50)

    # Reset robot pose
    robot.reset_manip_pose()
    viewer.redraw()
    time.sleep(0.3)

    # Create a rail (line segment) parallel to y-axis
    rail_start = np.array([0.35, -0.15, 0.5])
    rail_end = np.array([0.35, 0.15, 0.5])
    rail_center = (rail_start + rail_end) / 2
    rail_length = np.linalg.norm(rail_end - rail_start)

    rail = Cylinder(radius=0.012, height=rail_length,
                    face_colors=[0.6, 0.6, 1.0, 0.8])
    rail.translate(rail_center)
    rail.rotate(np.pi / 2, 'x')
    viewer.add(rail)

    # Add markers at rail endpoints
    add_target_marker(viewer, rail_start, color=[0.3, 0.3, 1.0, 1.0], radius=0.015)
    add_target_marker(viewer, rail_end, color=[0.3, 0.3, 1.0, 1.0], radius=0.015)

    # Add initial position marker
    add_target_marker(viewer, end_coords.worldpos().copy(),
                      color=[0.5, 0.5, 0.5, 0.5], radius=0.012)

    viewer.redraw()

    # Define line target
    line_target = LineTarget(
        start=rail_start,
        end=rail_end,
        margin=0.02,
        direction_axis=None,
        normal_tolerance=np.deg2rad(30.0)
    )

    # Solve IK to reach the line
    print("Solving IK for line target...")
    success = solve_ik_with_constraint(
        robot,
        line_target,
        move_target=end_coords,
        link_list=link_list,
        thre=0.02,
        rthre=np.deg2rad(30.0),
        stop=200,
        verbose=True
    )

    if success:
        ee_pos = end_coords.worldpos()
        error = line_target.compute_error(ee_pos, end_coords.worldrot())
        t = np.dot(ee_pos - rail_start, rail_end - rail_start) / rail_length**2
        closest_point = rail_start + t * (rail_end - rail_start)
        dist_from_line = np.linalg.norm(ee_pos - closest_point)

        print(f"\n  Success!")
        print(f"  End-effector position: {ee_pos}")
        print(f"  Distance from line: {dist_from_line:.4f} m")
        print(f"  Position along line (0-1): {t:.2f}")

        # Add marker for final position
        add_target_marker(viewer, ee_pos.copy(), color=[0.2, 1.0, 0.2, 1.0], radius=0.015)

        # Add marker for closest point on line
        add_target_marker(viewer, closest_point, color=[1.0, 1.0, 0.2, 1.0], radius=0.012)

        # Add axis at end-effector
        ee_axis = Axis(axis_radius=0.003, axis_length=0.05)
        ee_axis.newcoords(end_coords.copy_worldcoords())
        viewer.add(ee_axis)
    else:
        print("  IK failed!")

    viewer.redraw()
    print("\nPress Enter to continue to multi-point face reaching...")
    input()

    # ===================
    # Example 3: Multiple Points on Face
    # ===================
    print("\n" + "=" * 50)
    print("Example 3: Reaching multiple points on a face")
    print("=" * 50)

    # Reset robot pose
    robot.reset_manip_pose()
    viewer.redraw()
    time.sleep(0.3)

    # Create a larger face (table surface)
    table = Box(extents=[0.5, 0.5, 0.01], face_colors=[0.8, 0.6, 0.4, 0.6])
    table_pos = np.array([0.4, 0.0, 0.25])
    table.translate(table_pos)
    viewer.add(table)

    # Define face target on top of table
    table_face = FaceTarget(
        center=table_pos + np.array([0, 0, 0.005]),
        normal=np.array([0.0, 0.0, 1.0]),
        x_length=0.2,
        y_length=0.2,
        margin=0.03,
        normal_tolerance=np.deg2rad(30.0),
        approach_axis='-z'
    )

    # Define multiple target points on the face (grid pattern)
    grid_points = []
    for x in [-0.1, 0, 0.1]:
        for y in [-0.1, 0, 0.1]:
            point = table_face.center + x * table_face.x_axis + y * table_face.y_axis
            grid_points.append(point)
            # Add small marker for each target point
            add_target_marker(viewer, point, color=[1.0, 0.5, 0.0, 0.8], radius=0.01)

    viewer.redraw()

    # Try to reach each point and collect trajectory
    trajectory_positions = []
    successful_points = 0

    print(f"\nAttempting to reach {len(grid_points)} points on the table...")

    for i, target_point in enumerate(grid_points):
        # Create a face target centered at this point
        point_target = FaceTarget(
            center=target_point,
            normal=np.array([0.0, 0.0, 1.0]),
            x_length=0.02,  # Small area
            y_length=0.02,
            margin=0.0,
            normal_tolerance=np.deg2rad(45.0),
            approach_axis='-z'
        )

        success = solve_ik_with_constraint(
            robot,
            point_target,
            move_target=end_coords,
            link_list=link_list,
            thre=0.03,
            rthre=np.deg2rad(50.0),
            stop=100,
            verbose=False
        )

        if success:
            successful_points += 1
            ee_pos = end_coords.worldpos().copy()
            trajectory_positions.append(ee_pos)

            # Add trajectory marker
            add_target_marker(viewer, ee_pos, color=[0.2, 0.8, 0.2, 0.7], radius=0.008)
            viewer.redraw()
            time.sleep(0.2)

    print(f"\nReached {successful_points}/{len(grid_points)} points successfully")

    # Draw lines connecting trajectory points
    if len(trajectory_positions) > 1:
        print(f"Trajectory covers {len(trajectory_positions)} positions")

    viewer.redraw()
    print("\nPress Enter to exit...")
    input()

    print("\nDone!")


if __name__ == '__main__':
    main()

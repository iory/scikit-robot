#!/usr/bin/env python
"""Demo of RobotCollisionChecker for collision detection.

This example demonstrates:
1. Setting up RobotCollisionChecker with robot links
2. Adding world obstacles (primitives and SDF)
3. Computing collision distances
4. Visualizing collision spheres with color feedback

Usage:
    python robot_collision_checker_demo.py
    python robot_collision_checker_demo.py --no-interactive
"""

import argparse
import time

import numpy as np

import skrobot
from skrobot.collision import RobotCollisionChecker
from skrobot.model.primitives import Box
from skrobot.model.primitives import Sphere


def main():
    parser = argparse.ArgumentParser(
        description='RobotCollisionChecker demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--no-interactive', action='store_true',
        help='Run without waiting for user input'
    )
    args = parser.parse_args()

    # Create robot
    print("Loading PR2 robot...")
    robot = skrobot.models.PR2()
    robot.init_pose()

    # Define collision links (right arm)
    coll_links = [
        robot.r_upper_arm_link,
        robot.r_forearm_link,
        robot.r_gripper_palm_link,
        robot.r_gripper_r_finger_link,
        robot.r_gripper_l_finger_link,
    ]

    # Create RobotCollisionChecker
    print("\nSetting up RobotCollisionChecker...")
    checker = RobotCollisionChecker(robot)

    # Add collision links with auto geometry selection
    # Auto mode uses capsules for elongated shapes, spheres for compact shapes
    for link in coll_links:
        checker.add_link(link, geometry_type='auto')

    # Show what geometry type was selected for each link
    print(f"  Added {checker.n_feature} collision geometries from {len(coll_links)} links:")
    for lg in checker.link_geometries:
        geom_type = type(lg.geometry).__name__
        print(f"    {lg.link.name}: {geom_type}")

    # Create world obstacles using skrobot.model.primitives
    # These are automatically converted to collision geometry
    print("\nAdding world obstacles...")

    # Sphere obstacle (red) - in front of robot
    sphere_obs = Sphere(radius=0.12)
    sphere_obs.set_color([255, 0, 0, 180])
    sphere_obs.translate([0.75, -0.4, 0.85])
    checker.add_world_obstacle(sphere_obs, use_sdf=False)  # Use analytical
    print(f"  Added Sphere at {sphere_obs.worldpos()}, radius=0.12")

    # Box obstacle (green, with SDF) - to the side
    box_obs = Box(extents=[0.25, 0.25, 0.4], with_sdf=True)
    box_obs.set_color([0, 255, 0, 180])
    box_obs.translate([0.55, 0.35, 0.75])
    checker.add_world_obstacle(box_obs)  # Uses SDF by default
    print(f"  Added Box at {box_obs.worldpos()}, extents=[0.25, 0.25, 0.4]")

    print(f"\n  Total: {len(checker.world_obstacles)} analytical obstacles, "
          f"{len(checker._world_sdfs)} SDF obstacles")

    # Setup viewer
    print("\nStarting viewer...")
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(800, 600))
    viewer.add(robot)
    viewer.add(sphere_obs)
    viewer.add(box_obs)

    # Add collision spheres to viewer (with color feedback)
    checker.add_coll_spheres_to_viewer(viewer)

    viewer.show()
    viewer.set_camera([0, 0, np.pi / 2])

    # Define test configurations
    link_list = [
        robot.r_shoulder_pan_link, robot.r_shoulder_lift_link,
        robot.r_upper_arm_roll_link, robot.r_elbow_flex_link,
        robot.r_forearm_roll_link, robot.r_wrist_flex_link,
        robot.r_wrist_roll_link,
    ]
    joint_list = [link.joint for link in link_list]

    # Test configurations: some safe, some in collision
    # Joint order: shoulder_pan, shoulder_lift, upper_arm_roll,
    #              elbow_flex, forearm_roll, wrist_flex, wrist_roll
    test_configs = [
        # Safe configurations
        ("Safe (arm up)", np.array([0.0, -0.5, 0.0, -1.8, 0.0, -0.5, 0.0])),
        ("Safe (arm side)", np.array([1.2, 0.0, 0.0, -1.5, 0.0, -0.3, 0.0])),
        # Approach and hit sphere (red, front-right)
        ("Approach sphere", np.array([0.7, 0.1, 0.0, -1.3, 0.0, -0.3, 0.0])),
        ("Hit sphere!", np.array([0.55, 0.45, 0.0, -0.5, 0.0, -0.2, 0.0])),
        # Approach and hit box (green, front-left)
        ("Approach box", np.array([-0.4, 0.3, 0.0, -1.3, 0.0, -0.4, 0.0])),
        ("Hit box!", np.array([-0.35, 0.45, 0.0, -0.9, 0.0, -0.3, 0.0])),
    ]

    print("\n" + "=" * 60)
    print("Testing collision detection at different configurations")
    print("=" * 60)
    print("Yellow spheres = safe, Red spheres = collision")
    print()

    for name, config in test_configs:
        # Set robot configuration
        for joint, angle in zip(joint_list, config):
            joint.joint_angle(angle)

        # Compute collision distances
        min_dist = checker.compute_min_distance()
        world_dists = checker.compute_world_collision_distances()

        # Update visualization colors
        checker.update_color()
        viewer.redraw()

        # Print results
        status = "COLLISION" if min_dist < 0 else "Safe"
        color = "\033[91m" if min_dist < 0 else "\033[92m"
        reset = "\033[0m"
        print(f"{name:20s}: {color}{status:10s}{reset} (min_dist = {min_dist:+.4f})")

        if len(world_dists) > 0:
            # Show distances to each obstacle type
            n_spheres = checker.n_feature
            n_analytical = len(checker.world_obstacles)
            n_sdf = len(checker._world_sdfs)

            if n_analytical > 0:
                analytical_dists = world_dists[:n_spheres * n_analytical]
                print(f"  Analytical obstacles: min = {np.min(analytical_dists):+.4f}")

            if n_sdf > 0:
                sdf_dists = world_dists[n_spheres * n_analytical:]
                print(f"  SDF obstacles: min = {np.min(sdf_dists):+.4f}")

        time.sleep(1.5)

    # Interactive mode: move arm and see collision feedback
    if not args.no_interactive:
        print("\n" + "=" * 60)
        print("Interactive mode: Moving arm to show collision feedback")
        print("Press [q] to quit")
        print("=" * 60)

        t = 0
        while viewer.is_active:
            # Oscillate shoulder and elbow
            shoulder_angle = 0.3 * np.sin(t * 0.5)
            elbow_angle = -0.5 - 0.5 * np.sin(t * 0.3)

            joint_list[0].joint_angle(shoulder_angle)
            joint_list[1].joint_angle(0.4)
            joint_list[3].joint_angle(elbow_angle)

            # Update collision check and colors
            min_dist = checker.compute_min_distance()
            checker.update_color()

            # Print status periodically
            if int(t * 10) % 10 == 0:
                status = "COLLISION" if min_dist < 0 else "Safe"
                print(f"\rmin_dist = {min_dist:+.4f} ({status})    ", end="", flush=True)

            viewer.redraw()
            time.sleep(0.05)
            t += 0.05

        print()

    viewer.close()
    print("\nDemo completed!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Example: Optimize wall-mounted robot base configuration.

This example demonstrates how to find the optimal base position and
protrusion configuration for a Panda robot mounted on a vertical wall,
such that it can reach 4 target poses while maintaining static equilibrium.
"""

import argparse
import time

import numpy as np

import skrobot
from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.model.primitives import Cylinder
from skrobot.model.primitives import Sphere
from skrobot.planner import FaceTarget
from skrobot.planner import optimize_wall_mount_base
from skrobot.planner import WallMountedRobotModel
from skrobot.viewers import PyrenderViewer


def create_arrow(start, direction, length=0.1, color=[1, 0, 0, 1]):
    """Create a simple arrow using a cylinder."""
    arrow = Cylinder(radius=0.005, height=length, face_colors=color)
    mid = start + direction * length / 2
    arrow.translate(mid)
    z_axis = np.array([0, 0, 1])
    if np.linalg.norm(np.cross(z_axis, direction)) > 1e-6:
        rot_axis = np.cross(z_axis, direction)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
        arrow.rotate(angle, rot_axis)
    return arrow


class OptimizationVisualizer:
    """Visualizer for wall mount optimization progress."""

    def __init__(self, viewer, wall, target_poses, sleep_time=0.1):
        self.viewer = viewer
        self.wall = wall
        self.target_poses = target_poses
        self.sleep_time = sleep_time

        # Markers to update
        self.suction_marker = None
        self.protrusion_markers = []
        self.iteration_text = None

    def update(self, iteration, base_pos, suction_radius, protrusion_positions, area, info):
        """Update visualization with current optimization state."""
        model = info['model']
        robot = info['robot']

        # Update model configuration
        model.set_base_position(base_pos)
        model.suction_radius = suction_radius
        model.protrusion_positions = protrusion_positions

        # Remove old markers
        if self.suction_marker is not None:
            self.viewer.delete(self.suction_marker)
        for marker in self.protrusion_markers:
            self.viewer.delete(marker)
        self.protrusion_markers = []

        # Add new suction marker
        suction_pos = model.get_suction_position_world()
        self.suction_marker = Cylinder(
            radius=suction_radius,
            height=0.02,
            face_colors=[0.2, 0.4, 1.0, 0.6]
        )
        self.suction_marker.translate(suction_pos)
        self.suction_marker.rotate(np.pi / 2, 'y')
        self.viewer.add(self.suction_marker)

        # Add protrusion markers
        prot_world = model.get_protrusion_positions_world()
        for i, pos in enumerate(prot_world):
            marker = Sphere(radius=0.02, color=[0.2, 0.8, 0.2, 0.8])
            marker.translate(pos)
            self.viewer.add(marker)
            self.protrusion_markers.append(marker)

        # Update viewer
        self.viewer.redraw()

        # Print progress
        print(f"\rIteration {iteration}: area={area*10000:.1f}cm², "
              f"suction_r={suction_radius*100:.1f}cm", end='', flush=True)

        time.sleep(self.sleep_time)


def main():
    parser = argparse.ArgumentParser(
        description='Wall-mounted robot base optimization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--visualize-optimization',
        action='store_true',
        help='Visualize optimization progress in real-time'
    )
    parser.add_argument(
        '--sleep-time', type=float, default=0.1,
        help='Sleep time between visualization updates (seconds)'
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Run in non-interactive mode'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Wall-Mounted Robot Base Optimization")
    print("=" * 60)

    # Load robot
    print("\nLoading Panda robot...")
    robot = skrobot.models.Panda()
    robot.reset_manip_pose()

    # Define wall
    print("\nDefining wall surface...")
    wall = FaceTarget(
        center=np.array([0.8, 0.0, 0.8]),
        normal=np.array([-1.0, 0.0, 0.0]),
        x_length=1.0,
        y_length=1.0,
    )
    print(f"  Wall center: {wall.center}")
    print(f"  Wall normal: {wall.normal}")

    # Define target poses (further from wall - wall is at x=0.8)
    print("\nDefining 4 target poses...")
    target_poses = [
        {'coords': Coordinates(pos=[0.1, 0.3, 0.6]),
         'translation_axis': True, 'rotation_axis': False},
        {'coords': Coordinates(pos=[0.1, -0.3, 0.6]),
         'translation_axis': True, 'rotation_axis': False},
        {'coords': Coordinates(pos=[0.2, 0.0, 1.0]),
         'translation_axis': True, 'rotation_axis': False},
        {'coords': Coordinates(pos=[0.2, 0.0, 0.3]),
         'translation_axis': True, 'rotation_axis': False},
    ]

    for i, t in enumerate(target_poses):
        print(f"  Target {i + 1}: pos={t['coords'].worldpos()}")

    # Physical parameters
    print("\nPhysical parameters:")
    vacuum_pressure = 40000  # 40 kPa
    friction_coeff = 0.6
    payload_mass = 0.5
    print(f"  Vacuum pressure: {vacuum_pressure / 1000} kPa")
    print(f"  Friction coefficient: {friction_coeff}")
    print(f"  Payload mass: {payload_mass} kg")

    # Setup visualization if requested
    viewer = None
    visualizer = None
    callback = None

    if args.visualize_optimization:
        print("\nSetting up visualization...")
        viewer = PyrenderViewer()

        # Add floor (at z=0)
        floor = Box(extents=[3.0, 3.0, 0.01], face_colors=[0.8, 0.8, 0.7, 0.5])
        floor.translate([0.5, 0, -0.005])
        viewer.add(floor)

        # Add origin axis on floor
        origin_axis = Axis(axis_radius=0.01, axis_length=0.3)
        viewer.add(origin_axis)

        # Add wall
        wall_box = Box(extents=[0.02, 2.0, 2.0], face_colors=[0.7, 0.7, 0.8, 0.3])
        wall_box.translate(wall.center)
        viewer.add(wall_box)

        # Add target markers
        for t in target_poses:
            pos = t['coords'].worldpos()
            sphere = Sphere(radius=0.03, color=[1.0, 0.3, 0.3, 0.8])
            sphere.translate(pos)
            viewer.add(sphere)

        # Add robot
        viewer.add(robot)
        viewer.show()

        # Create visualizer callback
        visualizer = OptimizationVisualizer(
            viewer, wall, target_poses, sleep_time=args.sleep_time
        )
        callback = visualizer.update

    # Run optimization
    print("\n" + "=" * 60)
    print("Running optimization...")
    print("=" * 60)

    start_time = time.time()

    result = optimize_wall_mount_base(
        robot=robot,
        wall=wall,
        target_poses=target_poses,
        vacuum_pressure=vacuum_pressure,
        friction_coeff=friction_coeff,
        payload_mass=payload_mass,
        initial_suction_radius=0.20,
        initial_protrusion_size=0.45,
        optimize_suction_radius=True,
        min_suction_radius=0.15,
        max_suction_radius=0.35,
        min_protrusion_area=0.06,
        max_iterations=200,
        verbose=not args.visualize_optimization,
        callback=callback
    )

    elapsed = time.time() - start_time

    if args.visualize_optimization:
        print()  # New line after progress output

    print(f"\nOptimization completed in {elapsed:.2f} seconds")

    # Display results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"  Success: {result['success']}")
    print(f"  Base position (local): {result['base_position']}")
    print(f"  Suction radius: {result['suction_radius'] * 100:.1f} cm")
    print(f"  Suction force: {result['suction_force']:.1f} N")
    print(f"  Base area: {result['base_area'] * 10000:.1f} cm²")
    print(f"  Max protrusion force: {result['max_protrusion_force']:.1f} N")
    print("\n  Protrusion positions (relative to suction center):")
    for i, pos in enumerate(result['protrusion_positions']):
        print(f"    P{i + 1}: ({pos[0] * 100:.1f}, {pos[1] * 100:.1f}) cm")

    # Final visualization
    if viewer is None:
        print("\n" + "=" * 60)
        print("Final Visualization")
        print("=" * 60)
        viewer = PyrenderViewer()

        # Add floor (at z=0)
        floor = Box(extents=[3.0, 3.0, 0.01], face_colors=[0.8, 0.8, 0.7, 0.5])
        floor.translate([0.5, 0, -0.005])
        viewer.add(floor)

        # Add origin axis on floor
        origin_axis = Axis(axis_radius=0.01, axis_length=0.3)
        viewer.add(origin_axis)

        # Add wall
        wall_box = Box(extents=[0.02, 2.0, 2.0], face_colors=[0.7, 0.7, 0.8, 0.3])
        wall_box.translate(wall.center)
        viewer.add(wall_box)

        # Add target markers
        for t in target_poses:
            pos = t['coords'].worldpos()
            sphere = Sphere(radius=0.03, color=[1.0, 0.3, 0.3, 0.8])
            sphere.translate(pos)
            viewer.add(sphere)

        viewer.add(robot)
        viewer.show()

    # Update to final configuration
    model = WallMountedRobotModel(robot, wall, end_coords=robot.rarm_end_coords)
    model.set_base_position(result['base_position'])
    model.suction_radius = result['suction_radius']
    model.protrusion_positions = result['protrusion_positions']

    # Clear old markers if visualizing
    if visualizer is not None:
        if visualizer.suction_marker is not None:
            viewer.delete(visualizer.suction_marker)
        for marker in visualizer.protrusion_markers:
            viewer.delete(marker)

    # Add final markers
    suction_pos = model.get_suction_position_world()
    suction_marker = Cylinder(
        radius=result['suction_radius'],
        height=0.02,
        face_colors=[0.2, 0.4, 1.0, 0.8]
    )
    suction_marker.translate(suction_pos)
    suction_marker.rotate(np.pi / 2, 'y')
    viewer.add(suction_marker)

    prot_world = model.get_protrusion_positions_world()
    for pos in prot_world:
        marker = Sphere(radius=0.015, color=[0.2, 0.8, 0.2, 1.0])
        marker.translate(pos)
        viewer.add(marker)

    base_axis = Axis(axis_radius=0.005, axis_length=0.1)
    base_axis.translate(suction_pos)
    viewer.add(base_axis)

    viewer.redraw()

    # Animate through target poses
    if not args.no_interactive:
        print("\nAnimating through target poses...")
        print("Press 'q' to quit")

        link_list = [
            robot.panda_link1, robot.panda_link2, robot.panda_link3,
            robot.panda_link4, robot.panda_link5, robot.panda_link6,
            robot.panda_link7,
        ]
        end_coords = robot.rarm_end_coords

        pose_idx = 0
        while viewer.is_active:
            target = target_poses[pose_idx]
            coords = target['coords']
            trans_axis = target.get('translation_axis', True)
            rot_axis = target.get('rotation_axis', True)

            print(f"\n  Moving to target {pose_idx + 1}...")

            robot.inverse_kinematics(
                coords,
                move_target=end_coords,
                link_list=link_list,
                rotation_axis=rot_axis,
                translation_axis=trans_axis,
                stop=50
            )

            viewer.redraw()
            time.sleep(0.5)

            pose_idx = (pose_idx + 1) % 4

    print("\nDone!")


if __name__ == '__main__':
    main()

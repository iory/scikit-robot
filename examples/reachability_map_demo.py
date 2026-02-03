#!/usr/bin/env python
"""Demonstration of fast reachability map computation.

This example shows how to compute and visualize a robot's workspace
reachability map using batch forward kinematics (JAX or NumPy backend).

Usage:
    python examples/reachability_map_demo.py
    python examples/reachability_map_demo.py --robot panda --samples 500000
    python examples/reachability_map_demo.py --color manipulability
    python examples/reachability_map_demo.py --urdf /path/to/robot.urdf
"""

import argparse
import time

import numpy as np

import skrobot
from skrobot.coordinates import CascadedCoords
from skrobot.kinematics import ReachabilityMap
from skrobot.model.robot_model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF
from skrobot.urdf.robot_class_generator import generate_groups_from_geometry


def _select_from_list(prompt, options, allow_multiple=False):
    """Prompt user to select from a list of options.

    Parameters
    ----------
    prompt : str
        Prompt message.
    options : list
        List of options to choose from.
    allow_multiple : bool
        If True, allow multiple selections.

    Returns
    -------
    str or list
        Selected option(s).
    """
    print(f"\n{prompt}")
    for i, option in enumerate(options):
        print(f"  [{i}] {option}")

    while True:
        try:
            if allow_multiple:
                selection = input("Enter number(s) separated by comma: ").strip()
                indices = [int(x.strip()) for x in selection.split(',')]
                if all(0 <= idx < len(options) for idx in indices):
                    return [options[idx] for idx in indices]
            else:
                selection = int(input("Enter number: ").strip())
                if 0 <= selection < len(options):
                    return options[selection]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def _setup_robot_from_urdf(urdf_path, interactive=True):
    """Load robot from URDF and detect/select kinematic chain.

    Parameters
    ----------
    urdf_path : str
        Path to URDF file.
    interactive : bool
        If True, prompt user for selection when auto-detection fails.

    Returns
    -------
    tuple
        (robot, link_list, end_coords)
    """
    # Load robot from URDF
    print(f"Loading robot from URDF: {urdf_path}")
    robot = RobotModelFromURDF(urdf_file=urdf_path)

    # Try automatic detection using robot_class_generator
    print("Attempting automatic kinematic chain detection...")
    groups, end_effectors, end_coords_info, robot_name = generate_groups_from_geometry(robot)

    # Find the best arm group
    arm_group = None
    arm_group_name = None

    # Priority order for arm groups
    priority_order = ['right_arm', 'left_arm', 'arm']
    for group_name in priority_order:
        if group_name in groups and groups[group_name] is not None:
            arm_group = groups[group_name]
            arm_group_name = group_name
            break

    if arm_group is not None:
        # Setup end_coords first to get parent link
        ec_info = end_coords_info.get(arm_group_name)
        if ec_info is not None:
            parent_link_name = ec_info['parent_link']
            pos = ec_info.get('pos', [0.0, 0.0, 0.0])
            rot = ec_info.get('rot')

            # Convert to float to avoid np.float64 display issues
            pos = [float(v) for v in pos]
            if rot is not None:
                rot = [float(v) for v in rot]

            parent_link = None
            for link in robot.link_list:
                if link.name == parent_link_name:
                    parent_link = link
                    break

            if parent_link is not None:
                # Use robot.link_lists() to get correct kinematic chain
                all_links = robot.link_lists(parent_link, robot.root_link)
                link_list = RobotModel.filter_movable_links(all_links)

                if link_list:
                    print(f"  Detected {arm_group_name}: {len(link_list)} joints")
                    print(f"  Links: {[l.name for l in link_list]}")

                    has_offset = any(abs(v) > 1e-6 for v in pos)
                    has_rot = rot is not None and any(abs(v) > 1e-6 for v in rot)

                    if has_offset or has_rot:
                        end_coords = CascadedCoords(
                            parent=parent_link,
                            pos=pos if has_offset else None,
                            rot=rot if has_rot else None,
                            name=f'{arm_group_name}_end_coords'
                        )
                    else:
                        end_coords = CascadedCoords(
                            parent=parent_link,
                            name=f'{arm_group_name}_end_coords'
                        )

                    print(f"  End coords parent: {parent_link_name}")
                    if has_offset:
                        print(f"  End coords offset: [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
                    if has_rot:
                        print(f"  End coords rotation: [{rot[0]:.6f}, {rot[1]:.6f}, {rot[2]:.6f}]")

                    # Set end_coords on robot for IK support in viewer
                    robot.end_coords = end_coords

                    return robot, link_list, end_coords

    # Auto-detection failed, need user input
    if not interactive:
        raise RuntimeError(
            "Could not automatically detect kinematic chain. "
            "Run in interactive mode to manually select links."
        )

    print("\nAutomatic detection failed. Manual selection required.")

    # Get root and leaf links for user selection
    root_link_name = robot.root_link.name
    leaf_link_names = [link.name for link in robot.leaf_links]

    print(f"\nRoot link: {root_link_name}")
    print(f"Leaf links ({len(leaf_link_names)}): {leaf_link_names[:10]}...")

    # Let user select end link (end-effector)
    end_link_name = _select_from_list(
        "Select END link for kinematic chain (end-effector):",
        leaf_link_names[:20]
    )

    # Find end link object
    end_link_obj = getattr(robot, end_link_name.replace('-', '_'), None)
    if end_link_obj is None:
        raise RuntimeError(f"Could not find end link: {end_link_name}")

    # Use robot.link_lists() to get kinematic chain from root to end
    all_links = robot.link_lists(end_link_obj, robot.root_link)
    link_list = RobotModel.filter_movable_links(all_links)

    if not link_list:
        raise RuntimeError(
            f"Could not find movable joints from {root_link_name} to {end_link_name}"
        )

    print(f"\nKinematic chain: {[l.name for l in link_list]}")

    end_coords = CascadedCoords(
        parent=end_link_obj,
        name='end_coords'
    )

    # Set end_coords on robot for IK support in viewer
    robot.end_coords = end_coords

    return robot, link_list, end_coords


def main():
    parser = argparse.ArgumentParser(
        description="Compute and visualize robot reachability map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--robot', type=str, default='pr2',
        choices=['pr2', 'panda', 'fetch', 'r8', 'nextage', 'tycoon'],
        help='Robot model to use (ignored if --urdf is specified)'
    )
    parser.add_argument(
        '--urdf', type=str, default=None,
        help='Path to custom URDF file. When specified, --robot is ignored.'
    )
    parser.add_argument(
        '--backend', type=str, default=None,
        choices=['jax', 'numpy'],
        help='Backend to use for computation (default: auto-select best available)'
    )
    parser.add_argument(
        '--samples', type=int, default=2000000,
        help='Number of samples for random sampling'
    )
    parser.add_argument(
        '--sampling', type=str, default='random',
        choices=['random', 'grid'],
        help='Sampling method: random or grid'
    )
    parser.add_argument(
        '--bins', type=int, default=10,
        help='Bins per joint for grid sampling (total = bins^n_joints)'
    )
    parser.add_argument(
        '--voxel-size', type=float, default=0.05,
        help='Voxel size in meters'
    )
    parser.add_argument(
        '--color', type=str, default='reachability_index',
        choices=['reachability_index', 'reachability', 'manipulability'],
        help='Coloring mode: reachability_index (orientation diversity), '
             'reachability (raw count), manipulability'
    )
    parser.add_argument(
        '--orientation-bins', type=int, default=50,
        help='Number of orientation bins for Reachability Index calculation. '
             'Set to 0 for position-only mode.'
    )
    parser.add_argument(
        '--save', type=str, default=None,
        help='Save reachability map to file (.npz)'
    )
    parser.add_argument(
        '--load', type=str, default=None,
        help='Load reachability map from file (.npz)'
    )
    parser.add_argument(
        '--no-viz', action='store_true',
        help='Skip visualization'
    )
    parser.add_argument(
        '--viz-points', type=int, default=30000,
        help='Number of points to visualize (more = denser but slower)'
    )
    parser.add_argument(
        '--z-slice', type=float, nargs=2, default=None,
        metavar=('Z_MIN', 'Z_MAX'),
        help='Show only points within Z range (e.g., --z-slice 0.5 1.0)'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.3,
        help='Sphere opacity (0.0=transparent, 1.0=opaque)'
    )
    parser.add_argument(
        '--no-interactive', action='store_true',
        help='Run in non-interactive mode (exit after computation)'
    )
    args = parser.parse_args()

    # Use minimal settings for non-interactive mode (CI testing)
    if args.no_interactive:
        if args.samples == 2000000:
            args.samples = 1000
        args.no_viz = True

    # Setup robot
    if args.urdf is not None:
        # Load from custom URDF
        robot, link_list, end_coords = _setup_robot_from_urdf(
            args.urdf,
            interactive=not args.no_interactive
        )
    else:
        # Use built-in robot model
        print(f"Loading {args.robot} robot...")
        if args.robot == 'pr2':
            robot = skrobot.models.PR2()
            robot.init_pose()
        elif args.robot == 'panda':
            robot = skrobot.models.Panda()
        elif args.robot == 'fetch':
            robot = skrobot.models.Fetch()
        elif args.robot == 'r8':
            robot = skrobot.models.R8_6()
        elif args.robot == 'nextage':
            robot = skrobot.models.Nextage()
        elif args.robot == 'tycoon':
            robot = skrobot.models.RoverArmedTycoon()

        # tycoon uses 'arm' instead of 'rarm'
        if args.robot == 'tycoon':
            link_list = robot.arm.link_list
            end_coords = robot.arm.end_coords
        else:
            link_list = robot.rarm.link_list
            end_coords = robot.rarm.end_coords

    # Create reachability map
    rmap = ReachabilityMap(
        robot, link_list, end_coords,
        voxel_size=args.voxel_size,
        backend=args.backend,
    )

    # Load or compute
    if args.load:
        print(f"Loading reachability map from {args.load}...")
        rmap.load(args.load)
        print(f"  Loaded {rmap.n_reachable_voxels:,} reachable voxels")
    else:
        print()
        rmap.compute(
            n_samples=args.samples,
            sampling=args.sampling,
            bins_per_joint=args.bins,
            orientation_bins=args.orientation_bins,
            verbose=True
        )

    # Save if requested
    if args.save:
        print(f"\nSaving reachability map to {args.save}...")
        rmap.save(args.save)

    # Print summary
    print()
    print("=" * 50)
    print("Reachability Map Summary")
    print("=" * 50)
    robot_name = args.urdf if args.urdf else args.robot
    print(f"Robot: {robot_name}")
    print(f"Voxel size: {args.voxel_size * 100:.0f} cm")
    print(f"Reachable voxels: {rmap.n_reachable_voxels:,}")
    print(f"Reachable volume: {rmap.reachable_volume:.3f} m³")
    print()
    print("Workspace bounds:")
    print(f"  X: [{rmap.bounds['x'][0]:.3f}, {rmap.bounds['x'][1]:.3f}] m")
    print(f"  Y: [{rmap.bounds['y'][0]:.3f}, {rmap.bounds['y'][1]:.3f}] m")
    print(f"  Z: [{rmap.bounds['z'][0]:.3f}, {rmap.bounds['z'][1]:.3f}] m")

    # Visualization
    if not args.no_viz:
        print()
        print("Visualizing reachability map...")
        print("  (colored by", args.color + ")")

        # Get point cloud
        positions, colors = rmap.get_point_cloud(
            color_by=args.color,
            max_points=args.viz_points
        )

        # Apply Z-slice filter if specified
        if args.z_slice is not None:
            z_min, z_max = args.z_slice
            mask = (positions[:, 2] >= z_min) & (positions[:, 2] <= z_max)
            positions = positions[mask]
            colors = colors[mask]
            print(f"  Z-slice [{z_min:.2f}, {z_max:.2f}]: {len(positions)} points")

        # Create viewer
        viewer = skrobot.viewers.ViserViewer(
            enable_ik=True,
            enable_motion_planning=True)

        # Add robot
        viewer.add(robot)

        # Add reachability visualization using batched meshes
        import trimesh as tm
        server = viewer._server

        # Create unit icosphere mesh
        unit_sphere = tm.creation.icosphere(subdivisions=1, radius=1.0)
        sphere_vertices = unit_sphere.vertices.astype(np.float32)
        sphere_faces = unit_sphere.faces.astype(np.uint32)

        # Prepare batched data
        n_points = len(positions)
        print(f"  Adding {n_points} visualization points...")
        sphere_radius = args.voxel_size * 0.4
        batched_positions = positions.astype(np.float32)
        batched_wxyzs = np.tile([1.0, 0.0, 0.0, 0.0], (n_points, 1)).astype(np.float32)
        batched_scales = np.full((n_points, 3), sphere_radius, dtype=np.float32)
        batched_colors = (colors[:, :3] * 255).astype(np.uint8)

        # Add batched spheres with opacity
        reachability_mesh_handle = server.scene.add_batched_meshes_simple(
            "reachability_spheres",
            vertices=sphere_vertices,
            faces=sphere_faces,
            batched_wxyzs=batched_wxyzs,
            batched_positions=batched_positions,
            batched_scales=batched_scales,
            batched_colors=batched_colors,
            opacity=args.alpha,
        )

        # Show
        viewer.show()
        print()

        print("Viser viewer opened in browser. Press Ctrl+C to exit.")
        print("  Use the GUI sliders to adjust Z-slice range and opacity.")

        # Get Z bounds
        z_bounds = rmap.bounds['z']

        # Store full point cloud data for filtering
        full_positions, full_colors = rmap.get_point_cloud(
            color_by=args.color,
            max_points=args.viz_points * 2  # Get more points for filtering
        )
        full_colors_uint8 = (full_colors[:, :3] * 255).astype(np.uint8)

        # Add GUI folder for controls
        with server.gui.add_folder("Reachability Slice"):
            z_min_slider = server.gui.add_slider(
                "Z Min", min=z_bounds[0], max=z_bounds[1],
                initial_value=z_bounds[0], step=0.01
            )
            z_max_slider = server.gui.add_slider(
                "Z Max", min=z_bounds[0], max=z_bounds[1],
                initial_value=z_bounds[1], step=0.01
            )
            sphere_size_initial = max(0.005, args.voxel_size * 0.4)
            sphere_size_slider = server.gui.add_slider(
                "Sphere Size", min=0.005, max=0.1,
                initial_value=sphere_size_initial, step=0.005
            )
            alpha_slider = server.gui.add_slider(
                "Opacity", min=0.05, max=1.0,
                initial_value=args.alpha, step=0.05
            )

        mesh_handle = reachability_mesh_handle

        def update_mesh():
            nonlocal mesh_handle
            z_min = z_min_slider.value
            z_max = z_max_slider.value
            mask = (full_positions[:, 2] >= z_min) & (full_positions[:, 2] <= z_max)
            filtered_positions = full_positions[mask].astype(np.float32)
            filtered_colors = full_colors_uint8[mask]

            if mesh_handle is not None:
                mesh_handle.remove()

            if len(filtered_positions) > 0:
                n_filtered = len(filtered_positions)
                sphere_size = sphere_size_slider.value
                mesh_handle = server.scene.add_batched_meshes_simple(
                    "reachability_spheres",
                    vertices=sphere_vertices,
                    faces=sphere_faces,
                    batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (n_filtered, 1)).astype(np.float32),
                    batched_positions=filtered_positions,
                    batched_scales=np.full((n_filtered, 3), sphere_size, dtype=np.float32),
                    batched_colors=filtered_colors,
                    opacity=alpha_slider.value,
                )

        @z_min_slider.on_update
        def _(_):
            update_mesh()

        @z_max_slider.on_update
        def _(_):
            update_mesh()

        @sphere_size_slider.on_update
        def _(_):
            update_mesh()

        @alpha_slider.on_update
        def _(_):
            update_mesh()

        if not args.no_interactive:
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()

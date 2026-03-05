#!/usr/bin/env python
"""Skeleton visualization example.

Examples
--------
# Visualize a built-in robot model
$ python examples/skeleton_visualization.py --robot Fetch

# Visualize from URDF file
$ python examples/skeleton_visualization.py --urdf path/to/robot.urdf

# Save image instead of showing viewer
$ python examples/skeleton_visualization.py --robot PR2 --save output.png
"""

import argparse

import numpy as np

from skrobot.model import RobotModel
from skrobot.model.skeleton import SkeletonModel


def get_robot(robot_name):
    """Get robot model by name.

    Parameters
    ----------
    robot_name : str
        Name of the robot model (e.g., 'Fetch', 'Panda', 'PR2', 'Nextage').

    Returns
    -------
    RobotModel
        Robot model instance.
    """
    import skrobot.models as models

    robot_class = getattr(models, robot_name, None)
    if robot_class is None:
        available = [name for name in dir(models)
                     if not name.startswith('_') and name[0].isupper()]
        raise ValueError(
            f"Unknown robot: {robot_name}. Available: {', '.join(available)}")
    return robot_class()


def render_to_image(robot, skeleton, resolution=(640, 480)):
    """Render robot with skeleton to image using offscreen renderer.

    Parameters
    ----------
    robot : RobotModel
        Robot model to render.
    skeleton : SkeletonModel
        Skeleton model to overlay.
    resolution : tuple
        Image resolution (width, height).

    Returns
    -------
    np.ndarray
        RGB image array.
    """
    import inspect

    import pyrender

    # Check if always_on_top is supported
    from_trimesh_args = inspect.signature(pyrender.Mesh.from_trimesh).parameters
    supports_always_on_top = 'always_on_top' in from_trimesh_args

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])

    # Add robot meshes
    for link in robot.link_list:
        mesh = link.concatenated_visual_mesh
        if mesh is not None and len(mesh.vertices) > 0:
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            transform = link.worldcoords().T()
            scene.add(pyrender_mesh, pose=transform)

    # Add skeleton meshes
    for link in skeleton.link_list:
        mesh = link.concatenated_visual_mesh
        if mesh is not None and len(mesh.vertices) > 0:
            if supports_always_on_top:
                pyrender_mesh = pyrender.Mesh.from_trimesh(
                    mesh, smooth=False, always_on_top=True)
            else:
                pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            transform = link.worldcoords().T()
            scene.add(pyrender_mesh, pose=transform)

    # Calculate camera position based on scene bounds
    bounds = scene.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = np.linalg.norm(bounds[1] - bounds[0])
    distance = size * 1.2

    # Camera looking at center from front-right-above
    angle_h = np.pi / 4
    angle_v = np.pi / 6

    cam_x = center[0] + distance * np.cos(angle_v) * np.sin(angle_h)
    cam_y = center[1] + distance * np.cos(angle_v) * np.cos(angle_h)
    cam_z = center[2] + distance * np.sin(angle_v)

    forward = center - np.array([cam_x, cam_y, cam_z])
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = -forward
    camera_pose[:3, 3] = [cam_x, cam_y, cam_z]

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_pose)

    # Add lights
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
    color, _ = renderer.render(scene)
    renderer.delete()

    return color


def main():
    parser = argparse.ArgumentParser(
        description='Visualize robot skeleton',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--robot', '-r', type=str, default='Fetch',
                       help='Built-in robot model name (default: Fetch)')
    group.add_argument('--urdf', '-u', type=str,
                       help='Path to URDF file')
    parser.add_argument('--save', '-s', type=str, default=None,
                        help='Save image to file instead of showing viewer')
    parser.add_argument('--resolution', type=int, nargs=2, default=[640, 480],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Image resolution (default: 640 480)')
    parser.add_argument('--no-skeleton', action='store_true',
                        help='Show only skeleton without robot mesh')
    parser.add_argument('--no-interactive', action='store_true',
                        help='Run in non-interactive mode (exit immediately)')
    args = parser.parse_args()

    # Load robot
    if args.urdf:
        print(f"Loading URDF: {args.urdf}")
        robot = RobotModel.from_urdf(args.urdf)
    else:
        print(f"Loading robot: {args.robot}")
        robot = get_robot(args.robot)

    # Create skeleton
    skeleton = SkeletonModel(robot)
    print(f"Created skeleton with {len(skeleton.link_list)} links")

    if args.save or args.no_interactive:
        # Save to image file (use temp file if --no-interactive without --save)
        import tempfile

        from PIL import Image

        if args.no_skeleton:
            # Render only skeleton
            robot = skeleton
            skeleton = SkeletonModel.__new__(SkeletonModel)
            skeleton.link_list = []

        img = render_to_image(robot, skeleton, tuple(args.resolution))
        if args.save:
            Image.fromarray(img).save(args.save)
            print(f"Saved to {args.save}")
        else:
            # Save to temp file for --no-interactive mode
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                Image.fromarray(img).save(f.name)
                print(f"Saved to {f.name}")
    else:
        # Show in viewer
        import time

        from skrobot.viewers import PyrenderViewer

        viewer = PyrenderViewer()
        if not args.no_skeleton:
            viewer.add(robot)
        viewer.add(skeleton, always_on_top=True)
        viewer.show()

        print("Press 'q' to quit")
        while viewer.is_active:
            time.sleep(0.1)
            viewer.redraw()
        viewer.close()


if __name__ == "__main__":
    main()

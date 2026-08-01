#!/usr/bin/env python
"""Self-collision-free trajectory optimization with per-link GridSDFs.

``TrajectoryProblem.add_self_collision_cost`` approximates every link with a
few spheres by default. On a robot with bulky links -- the Panda housings are
roughly 0.11 x 0.19 x 0.25 m -- those spheres are so much fatter than the
geometry that even the rest pose reads as self-colliding, which is why
self-collision is usually left switched off. ``mode='gridsdf'`` instead gives
each collision link its own GridSDF and looks the other links' sampled surface
points up in it, which tracks the exact mesh distance closely enough to be
used as a cost.

This example prints both models' verdict on the rest pose, then plans between
two configurations that are each self-collision-free but whose straight-line
interpolation swings the hand into the base link, and animates the result.

Dependencies: ``jax`` and ``jaxls`` (jaxls is not on PyPI). Install with::

    pip install jax
    pip install "git+https://github.com/brentyi/jaxls.git"

Usage::

    python examples/self_collision_free_trajectory.py
    python examples/self_collision_free_trajectory.py --viewer mitsuba \\
        --save-video self_collision_free_trajectory.mp4
    python examples/self_collision_free_trajectory.py --no-interactive
"""

import argparse
import importlib.util
import sys
import time

import numpy as np


_INSTALL_HINT = (
    'This example requires the JAX and JAXls packages.\n'
    'JAXls is not on PyPI, so the usual ``pip install jaxls`` will not '
    'work.\n'
    'Install both with:\n'
    '    pip install jax\n'
    '    pip install "git+https://github.com/brentyi/jaxls.git"'
)


def _require_jax_and_jaxls():
    """Exit cleanly with install instructions if jax / jaxls is missing."""
    missing = [name for name in ('jax', 'jaxls')
               if importlib.util.find_spec(name) is None]
    if missing:
        print('Missing dependency: {}'.format(', '.join(missing)),
              file=sys.stderr)
        print(_INSTALL_HINT, file=sys.stderr)
        sys.exit(0)


_require_jax_and_jaxls()

import skrobot  # noqa: E402
from skrobot.planner.trajectory_optimization import TrajectoryProblem  # noqa: E402
from skrobot.planner.trajectory_optimization.collision import create_self_collision_pairs  # noqa: E402
from skrobot.planner.trajectory_optimization.fk_utils import compute_self_collision_distances  # noqa: E402
from skrobot.planner.trajectory_optimization.gridsdf_collision import gridsdf_self_distances  # noqa: E402
from skrobot.planner.trajectory_optimization.solvers import create_solver  # noqa: E402
from skrobot.planner.trajectory_optimization.trajectory import interpolate_trajectory  # noqa: E402
from skrobot.utils.video import record_viewer  # noqa: E402
from skrobot.viewers import VIEWER_HELP  # noqa: E402
from skrobot.viewers import VIEWER_TYPES  # noqa: E402


# Goal configuration. It is self-collision-free, as is the rest pose the
# motion starts from, but interpolating straight between the two swings the
# hand into the base link -- the case the optimizer has to fix.
GOAL_ANGLES = np.array(
    [1.509, 1.586, -0.472, -2.569, 0.715, 1.894, -1.617, 0.0, 0.0])


def make_panda():
    """Return a Panda plus its joint links and self-collision links.

    The fingers are excluded from the collision link list. They are physically
    attached to the hand but sit far apart in the list, and
    ``create_self_collision_pairs`` treats "adjacent" purely as list
    adjacency, so the hand/finger contact would be reported forever.
    """
    robot = skrobot.models.Panda()
    robot.reset_manip_pose()
    link_list = [link for link in robot.link_list
                 if link.joint is not None
                 and link.joint.__class__.__name__ != 'FixedJoint']
    collision_link_list = [link for link in robot.link_list
                           if link.collision_mesh is not None
                           and 'finger' not in link.name]
    return robot, link_list, collision_link_list


def sphere_min_distance(problem, collision_link_list):
    """Minimum self-collision distance of the sphere approximation."""
    spheres = problem.collision_spheres
    pairs_i, pairs_j = problem.self_collision_pairs
    positions = np.stack([
        collision_link_list[link_index].worldcoords().transform_vector(center)
        for link_index, center in zip(spheres['link_indices'],
                                      spheres['sphere_centers_local'])])
    return compute_self_collision_distances(
        positions, spheres['sphere_radii'], pairs_i, pairs_j, np).min()


def gridsdf_min_distance(gridsdf_data, collision_link_list):
    """Minimum self-collision distance of the GridSDF representation."""
    positions = np.stack(
        [link.worldpos() for link in collision_link_list])
    rotations = np.stack(
        [link.worldrot() for link in collision_link_list])
    return gridsdf_self_distances(
        positions, rotations, gridsdf_data, np).min()


def penetrating_links(gridsdf_data, collision_link_list):
    """Indices of the links currently penetrating another link."""
    positions = np.stack(
        [link.worldpos() for link in collision_link_list])
    rotations = np.stack(
        [link.worldrot() for link in collision_link_list])
    per_pair = gridsdf_self_distances(
        positions, rotations, gridsdf_data, np).min(axis=1)
    hit = per_pair < 0.0
    return set(gridsdf_data['pairs_a'][hit].tolist()
               + gridsdf_data['pairs_b'][hit].tolist())


def exact_min_distance(collision_link_list):
    """Exact mesh distance via trimesh/fcl, or None when fcl is missing."""
    try:
        import trimesh.collision
    except ImportError:
        return None
    meshes = []
    for link in collision_link_list:
        mesh = link.collision_mesh.copy()
        mesh.apply_transform(link.worldcoords().T())
        meshes.append(mesh)
    try:
        distances = []
        for i, j in create_self_collision_pairs(collision_link_list):
            manager = trimesh.collision.CollisionManager()
            manager.add_object('link', meshes[i])
            distances.append(manager.min_distance_single(meshes[j]))
    except ValueError:
        # trimesh raises when python-fcl is not installed.
        return None
    return min(distances)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-n', type=int, default=10,
        help='Number of waypoints.')
    parser.add_argument(
        '--dim-grid', type=int, default=24,
        help='GridSDF resolution per axis.')
    parser.add_argument(
        '--n-surface', type=int, default=24,
        help='Surface sample points per link.')
    parser.add_argument(
        '--iterations', type=int, default=50,
        help='Maximum solver iterations.')
    parser.add_argument(
        '--viewer', type=str,
        choices=VIEWER_TYPES, default='pyrender',
        help=VIEWER_HELP)
    parser.add_argument(
        '--save-video', type=str, default=None,
        help='Record the animation to this video file (e.g. out.mp4). Works '
             'with any --viewer; use --viewer mitsuba to record headlessly.')
    parser.add_argument(
        '--play-initial', action='store_true',
        help='Play the straight-line initial guess before the optimized '
             'trajectory, with penetrating links tinted red.')
    parser.add_argument(
        '--no-interactive', action='store_true',
        help='Run in non-interactive mode (do not wait for user input).')
    args = parser.parse_args()

    robot, link_list, collision_link_list = make_panda()
    print('collision links: {}'.format(
        ', '.join(link.name for link in collision_link_list)))

    problem = TrajectoryProblem(robot, link_list, n_waypoints=args.n, dt=0.1)
    problem.add_collision_cost(collision_link_list, world_obstacles=[])
    problem.add_smoothness_cost(weight=0.01)

    # Register the sphere model first so both verdicts can be printed, then
    # drop that residual again and keep only the GridSDF one, which is what
    # actually drives the solve.
    problem.add_self_collision_cost(mode='sphere')
    problem.residuals.pop()

    start = time.time()
    problem.add_self_collision_cost(
        mode='gridsdf', dim_grid=args.dim_grid, n_surface=args.n_surface,
        weight=1000.0, activation_distance=0.02, as_constraint=False)
    gridsdf_data = problem.residuals[-1].params['gridsdf_data']
    print('GridSDF build: {:.2f} s for {} links, {} ordered pairs '
          '(voxelization is cached on disk after the first run)'.format(
              time.time() - start, len(collision_link_list),
              len(gridsdf_data['pairs_a'])))

    robot.reset_manip_pose()
    exact = exact_min_distance(collision_link_list)
    print('\nMinimum self-collision distance at the rest pose [m]:')
    print('  sphere  : {:+.5f}{}'.format(
        sphere_min_distance(problem, collision_link_list),
        '  <- reports a collision that is not there'))
    print('  gridsdf : {:+.5f}'.format(
        gridsdf_min_distance(gridsdf_data, collision_link_list)))
    if exact is None:
        print('  exact   : (install python-fcl for the mesh reference)')
    else:
        print('  exact   : {:+.5f}  (trimesh/fcl reference)'.format(exact))

    start_angles = np.array(
        [link.joint.joint_angle() for link in link_list])
    initial_traj = interpolate_trajectory(start_angles, GOAL_ANGLES, args.n)

    def trajectory_min_distance(trajectory):
        worst = np.inf
        for angles in trajectory:
            for link, angle in zip(link_list, angles):
                link.joint.joint_angle(float(angle))
            worst = min(
                worst, gridsdf_min_distance(gridsdf_data,
                                            collision_link_list))
        return worst

    before = trajectory_min_distance(initial_traj)

    solver = create_solver('jaxls', max_iterations=args.iterations)
    start = time.time()
    result = solver.solve(problem, initial_traj)
    print('\nsolve: success={} in {:.2f} s'.format(
        result.success, time.time() - start))

    after = trajectory_min_distance(result.trajectory)
    print('worst self-collision distance over the trajectory [m]:')
    print('  straight-line initial guess : {:+.5f}'.format(before))
    print('  optimized                   : {:+.5f}'.format(after))

    viewer = skrobot.viewers.create_viewer(args.viewer, resolution=(640, 480))
    viewer.add(robot)
    viewer.show()
    viewer.set_camera(angles=[np.deg2rad(72), 0, np.deg2rad(60)],
                      distance=1.4, center=[0, 0, 0.45])
    recorder = record_viewer(viewer, args.save_video, fps=10)

    def play(trajectory, highlight):
        for angles in trajectory:
            for link, angle in zip(link_list, angles):
                link.joint.joint_angle(float(angle))
            if highlight:
                hit = penetrating_links(gridsdf_data, collision_link_list)
                for index, link in enumerate(collision_link_list):
                    if index in hit:
                        link.set_color([220, 40, 40, 255])
                    else:
                        link.reset_color()
            # viewer.pause keeps the camera draggable during the pause; a
            # bare time.sleep would freeze the window on macOS (main-thread
            # GL loop).
            viewer.pause(0.2)

    if args.play_initial:
        play(initial_traj, highlight=True)
        for link in collision_link_list:
            link.reset_color()
    play(result.trajectory, highlight=False)

    if recorder is not None:
        print('saving video to {}'.format(recorder.save()))

    if args.no_interactive:
        viewer.close()
    else:
        viewer.wait_until_close()


if __name__ == '__main__':
    main()

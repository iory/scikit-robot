#!/usr/bin/env python

import argparse
import time

import numpy as np

import skrobot
from skrobot.collision import RobotCollisionChecker
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.model.primitives import LineString
from skrobot.planner import sqp_plan_trajectory
from skrobot.planner import SweptSphereSdfCollisionChecker
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config
from skrobot.utils.visualization import trajectory_visualization


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '-n', type=int, default=10,
    help='number of waypoints.')
parser.add_argument(
    '--without-base',
    action='store_true',
    help='Solve motion planning without base.'
)
parser.add_argument(
    '--viewer', type=str,
    choices=['trimesh', 'pyrender'], default='trimesh',
    help='Choose the viewer type: trimesh or pyrender')
parser.add_argument(
    '--no-interactive',
    action='store_true',
    help="Run in non-interactive mode (do not wait for user input)"
)
parser.add_argument(
    '--trajectory-visualization',
    action='store_true',
    help="Enable trajectory optimization visualization"
)
parser.add_argument(
    '--solver', type=str,
    choices=['sqp', 'jaxls', 'gradient_descent', 'scipy'], default='sqp',
    help='Trajectory optimization solver: sqp (default), '
         'jaxls (JAX nonlinear least squares), gradient_descent (JAX autodiff), '
         'scipy (SciPy SLSQP)'
)
parser.add_argument(
    '--iterations', type=int, default=300,
    help='Number of iterations for optimizer'
)
args = parser.parse_args()

# initialization stuff
np.random.seed(0)
robot_model = skrobot.models.PR2()
robot_model.init_pose()

# create obstacle's visual element with the corresponding sdf.
# The sdf is stored as a member variable of box.
box_center = np.array([0.9, -0.2, 0.9])
box = Box(extents=[0.7, 0.5, 0.6], with_sdf=True)
box.translate(box_center)

link_list = [
    robot_model.r_shoulder_pan_link, robot_model.r_shoulder_lift_link,
    robot_model.r_upper_arm_roll_link, robot_model.r_elbow_flex_link,
    robot_model.r_forearm_roll_link, robot_model.r_wrist_flex_link,
    robot_model.r_wrist_roll_link]
joint_list = [link.joint for link in link_list]

# Collision links for right arm (being controlled)
coll_link_list = [
    robot_model.r_upper_arm_link, robot_model.r_forearm_link,
    robot_model.r_gripper_palm_link, robot_model.r_gripper_r_finger_link,
    robot_model.r_gripper_l_finger_link,
]


# obtain av_start (please try both with_base=True, False)
with_base = not args.without_base
av_start = np.array([0.564, 0.35, -0.74, -0.7, -0.7, -0.17, -0.63])
if with_base:
    # base pose is specified by [x, y, theta]
    base_pose_start = [-0.5, 0.8, 0]
    av_start = np.hstack([av_start, base_pose_start])

# solve inverse kinematics to obtain av_goal
joint_angles = np.deg2rad([-60, 74, -70, -120, -20, -30, 180])
set_robot_config(robot_model, joint_list, joint_angles)
target_coords = skrobot.coordinates.Coordinates([0.8, -0.6, 0.8], [0, 0, 0])

right_arm_end_coords = skrobot.coordinates.CascadedCoords(
    parent=robot_model.r_gripper_tool_frame,
    name='right_arm_end_coords')
robot_model.inverse_kinematics(
    target_coords=target_coords,
    link_list=link_list,
    move_target=right_arm_end_coords, rotation_axis=True)
av_goal = get_robot_config(robot_model, joint_list, with_base=with_base)

# collision checker setup
# SweptSphereSdfCollisionChecker for SQP solver (mesh SDF-based)
# Only uses moving arm links for SQP (original behavior)
sscc = SweptSphereSdfCollisionChecker(box.sdf, robot_model)
for link in coll_link_list:
    sscc.add_collision_link(link)

# RobotCollisionChecker for new solvers and verification
robot_coll_checker = RobotCollisionChecker(robot_model)
print("Adding collision links...")
for link in coll_link_list:
    robot_coll_checker.add_link(link)
print(f"  Total collision geometries: {robot_coll_checker.n_feature}")

robot_coll_checker.add_world_obstacle(box)

# Note: Self-collision checking disabled for this example
# (SweptSphereSdfCollisionChecker also doesn't check self-collision)
# For full-body self-collision, geometry approximations cause many false positives.
# Consider using SDF-based methods or explicit collision meshes instead.

# visualization
print("show trajectory")
if args.viewer == 'trimesh':
    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
elif args.viewer == 'pyrender':
    viewer = skrobot.viewers.PyrenderViewer(resolution=(640, 480))

viewer.add(robot_model)
viewer.add(box)
viewer.add(Axis(pos=target_coords.worldpos(), rot=target_coords.worldrot()))

# Add collision spheres to viewer (choose one)
if args.solver in ['jaxls', 'gradient_descent', 'scipy']:
    # Use RobotCollisionChecker for new solvers
    robot_coll_checker.add_coll_spheres_to_viewer(viewer)
else:
    # Use SweptSphereSdfCollisionChecker for SQP solver
    sscc.add_coll_spheres_to_viewer(viewer)
viewer.show()
viewer.set_camera([0, 0, np.pi / 2.0])
# motion planning
ts = time.time()
n_waypoint = args.n

if args.solver in ['jaxls', 'gradient_descent', 'scipy']:
    # New unified trajectory optimization interface
    print(f"Using {args.solver} solver with unified interface...")
    from skrobot.planner.trajectory_optimization import TrajectoryProblem
    from skrobot.planner.trajectory_optimization.solvers import create_solver
    from skrobot.planner.trajectory_optimization.trajectory import interpolate_trajectory

    # Create obstacle representation for trajectory optimization
    # Use the box directly since RobotCollisionChecker may store it as SDF
    world_obstacles = []

    # Box obstacle - approximate as sphere for gradient-based optimization
    # Use half-diagonal as radius to ensure coverage
    box_half_extents = np.array(box.extents) / 2
    box_center = box.worldpos()
    box_radius = float(np.linalg.norm(box_half_extents)) * 1.0  # Full coverage
    world_obstacles.append({
        'type': 'sphere',
        'center': box_center.tolist(),
        'radius': box_radius
    })
    print(f"  Obstacle: Box at {box_center}, approximated as sphere r={box_radius:.3f}")

    # Create problem definition (backend-agnostic)
    problem = TrajectoryProblem(
        robot_model=robot_model,
        link_list=link_list,
        n_waypoints=n_waypoint,
        dt=0.1,
        move_target=right_arm_end_coords,
    )

    # Add costs and constraints
    problem.add_smoothness_cost(weight=1.0)
    problem.add_acceleration_cost(weight=0.1)
    problem.add_collision_cost(
        collision_link_list=coll_link_list,
        world_obstacles=world_obstacles,
        weight=1000.0,
        activation_distance=0.15,
    )
    problem.add_self_collision_cost(
        weight=1000.0,
        activation_distance=0.02,
    )

    # Create solver
    if args.solver == 'jaxls':
        solver = create_solver('jaxls', max_iterations=args.iterations, verbose=True)
    elif args.solver == 'scipy':
        solver = create_solver(
            'scipy',
            max_iterations=args.iterations,
            safety_margin=5e-2,
            verbose=True,
        )
    else:
        solver = create_solver(
            'gradient_descent',
            max_iterations=args.iterations,
            learning_rate=0.001,
            verbose=True,
        )

    # Create initial trajectory (linear interpolation)
    av_start_arm = av_start[:7]
    av_goal_arm = av_goal[:7]
    initial_traj = interpolate_trajectory(av_start_arm, av_goal_arm, n_waypoint)

    # Solve (scipy solver can use SDF collision checker for more accurate collision)
    if args.solver == 'scipy':
        # Use full trajectory with base for scipy to match original sqp behavior
        initial_traj_full = interpolate_trajectory(av_start, av_goal, n_waypoint)
        # Create a problem with full DOF for scipy
        from copy import deepcopy
        problem_full = deepcopy(problem)
        problem_full.n_joints = len(av_start)
        problem_full.joint_list = joint_list
        problem_full.joint_limits_lower = np.array([
            j.min_angle if j.min_angle is not None else -np.inf
            for j in joint_list
        ])
        problem_full.joint_limits_upper = np.array([
            j.max_angle if j.max_angle is not None else np.inf
            for j in joint_list
        ])
        if with_base:
            # Add base limits (no limits for base)
            problem_full.joint_limits_lower = np.concatenate([
                problem_full.joint_limits_lower, [-np.inf, -np.inf, -np.inf]
            ])
            problem_full.joint_limits_upper = np.concatenate([
                problem_full.joint_limits_upper, [np.inf, np.inf, np.inf]
            ])
        result = solver.solve(
            problem_full, initial_traj_full,
            collision_checker=sscc,
            with_base=with_base,
            joint_list=joint_list,
        )
    else:
        result = solver.solve(problem, initial_traj)
    print(f"Solver result: success={result.success}, "
          f"iterations={result.iterations}, message={result.message}")

    # Convert to av_seq with optional base pose
    if args.solver == 'scipy':
        # scipy solver already optimizes full trajectory including base
        av_seq = result.trajectory
    else:
        # JAX solvers optimize arm only, need to add base pose
        av_seq = []
        for i in range(n_waypoint):
            if with_base:
                t = i / (n_waypoint - 1)
                base_pose = (1 - t) * np.array(av_start[7:]) + t * np.array(av_goal[7:])
                av = np.concatenate([result.trajectory[i], base_pose])
            else:
                av = result.trajectory[i]
            av_seq.append(av)
        av_seq = np.array(av_seq)

else:
    # SQP-based trajectory planning (original method)
    print("Using SQP-based trajectory optimizer...")
    # Trajectory planning with optional visualization
    if args.trajectory_visualization:
        with trajectory_visualization(
            viewer=viewer,
            robot_model=robot_model,
            joint_list=joint_list,
            sleep_time=0.3,
            with_base=with_base,
            update_every_n_iterations=1,
            debug=False,  # Disable debug for normal use
            waypoint_mode='cycle',  # Cycle through waypoints to see trajectory evolution
        ):
            av_seq = sqp_plan_trajectory(
                sscc, av_start, av_goal, joint_list, n_waypoint,
                safety_margin=5.0e-2, with_base=with_base)
    else:
        av_seq = sqp_plan_trajectory(
            sscc, av_start, av_goal, joint_list, n_waypoint,
            safety_margin=5.0e-2, with_base=with_base)

print("solving time : {0} sec".format(time.time() - ts))

# Verify solution with RobotCollisionChecker
print("\nVerifying trajectory with RobotCollisionChecker...")
collision_free = True
for i, av in enumerate(av_seq):
    set_robot_config(robot_model, joint_list, av, with_base=with_base)
    min_dist = robot_coll_checker.compute_min_distance()
    if min_dist < 0:
        print(f"  Waypoint {i}: COLLISION (min_dist={min_dist:.4f})")
        collision_free = False
    elif min_dist < 0.02:
        print(f"  Waypoint {i}: Close to collision (min_dist={min_dist:.4f})")
if collision_free:
    print("  All waypoints are collision-free!")

arm_point_history = []
line_string = None
for av in av_seq:
    set_robot_config(robot_model, joint_list, av, with_base=with_base)
    arm_point_history.append(right_arm_end_coords.worldpos())

    # update arm trajectory visualization
    if line_string is not None:
        viewer.delete(line_string)
    if len(arm_point_history) > 1:
        line_string = LineString(np.array(arm_point_history))
        viewer.add(line_string)

    # Update collision sphere colors
    if args.solver in ['jaxls', 'gradient_descent', 'scipy']:
        robot_coll_checker.update_color()
    else:
        sscc.update_color()
    viewer.redraw()
    time.sleep(1.0)

if not args.no_interactive:
    print('==> Press [q] to close window')
    while viewer.is_active:
        time.sleep(0.1)
        viewer.redraw()
viewer.close()
time.sleep(1.0)

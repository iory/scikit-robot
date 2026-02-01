#!/usr/bin/env python
"""Benchmark script comparing IK methods and backends.

This script compares the performance of:
- Naive IK (sequential inverse_kinematics calls)
- Batch IK with NumPy backend
- Batch IK with JAX backend

Key differences between methods:

1. Naive IK (inverse_kinematics):
   - Uses Jacobian pseudo-inverse method (Newton-like convergence)
   - High accuracy: typically achieves 1mm position error
   - Sequential: solves one target at a time
   - Good convergence properties

2. Batch IK (batch_inverse_kinematics):
   - Uses gradient descent optimization
   - Lower accuracy: typically achieves 10mm position error
   - Parallel: solves all targets simultaneously
   - Sensitive to initial conditions (may get stuck in local minima)
   - Much faster for large batches due to vectorization/GPU

Trade-offs:
- Naive IK: Higher success rate, slower
- Batch IK: Lower success rate, much faster (3000x+ with JAX JIT)

Usage:
    python benchmarks/ik_benchmark.py
    python benchmarks/ik_benchmark.py --robot fetch
    python benchmarks/ik_benchmark.py --poses 1 10 100 500
    python benchmarks/ik_benchmark.py --methods batch_numpy batch_jax batch_jax_precompiled
    python benchmarks/ik_benchmark.py --attempts 5
    python benchmarks/ik_benchmark.py --reachable  # Use reachability map for targets

Methods:
    naive: Sequential inverse_kinematics calls (Jacobian pseudo-inverse)
    batch_numpy: Batch IK with NumPy backend (gradient descent)
    batch_jax: Batch IK with JAX backend (gradient descent, JIT per call)
    batch_jax_precompiled: Batch IK with pre-compiled JAX solver (fastest)
"""

import argparse
import time

import numpy as np

import skrobot
from skrobot.coordinates import Coordinates


def generate_random_targets(robot, end_coords, n_poses, seed=42):
    """Generate random target poses (may be unreachable).

    Parameters
    ----------
    robot : RobotModel
        Robot model.
    end_coords : CascadedCoords
        End-effector coordinates.
    n_poses : int
        Number of poses to generate.
    seed : int
        Random seed.

    Returns
    -------
    list of Coordinates
        Target poses.
    """
    rng = np.random.RandomState(seed)
    targets = []

    # Get current end-effector pose as reference
    base_pos = end_coords.worldpos()

    # Generate poses in a reasonable workspace
    for _ in range(n_poses):
        # Random offset from current position
        offset = rng.uniform(-0.3, 0.3, 3)
        pos = base_pos + offset

        # Random rotation
        rpy = rng.uniform(-np.pi / 4, np.pi / 4, 3)
        target = Coordinates(pos=pos)
        target.rotate(rpy[0], 'x')
        target.rotate(rpy[1], 'y')
        target.rotate(rpy[2], 'z')
        targets.append(target)

    return targets


def generate_reachable_targets(robot, link_list, end_coords, n_poses, seed=42):
    """Generate reachable target poses using ReachabilityMap.

    Parameters
    ----------
    robot : RobotModel
        Robot model.
    link_list : list
        Kinematic chain links.
    end_coords : CascadedCoords
        End-effector coordinates.
    n_poses : int
        Number of poses to generate.
    seed : int
        Random seed.

    Returns
    -------
    list of Coordinates
        Reachable target poses.
    """
    from skrobot.kinematics.reachability_map import ReachabilityMap

    print("Computing reachability map for target generation...")
    rmap = ReachabilityMap(robot, link_list, end_coords, voxel_size=0.05)
    rmap.compute(n_samples=50000, seed=seed, verbose=False, orientation_bins=0)

    # Sample from reachable voxels
    rng = np.random.RandomState(seed)
    reachable_positions = rmap.get_reachable_points()

    if len(reachable_positions) == 0:
        raise ValueError("No reachable positions found")

    print(f"Found {len(reachable_positions)} reachable positions")

    # Sample positions
    indices = rng.choice(len(reachable_positions), size=n_poses, replace=True)
    sampled_positions = reachable_positions[indices]

    # Create targets with random rotations
    targets = []
    base_rot = end_coords.worldrot()
    for pos in sampled_positions:
        # Small random rotation from current orientation
        rpy = rng.uniform(-np.pi / 6, np.pi / 6, 3)
        target = Coordinates(pos=pos, rot=base_rot)
        target.rotate(rpy[0], 'x')
        target.rotate(rpy[1], 'y')
        target.rotate(rpy[2], 'z')
        targets.append(target)

    return targets


def benchmark_naive_ik(robot, link_list, end_coords, targets, stop=100):
    """Benchmark naive sequential IK.

    Parameters
    ----------
    robot : RobotModel
        Robot model.
    link_list : list
        Kinematic chain links.
    end_coords : CascadedCoords
        End-effector coordinates.
    targets : list
        Target poses.
    stop : int
        Max iterations per IK solve.

    Returns
    -------
    dict
        Benchmark results.
    """
    n_poses = len(targets)
    print(f"\n{'=' * 60}")
    print("Method: Naive IK (sequential inverse_kinematics)")
    print(f"Poses: {n_poses}")
    print(f"{'=' * 60}")

    # Store initial angles
    initial_angles = robot.angle_vector()

    successes = 0
    t_start = time.time()

    for target in targets:
        # Reset to initial pose
        robot.angle_vector(initial_angles)

        result = robot.inverse_kinematics(
            target,
            link_list=link_list,
            move_target=end_coords,
            stop=stop,
        )
        if result is not False:
            successes += 1

    t_total = time.time() - t_start

    # Restore initial angles
    robot.angle_vector(initial_angles)

    ik_rate = n_poses / t_total
    success_rate = successes / n_poses * 100

    results = {
        'method': 'naive',
        'n_poses': n_poses,
        'total_time': t_total,
        'ik_rate': ik_rate,
        'successes': successes,
        'success_rate': success_rate,
    }

    print("\nResults:")
    print(f"  Total time: {t_total:.3f}s")
    print(f"  IK rate: {ik_rate:.1f} IK/sec")
    print(f"  Success: {successes}/{n_poses} ({success_rate:.1f}%)")

    return results


def benchmark_batch_ik(robot, link_list, end_coords, targets,
                       backend='numpy', stop=500, attempts_per_pose=3):
    """Benchmark batch IK with specified backend.

    Parameters
    ----------
    robot : RobotModel
        Robot model.
    link_list : list
        Kinematic chain links.
    end_coords : CascadedCoords
        End-effector coordinates.
    targets : list
        Target poses.
    backend : str
        Backend name ('numpy' or 'jax').
    stop : int
        Max iterations.
    attempts_per_pose : int
        Number of random initial attempts per pose.

    Returns
    -------
    dict
        Benchmark results.
    """
    n_poses = len(targets)
    print(f"\n{'=' * 60}")
    print(f"Method: Batch IK ({backend.upper()} backend)")
    print(f"Poses: {n_poses}, Attempts: {attempts_per_pose}")
    print(f"{'=' * 60}")

    # For JAX, run the same batch size first to trigger JIT compilation
    if backend == 'jax':
        print(f"JIT compilation run ({n_poses} poses)...")
        t_warmup_start = time.time()
        robot.batch_inverse_kinematics(
            targets,
            link_list=link_list,
            move_target=end_coords,
            stop=stop,
            backend=backend,
            attempts_per_pose=attempts_per_pose,
        )
        t_warmup = time.time() - t_warmup_start
        print(f"JIT compile time: {t_warmup:.3f}s")
    else:
        # NumPy warm-up (smaller batch is fine)
        print("Warm-up run (10 poses)...")
        warmup_targets = targets[:min(10, len(targets))]
        t_warmup_start = time.time()
        robot.batch_inverse_kinematics(
            warmup_targets,
            link_list=link_list,
            move_target=end_coords,
            stop=stop,
            backend=backend,
            attempts_per_pose=attempts_per_pose,
        )
        t_warmup = time.time() - t_warmup_start
        print(f"Warm-up time: {t_warmup:.3f}s")

    # Main benchmark (post-JIT for JAX)
    print(f"\nMain benchmark ({n_poses} poses)...")
    t_start = time.time()
    solutions, success_flags, errors = robot.batch_inverse_kinematics(
        targets,
        link_list=link_list,
        move_target=end_coords,
        stop=stop,
        backend=backend,
        attempts_per_pose=attempts_per_pose,
    )
    t_total = time.time() - t_start

    successes = int(np.sum(success_flags))
    ik_rate = n_poses / t_total
    success_rate = successes / n_poses * 100

    results = {
        'method': f'batch_{backend}',
        'n_poses': n_poses,
        'warmup_time': t_warmup,
        'total_time': t_total,
        'ik_rate': ik_rate,
        'successes': successes,
        'success_rate': success_rate,
    }

    print("\nResults:")
    print(f"  Total time: {t_total:.3f}s")
    print(f"  IK rate: {ik_rate:.1f} IK/sec")
    print(f"  Success: {successes}/{n_poses} ({success_rate:.1f}%)")

    return results


def benchmark_batch_ik_precompiled(robot, link_list, end_coords, targets,
                                   backend='jax', stop=500, attempts_per_pose=1,
                                   thre=0.01, rthre=np.deg2rad(5.0)):
    """Benchmark batch IK with pre-compiled solver (reused across calls).

    This shows the true post-JIT performance by reusing the solver object.

    Note: Batch IK uses gradient descent which converges slower than the
    Jacobian pseudo-inverse method used in naive IK. Default thresholds
    are looser (10mm, 5deg) vs naive IK (1mm, 1deg) to account for this.
    The trade-off is speed over accuracy.

    Parameters
    ----------
    robot : RobotModel
        Robot model.
    link_list : list
        Kinematic chain links.
    end_coords : CascadedCoords
        End-effector coordinates.
    targets : list
        Target poses.
    backend : str
        Backend name ('numpy' or 'jax').
    stop : int
        Max iterations.
    attempts_per_pose : int
        Number of attempts per pose.
    thre : float
        Position threshold (default 0.01m = 10mm, looser than naive IK).
    rthre : float
        Rotation threshold (default ~5deg, looser than naive IK).

    Returns
    -------
    dict
        Benchmark results.
    """
    from skrobot.kinematics.differentiable import create_batch_ik_solver

    n_poses = len(targets)
    print(f"\n{'=' * 60}")
    print(f"Method: Batch IK Pre-compiled ({backend.upper()} backend)")
    print(f"Poses: {n_poses}, Attempts: {attempts_per_pose}")
    print(f"{'=' * 60}")

    # Create solver once
    print("Creating solver...")
    t_create_start = time.time()
    solver = create_batch_ik_solver(robot, link_list, end_coords,
                                    backend_name=backend)
    t_create = time.time() - t_create_start
    print(f"Solver creation time: {t_create:.3f}s")

    # Prepare data
    target_positions = np.array([t.worldpos() for t in targets])
    target_rotations = np.array([t.worldrot() for t in targets])
    current_angles = np.array([link.joint.joint_angle() for link in link_list])
    initial_angles = np.tile(current_angles, (n_poses, 1))

    # JIT compilation run (run twice to ensure cache is warm)
    print(f"JIT compilation run ({n_poses} poses, {attempts_per_pose} attempts)...")
    t_compile_start = time.time()
    solver(target_positions, target_rotations,
           initial_angles=initial_angles, max_iterations=stop,
           attempts_per_pose=attempts_per_pose,
           pos_threshold=thre, rot_threshold=rthre)
    # Second run to ensure JIT cache is fully warm
    solver(target_positions, target_rotations,
           initial_angles=initial_angles, max_iterations=stop,
           attempts_per_pose=attempts_per_pose,
           pos_threshold=thre, rot_threshold=rthre)
    t_compile = time.time() - t_compile_start
    print(f"JIT compile + warmup time: {t_compile:.3f}s")

    # Main benchmark (post-JIT)
    print(f"\nMain benchmark ({n_poses} poses, post-JIT)...")
    t_start = time.time()
    solutions, success_flags, errors = solver(
        target_positions, target_rotations,
        initial_angles=initial_angles, max_iterations=stop,
        attempts_per_pose=attempts_per_pose,
        pos_threshold=thre, rot_threshold=rthre)
    t_total = time.time() - t_start

    successes = int(np.sum(success_flags))
    ik_rate = n_poses / t_total
    success_rate = successes / n_poses * 100

    results = {
        'method': f'batch_{backend}_precompiled',
        'n_poses': n_poses,
        'compile_time': t_compile,
        'total_time': t_total,
        'ik_rate': ik_rate,
        'successes': successes,
        'success_rate': success_rate,
    }

    print("\nResults:")
    print(f"  Total time: {t_total:.6f}s")
    print(f"  IK rate: {ik_rate:.1f} IK/sec")
    print(f"  Success: {successes}/{n_poses} ({success_rate:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark IK methods: naive vs batch (NumPy/JAX)"
    )
    parser.add_argument(
        '--robot', type=str, default='fetch',
        choices=['panda', 'fetch', 'pr2'],
        help='Robot model to use'
    )
    parser.add_argument(
        '--poses', type=int, nargs='+',
        default=[10, 50, 100, 500],
        help='Number of poses to test'
    )
    parser.add_argument(
        '--methods', type=str, nargs='+',
        default=['naive', 'batch_numpy', 'batch_jax'],
        choices=['naive', 'batch_numpy', 'batch_jax', 'batch_jax_precompiled'],
        help='Methods to benchmark'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--attempts', type=int, default=3,
        help='Attempts per pose for batch IK'
    )
    parser.add_argument(
        '--reachable', action='store_true',
        help='Use reachability map to generate reachable targets'
    )
    args = parser.parse_args()

    # Setup robot
    print(f"Loading {args.robot}...")
    if args.robot == 'panda':
        robot = skrobot.models.Panda()
        link_list = [
            robot.panda_link1, robot.panda_link2, robot.panda_link3,
            robot.panda_link4, robot.panda_link5, robot.panda_link6,
            robot.panda_link7
        ]
        end_coords = skrobot.coordinates.CascadedCoords(
            parent=robot.panda_hand, name='end_coords'
        )
    elif args.robot == 'fetch':
        robot = skrobot.models.Fetch()
        link_list = robot.rarm.link_list
        end_coords = robot.rarm.end_coords
    else:  # pr2
        robot = skrobot.models.PR2()
        link_list = robot.rarm.link_list
        end_coords = robot.rarm.end_coords

    print(f"Robot: {args.robot}")
    print(f"Joints: {len(link_list)}")

    # Generate targets (use max poses count)
    max_poses = max(args.poses)
    if args.reachable:
        print(f"\nGenerating {max_poses} reachable target poses...")
        all_targets = generate_reachable_targets(
            robot, link_list, end_coords, max_poses, seed=args.seed
        )
    else:
        print(f"\nGenerating {max_poses} random target poses...")
        all_targets = generate_random_targets(
            robot, end_coords, max_poses, seed=args.seed
        )

    # Run benchmarks
    all_results = []

    for n_poses in args.poses:
        targets = all_targets[:n_poses]

        for method in args.methods:
            try:
                if method == 'naive':
                    results = benchmark_naive_ik(
                        robot, link_list, end_coords, targets
                    )
                elif method == 'batch_numpy':
                    results = benchmark_batch_ik(
                        robot, link_list, end_coords, targets,
                        backend='numpy',
                        attempts_per_pose=args.attempts,
                    )
                elif method == 'batch_jax':
                    results = benchmark_batch_ik(
                        robot, link_list, end_coords, targets,
                        backend='jax',
                        attempts_per_pose=args.attempts,
                    )
                elif method == 'batch_jax_precompiled':
                    results = benchmark_batch_ik_precompiled(
                        robot, link_list, end_coords, targets,
                        backend='jax',
                        attempts_per_pose=args.attempts,
                    )
                all_results.append(results)
            except Exception as e:
                print(f"\nError with {method}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    # Group by pose count
    pose_counts = sorted(set(r['n_poses'] for r in all_results))

    for n_poses in pose_counts:
        print(f"\n{n_poses} poses:")
        print("-" * 50)

        results_for_poses = [r for r in all_results if r['n_poses'] == n_poses]

        # Find naive result as baseline
        naive_result = next(
            (r for r in results_for_poses if r['method'] == 'naive'), None
        )

        for r in results_for_poses:
            line = (f"  {r['method']:24s}: {r['total_time']:.6f}s "
                    f"({r['ik_rate']:.1f} IK/sec)")
            if naive_result and r['method'] != 'naive':
                speedup = naive_result['total_time'] / r['total_time']
                line += f"  [{speedup:.2f}x vs naive]"
            print(line)

    # Final summary table
    print("\n" + "=" * 90)
    print("DETAILED RESULTS TABLE")
    print("=" * 90)
    print(f"{'Method':<24} {'Poses':>8} {'Time (s)':>12} {'IK/sec':>12} "
          f"{'Success':>8} {'Rate':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['method']:<24} {r['n_poses']:>8} {r['total_time']:>12.6f} "
              f"{r['ik_rate']:>12.1f} {r['successes']:>8} "
              f"{r['success_rate']:>7.1f}%")

    # Trade-off notes
    print("\n" + "=" * 90)
    print("NOTES ON METHOD TRADE-OFFS")
    print("=" * 90)
    print("""
Naive IK (inverse_kinematics):
  - Jacobian pseudo-inverse method with high accuracy (1mm tolerance)
  - Sequential processing
  - Higher success rate due to better convergence

Batch IK (batch_inverse_kinematics):
  - Gradient descent optimization (looser 10mm tolerance)
  - Parallel processing with vectorization/JIT
  - Lower success rate (sensitive to initial conditions)
  - Much faster throughput (especially batch_jax_precompiled)

Use cases:
  - Naive IK: When accuracy matters, single target solutions
  - Batch IK: Sampling-based planning, reachability analysis,
              scenarios where speed >> accuracy
""")


if __name__ == '__main__':
    main()

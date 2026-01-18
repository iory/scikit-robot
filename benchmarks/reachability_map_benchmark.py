#!/usr/bin/env python
"""Benchmark script comparing backends for ReachabilityMap.

This script compares the performance of NumPy and JAX backends
for computing robot reachability maps.

Usage:
    python examples/reachability_map_benchmark.py
    python examples/reachability_map_benchmark.py --robot fetch
    python examples/reachability_map_benchmark.py --samples 100000 500000 1000000
    python examples/reachability_map_benchmark.py --backends jax
"""

import argparse
import time

import skrobot
from skrobot.kinematics.reachability_map import ReachabilityMap


def benchmark_backend(robot, link_list, end_coords, backend, n_samples, seed=42):
    """Benchmark a specific backend.

    Parameters
    ----------
    robot : RobotModel
        Robot model.
    link_list : list
        Kinematic chain links.
    end_coords : CascadedCoords
        End-effector coordinates.
    backend : str
        Backend name ('numpy' or 'jax').
    n_samples : int
        Number of samples.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Benchmark results.
    """
    print(f"\n{'=' * 60}")
    print(f"Backend: {backend.upper()}")
    print(f"Samples: {n_samples:,}")
    print(f"{'=' * 60}")

    # Create reachability map
    t_init_start = time.time()
    rmap = ReachabilityMap(
        robot, link_list, end_coords,
        voxel_size=0.05,
        backend=backend
    )
    t_init = time.time() - t_init_start
    print(f"Initialization time: {t_init:.3f}s")

    # Warm-up run (especially important for JAX JIT compilation)
    print("Warm-up run (1000 samples)...")
    t_warmup_start = time.time()
    rmap.compute(n_samples=1000, seed=seed, verbose=False, orientation_bins=0)
    t_warmup = time.time() - t_warmup_start
    print(f"Warm-up time: {t_warmup:.3f}s")

    # Main benchmark
    print(f"\nMain benchmark ({n_samples:,} samples)...")
    t_main_start = time.time()
    rmap.compute(n_samples=n_samples, seed=seed, verbose=True, orientation_bins=50)
    t_main = time.time() - t_main_start

    # Compute FK rate
    fk_rate = n_samples / t_main

    results = {
        'backend': backend,
        'n_samples': n_samples,
        'init_time': t_init,
        'warmup_time': t_warmup,
        'main_time': t_main,
        'fk_rate': fk_rate,
        'reachable_voxels': rmap.n_reachable_voxels,
        'reachable_volume': rmap.reachable_volume,
    }

    print("\nResults:")
    print(f"  Total time: {t_main:.3f}s")
    print(f"  FK rate: {fk_rate:,.0f} FK/sec")
    print(f"  Reachable voxels: {rmap.n_reachable_voxels:,}")
    print(f"  Reachable volume: {rmap.reachable_volume:.4f} m³")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NumPy vs JAX for ReachabilityMap"
    )
    parser.add_argument(
        '--robot', type=str, default='panda',
        choices=['panda', 'fetch', 'pr2'],
        help='Robot model to use'
    )
    parser.add_argument(
        '--samples', type=int, nargs='+',
        default=[10000, 50000, 100000, 1000000],
        help='Sample counts to test'
    )
    parser.add_argument(
        '--backends', type=str, nargs='+',
        default=['numpy', 'jax'],
        help='Backends to benchmark'
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

    # Run benchmarks
    all_results = []

    for n_samples in args.samples:
        for backend in args.backends:
            try:
                results = benchmark_backend(
                    robot, link_list, end_coords,
                    backend, n_samples
                )
                all_results.append(results)
            except Exception as e:
                print(f"\nError with {backend}: {e}")
                continue

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    # Group by sample count
    sample_counts = sorted(set(r['n_samples'] for r in all_results))

    for n_samples in sample_counts:
        print(f"\n{n_samples:,} samples:")
        print("-" * 50)

        results_for_samples = [r for r in all_results if r['n_samples'] == n_samples]

        # Find NumPy result as baseline
        numpy_result = next(
            (r for r in results_for_samples if r['backend'] == 'numpy'), None
        )

        for r in results_for_samples:
            line = f"  {r['backend']:8s}: {r['main_time']:.3f}s ({r['fk_rate']:,.0f} FK/sec)"
            if numpy_result and r['backend'] != 'numpy':
                speedup = numpy_result['main_time'] / r['main_time']
                line += f"  [{speedup:.2f}x vs numpy]"
            print(line)

    # Final summary table
    print("\n" + "=" * 70)
    print("DETAILED RESULTS TABLE")
    print("=" * 70)
    print(f"{'Backend':<10} {'Samples':>12} {'Time (s)':>12} {'FK/sec':>15} {'Voxels':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['backend']:<10} {r['n_samples']:>12,} {r['main_time']:>12.3f} "
              f"{r['fk_rate']:>15,.0f} {r['reachable_voxels']:>10,}")


if __name__ == '__main__':
    main()

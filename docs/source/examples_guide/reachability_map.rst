================
Reachability Map
================

**Example script**: ``examples/reachability_map_demo.py``

This example demonstrates how to compute and visualize a robot's workspace reachability map
using batch forward kinematics with JAX or NumPy backend.

Source Code
===========

https://github.com/iory/scikit-robot/blob/main/examples/reachability_map_demo.py

Overview
========

A reachability map is a discretized representation of a robot's workspace that indicates
which positions (and orientations) are reachable by the end-effector. This is useful for:

- Task planning: Quickly check if a target is reachable before attempting IK
- Workspace analysis: Understand the extent and characteristics of the robot's workspace
- Grasp planning: Filter candidate grasp positions to only reachable ones
- Robot placement: Optimize where to place the robot for a given task

The ``ReachabilityMap`` class uses batch forward kinematics computation, with optional
JIT compilation via JAX for high performance.

Basic Usage
===========

.. code-block:: python

    import skrobot
    from skrobot.coordinates import CascadedCoords
    from skrobot.kinematics import ReachabilityMap

    # Load robot
    robot = skrobot.models.Panda()

    # Get kinematic chain
    link_list = robot.rarm.link_list
    end_coords = robot.rarm.end_coords

    # Create and compute reachability map
    rmap = ReachabilityMap(robot, link_list, end_coords, voxel_size=0.05)
    rmap.compute(n_samples=500000)

    # Check reachability
    target = [0.5, -0.3, 0.8]
    if rmap.is_reachable(target):
        print("Target is reachable!")
        print(f"Manipulability: {rmap.get_manipulability(target):.4f}")

Key Concepts
============

Voxel Grid
----------

The workspace is discretized into a 3D voxel grid. Each voxel stores:

- **Reachability count**: Number of configurations that can reach this voxel
- **Reachability index**: Ratio of reachable orientations (0.0 to 1.0)
- **Manipulability**: Average manipulability measure for configurations reaching this voxel

.. code-block:: python

    # Create map with 5cm voxel size
    rmap = ReachabilityMap(robot, link_list, end_coords, voxel_size=0.05)

    # Create map with 10cm voxel size (faster, less detailed)
    rmap = ReachabilityMap(robot, link_list, end_coords, voxel_size=0.10)

Sampling Methods
----------------

Two sampling methods are available:

**Random sampling** (default): Uniformly samples joint configurations at random.

.. code-block:: python

    rmap.compute(n_samples=500000, sampling='random', seed=42)

**Grid sampling**: Systematically samples joint space on a regular grid.
Total samples = ``bins_per_joint^n_joints``.

.. code-block:: python

    # For a 7-DOF robot with 10 bins: 10^7 = 10M samples
    rmap.compute(sampling='grid', bins_per_joint=10)

Orientation-Aware Reachability
------------------------------

By default, the reachability map tracks orientation diversity using Fibonacci sphere sampling.
The **Reachability Index** measures what fraction of orientations are reachable at each position:

.. code-block:: python

    # Track 50 orientation bins (default)
    rmap.compute(n_samples=500000, orientation_bins=50)

    # Position-only mode (no orientation tracking)
    rmap.compute(n_samples=500000, orientation_bins=0)

Backend Selection
=================

The ``ReachabilityMap`` supports two backends:

**JAX backend** (recommended): Uses JIT compilation for fast batch computation.

.. code-block:: python

    # Auto-select (uses JAX if available)
    rmap = ReachabilityMap(robot, link_list, end_coords, backend=None)

    # Explicitly use JAX
    rmap = ReachabilityMap(robot, link_list, end_coords, backend='jax')

**NumPy backend**: Pure NumPy implementation, no additional dependencies.

.. code-block:: python

    rmap = ReachabilityMap(robot, link_list, end_coords, backend='numpy')

Performance comparison (approximate):

- JAX: ~1,000,000 FK/sec (with JIT compilation)
- NumPy: ~100,000 FK/sec

Querying the Reachability Map
=============================

Single Position Query
---------------------

.. code-block:: python

    position = [0.5, -0.3, 0.8]

    # Check if reachable
    if rmap.is_reachable(position):
        print("Reachable!")

    # Get reachability score (number of configurations)
    score = rmap.get_reachability(position)
    print(f"Reachability score: {score}")

    # Get manipulability
    manip = rmap.get_manipulability(position)
    print(f"Manipulability: {manip:.4f}")

Batch Position Query
--------------------

.. code-block:: python

    import numpy as np

    # Query multiple positions efficiently
    positions = np.array([
        [0.5, -0.3, 0.8],
        [0.6, 0.0, 1.0],
        [0.4, 0.2, 0.6],
    ])
    scores = rmap.get_reachability_at_positions(positions)
    print(f"Scores: {scores}")

Filtering Targets
-----------------

.. code-block:: python

    # Filter to only reachable positions
    candidate_targets = np.random.uniform(-1, 1, (100, 3))
    reachable_targets, indices = rmap.filter_reachable_targets(candidate_targets)
    print(f"Reachable: {len(reachable_targets)} / {len(candidate_targets)}")

Finding Nearest Reachable Position
----------------------------------

.. code-block:: python

    # Find nearest reachable position to an unreachable target
    unreachable_target = [2.0, 0.0, 1.5]  # Outside workspace
    nearest = rmap.find_nearest_reachable(unreachable_target)
    if nearest is not None:
        print(f"Nearest reachable: {nearest}")

Visualization
=============

Get Point Cloud for Visualization
---------------------------------

.. code-block:: python

    # Get point cloud with colors based on reachability index
    positions, colors = rmap.get_point_cloud(
        color_by='reachability_index',  # or 'reachability', 'manipulability'
        max_points=50000
    )

Using with ViserViewer
----------------------

.. code-block:: python

    import trimesh as tm
    import numpy as np

    # Get point cloud
    positions, colors = rmap.get_point_cloud(color_by='reachability_index')

    # Create viewer
    viewer = skrobot.viewers.ViserViewer()
    viewer.add(robot)

    # Create sphere mesh for visualization
    unit_sphere = tm.creation.icosphere(subdivisions=1, radius=1.0)
    sphere_vertices = unit_sphere.vertices.astype(np.float32)
    sphere_faces = unit_sphere.faces.astype(np.uint32)

    # Add batched spheres
    n_points = len(positions)
    sphere_radius = rmap.voxel_size * 0.4
    server = viewer._server
    server.scene.add_batched_meshes_simple(
        "reachability_spheres",
        vertices=sphere_vertices,
        faces=sphere_faces,
        batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (n_points, 1)).astype(np.float32),
        batched_positions=positions.astype(np.float32),
        batched_scales=np.full((n_points, 3), sphere_radius, dtype=np.float32),
        batched_colors=(colors[:, :3] * 255).astype(np.uint8),
        opacity=0.5,
    )

    viewer.show()

Saving and Loading
==================

Reachability maps can be saved to and loaded from ``.npz`` files:

.. code-block:: python

    # Save
    rmap.compute(n_samples=1000000)
    rmap.save('reachability_map.npz')

    # Load
    rmap2 = ReachabilityMap(robot, link_list, end_coords)
    rmap2.load('reachability_map.npz')

    # Use loaded map
    print(f"Reachable volume: {rmap2.reachable_volume:.3f} m³")

Command Line Interface
======================

The demo script provides a command-line interface for quick experiments:

.. code-block:: bash

    # Basic usage with PR2 robot
    python examples/reachability_map_demo.py

    # Use Panda robot with more samples
    python examples/reachability_map_demo.py --robot panda --samples 1000000

    # Use custom URDF
    python examples/reachability_map_demo.py --urdf /path/to/robot.urdf

    # Color by manipulability
    python examples/reachability_map_demo.py --color manipulability

    # Save result
    python examples/reachability_map_demo.py --save my_rmap.npz

    # Load and visualize
    python examples/reachability_map_demo.py --load my_rmap.npz

Available options:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Option
     - Description
     - Default
   * - ``--robot``
     - Built-in robot (pr2, panda, fetch, r8, nextage, tycoon)
     - pr2
   * - ``--urdf``
     - Path to custom URDF file
     - None
   * - ``--backend``
     - Computation backend (jax, numpy)
     - auto
   * - ``--samples``
     - Number of random samples
     - 2000000
   * - ``--sampling``
     - Sampling method (random, grid)
     - random
   * - ``--voxel-size``
     - Voxel size in meters
     - 0.05
   * - ``--color``
     - Coloring mode (reachability_index, reachability, manipulability)
     - reachability_index
   * - ``--orientation-bins``
     - Number of orientation bins (0 for position-only)
     - 50
   * - ``--save``
     - Save reachability map to file
     - None
   * - ``--load``
     - Load reachability map from file
     - None
   * - ``--z-slice``
     - Show only points within Z range
     - None
   * - ``--alpha``
     - Sphere opacity (0.0-1.0)
     - 0.3

Properties and Statistics
=========================

.. code-block:: python

    # After computing the map
    rmap.compute(n_samples=500000)

    # Get statistics
    print(f"Reachable volume: {rmap.reachable_volume:.3f} m³")
    print(f"Reachable voxels: {rmap.n_reachable_voxels}")
    print(f"Backend: {rmap.backend}")
    print(f"Voxel size: {rmap.voxel_size} m")

    # Workspace bounds
    print(f"X range: {rmap.bounds['x']}")
    print(f"Y range: {rmap.bounds['y']}")
    print(f"Z range: {rmap.bounds['z']}")

    # Get reachable points for further analysis
    points = rmap.get_reachable_points(min_score=5, max_points=10000)

API Reference
=============

Class: ReachabilityMap
----------------------

``skrobot.kinematics.ReachabilityMap(robot_model, link_list, end_coords, voxel_size=0.05, backend=None)``

**Constructor Parameters:**

- ``robot_model``: Robot model instance
- ``link_list``: List of links in the kinematic chain
- ``end_coords``: End-effector coordinates (CascadedCoords)
- ``voxel_size``: Size of each voxel in meters (default: 0.05)
- ``backend``: Backend to use ('jax' or 'numpy', default: auto-select)

**Methods:**

- ``compute(n_samples, seed, verbose, sampling, bins_per_joint, orientation_bins)``: Compute the reachability map
- ``is_reachable(position, threshold)``: Check if a position is reachable
- ``get_reachability(position)``: Get reachability score at a position
- ``get_manipulability(position)``: Get average manipulability at a position
- ``get_reachability_at_positions(positions)``: Batch query for multiple positions
- ``filter_reachable_targets(positions, min_score)``: Filter targets to reachable ones
- ``find_nearest_reachable(position, min_score, max_distance)``: Find nearest reachable position
- ``get_reachable_points(min_score, max_points)``: Get positions of reachable voxels
- ``get_point_cloud(color_by, max_points)``: Get point cloud for visualization
- ``save(filepath)``: Save reachability map to .npz file
- ``load(filepath)``: Load reachability map from .npz file

**Properties:**

- ``reachable_volume``: Total reachable volume in cubic meters
- ``n_reachable_voxels``: Number of reachable voxels
- ``backend``: Name of the backend being used
- ``bounds``: Workspace bounds dictionary
- ``reachability``: 3D array of reachability counts
- ``manipulability``: 3D array of average manipulability
- ``reachability_index``: 3D array of orientation diversity ratio (0-1)

Related Documentation
=====================

- :doc:`batch_ik` - Batch inverse kinematics for multiple IK solutions
- :doc:`../tutorials/inverse_kinematics` - IK tutorial
- :doc:`../reference/robot_model_tips` - Robot model usage tips

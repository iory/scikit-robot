===============================
Self-Collision-Free Trajectory
===============================

**Example script**: ``examples/self_collision_free_trajectory.py``

This example plans a Panda arm motion that avoids the robot colliding with
itself, using per-link GridSDFs as the self-collision model.

Source Code
===========

https://github.com/iory/scikit-robot/blob/main/examples/self_collision_free_trajectory.py

Why a GridSDF Self-Collision Model
==================================

:meth:`~skrobot.planner.trajectory_optimization.problem.TrajectoryProblem.add_self_collision_cost`
approximates every link with a few spheres by default. That is cheap, but on a
robot with bulky links -- the Panda housings are roughly
0.11 x 0.19 x 0.25 m -- the spheres are much fatter than the geometry they
stand in for, so the model reports collisions that are not there. Measured at
the Panda rest pose, over the nine collision links (fingers excluded):

.. list-table::
   :header-rows: 1

   * - Self-collision model
     - Minimum signed distance
   * - ``mode='sphere'`` (default)
     - −0.056 m (reports penetration)
   * - ``mode='gridsdf'``
     - +0.026 m
   * - exact mesh distance (trimesh/fcl)
     - +0.022 m

The sphere model calls a perfectly valid rest pose a collision, which is why
self-collision costs are usually left switched off. ``mode='gridsdf'`` gives
each collision link its own :class:`~skrobot.sdf.GridSDF` and looks the other
links' sampled surface points up in it with trilinear interpolation, which
tracks the exact mesh distance closely enough to drive an optimizer. The
example prints the table above at startup, so the numbers can be reproduced
directly.

The cost is that the residual vector is much larger (one entry per ordered
link pair per surface sample) and each link has to be voxelized once. The
voxelization result is cached on disk, so only the first run pays for it.

What This Example Shows
========================

- ``mode='gridsdf'`` self-collision cost in ``TrajectoryProblem``
- Side-by-side comparison of the sphere, GridSDF and exact mesh distances
- Trajectory optimization with the ``jaxls`` solver
- Headless rendering of the result with the Mitsuba viewer

The start and goal configurations are both self-collision-free, but
interpolating straight between them swings the hand into the base link. The
optimizer pushes the trajectory out of that penetration (default run, 10
waypoints):

.. list-table::
   :header-rows: 1

   * - Trajectory
     - Worst self-collision distance
   * - straight-line initial guess
     - −0.018 m
   * - optimized
     - +0.023 m

In the video below, the straight-line guess is played first with the
penetrating links tinted red, followed by the optimized trajectory:

.. raw:: html

   <video width="100%" controls>
     <source src="../../image/self-collision-gridsdf.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

Running the Example
===================

.. code-block:: bash

   python examples/self_collision_free_trajectory.py

Replay the colliding initial guess before the solution:

.. code-block:: bash

   python examples/self_collision_free_trajectory.py --play-initial

Record the animation headlessly with the Mitsuba viewer -- this is the exact
command that produced the video above:

.. code-block:: bash

   python examples/self_collision_free_trajectory.py \
       --viewer mitsuba -n 24 --play-initial --no-interactive \
       --save-video docs/image/self-collision-gridsdf.mp4

The example needs ``jax`` and ``jaxls``; ``jaxls`` is not on PyPI:

.. code-block:: bash

   pip install jax
   pip install "git+https://github.com/brentyi/jaxls.git"

Using It in Your Own Code
=========================

.. code-block:: python

    problem = TrajectoryProblem(robot, link_list, n_waypoints=10)
    problem.add_collision_cost(collision_link_list, world_obstacles=[])
    problem.add_self_collision_cost(
        mode='gridsdf',      # 'sphere' (default) or 'gridsdf'
        dim_grid=24,         # GridSDF resolution per axis
        n_surface=24,        # surface sample points per link
        weight=1000.0,
        activation_distance=0.02,
    )

``collision_link_list`` should be ordered along the kinematic chain: link
pairs are selected with
:func:`~skrobot.planner.trajectory_optimization.collision.create_self_collision_pairs`,
which treats neighbours *in that list* as adjacent and skips them. Branches
that are physically adjacent but far apart in the list -- a Panda finger and
the hand, for instance -- would otherwise report a permanent contact, so leave
them out of the list.

Related Documentation
=====================

- :doc:`../reference/planner` - Motion planning API
- :doc:`collision_free_trajectory` - World-obstacle avoidance
- :doc:`../reference/sdfs` - Signed Distance Functions
- :doc:`../reference/viewers` - Mitsuba and other viewer backends

==========================
Collision-Free Trajectory
==========================

**Example script**: ``examples/collision_free_trajectory.py``

This example demonstrates collision-free motion planning using SQP-based trajectory optimization.

Source Code
===========

https://github.com/iory/scikit-robot/blob/main/examples/collision_free_trajectory.py

What This Example Shows
========================

- SQP-based trajectory planning
- Swept sphere collision detection
- Box obstacle avoidance
- Smooth trajectory generation from start to goal pose

Running the Example
===================

.. code-block:: bash

   python examples/collision_free_trajectory.py

With trajectory visualization:

.. code-block:: bash

   python examples/collision_free_trajectory.py --trajectory-visualization

Key Features
============

- Creates a box obstacle using SDF (Signed Distance Function)
- Plans collision-free trajectory for PR2 right arm
- Optimizes for smoothness while avoiding obstacles
- Visualizes the planned trajectory

Related Documentation
=====================

- :doc:`../reference/planner` - Motion planning API
- :doc:`../reference/sdfs` - Signed Distance Functions

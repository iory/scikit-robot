Planning
========

Sdf-swept-sphere-based collision checker
----------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   skrobot.planner.SweptSphereSdfCollisionChecker

SQP-based trajectory planner
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   skrobot.planner.sqp_plan_trajectory

Trajectory optimization
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   skrobot.planner.trajectory_optimization.TrajectoryProblem
   skrobot.planner.trajectory_optimization.create_solver

Self-collision models
~~~~~~~~~~~~~~~~~~~~~

``TrajectoryProblem.add_self_collision_cost`` supports two representations of
the robot's own geometry, selected with ``mode``:

``'sphere'`` (default)
    Each link is approximated by a few spheres. Cheap, but conservative
    enough that bulky links report collisions that are not there -- on a
    Panda it flags even the rest pose.

``'gridsdf'``
    Each collision link gets its own :class:`~skrobot.sdf.GridSDF`, and
    penetration is measured by looking the other links' sampled surface
    points up in it. Far more accurate, at the cost of a one-time
    voxelization (cached on disk) and a larger residual vector.

See :doc:`../examples_guide/self_collision_free_trajectory` for a worked
comparison against the exact mesh distance.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   skrobot.planner.trajectory_optimization.collision.create_self_collision_pairs
   skrobot.planner.trajectory_optimization.gridsdf_collision.build_gridsdf_self_data
   skrobot.planner.trajectory_optimization.gridsdf_collision.gridsdf_self_distances
   skrobot.planner.trajectory_optimization.gridsdf_collision.make_gridsdf_self_distance_fn

Swept sphere generator
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   skrobot.planner.swept_sphere.compute_swept_sphere

Planner utils
-------------
.. autosummary::
   :toctree: generated/
   :nosignatures:

   skrobot.planner.utils.scipinize
   skrobot.planner.utils.set_robot_config
   skrobot.planner.utils.get_robot_config
   skrobot.planner.utils.forward_kinematics_multi

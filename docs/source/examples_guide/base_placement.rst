==============
Base Placement
==============

**Example script**: ``examples/base_placement_demo.py``

This example finds one mobile-base pose from which a robot reaches several
end-effector targets without moving the base in between.

Source Code
===========

https://github.com/iory/scikit-robot/blob/main/examples/base_placement_demo.py

Overview
========

``batch_inverse_kinematics(..., use_base='planar')`` gives every target its
own base pose. ``find_base_placement`` solves for a single planar base
``(x, y, yaw)`` shared by all targets, together with one arm configuration
per target. Each target's error depends only on the base and its own arm
configuration, so the Gauss-Newton system is arrow shaped and the base step
comes from a 3x3 Schur complement: the cost grows linearly with the number
of targets.

The optimization is local. ``InverseReachabilityMap`` gives it global
starts: it samples the arm once, inverts each end-effector pose into the
base pose that would put the hand on a target, and lets every target vote
for the ``(x, y, yaw)`` cells its matching samples land in. Cells most
targets voted for become starting bases.

Basic Usage
===========

.. code-block:: python

    import numpy as np
    from skrobot.kinematics import InverseReachabilityMap
    from skrobot.kinematics import find_base_placement
    from skrobot.models import Fetch

    robot = Fetch()
    robot.reset_pose()
    move_target = robot.rarm.end_coords

    irm = InverseReachabilityMap(robot, move_target).compute(
        n_samples=500000)
    result = find_base_placement(
        robot, targets, move_target=move_target, irm=irm)

    print(result.all_reached, result.success)
    print(result.base_pose)          # root-link pose
    print(result.angle_vectors[0])   # robot.angle_vector() for target 0

Options
=======

- ``position_mask`` / ``rotation_mask``: end-effector axes to constrain,
  with the same meaning as in ``inverse_kinematics``; one mask for all
  targets or one per target.
- ``rotation_tolerance``: rotation error per end-effector axis that counts
  as reached, e.g. ``(0, np.deg2rad(60), 0)`` to let the hand turn up to 60
  degrees about its y axis.
- ``base_height``: the root link's fixed height, or a list of heights to
  choose from.
- ``base_bounds`` and ``is_valid_base``: where the base may stand.
  ``result.at_bound`` and ``result.bound_distance`` tell whether the bounds
  held the base back.
- ``target_frame='base'``: targets fixed to the root link. A target can
  also be a function of the base pose, returning its world pose, for a
  fixture that follows the base only partly.
- ``margin_target`` and ``maximize_margin``: keep the joints off their
  limits, and climb the smallest joint-limit margin. Placements that
  reach the same number of targets are ranked by that margin, or by
  ``score``.
- ``initial_angles``: one arm seed or several. The solve is local in the
  arm too, so three to five seeds spread over the arm's working poses make
  the result much less dependent on where the model happens to be.

Results
=======

In the example, Fetch grasps two objects on a table from above and two on
a shelf from the side. Solved from its current base alone, shared-base IK
reaches two of the four; started from the map's candidates it reaches all
four.

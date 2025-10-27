=======================
PR2 Inverse Kinematics
=======================

**Example script**: ``examples/pr2_inverse_kinematics.py``

This example demonstrates inverse kinematics solving with the PR2 robot, including the ``revert_if_fail`` parameter.

Source Code
===========

https://github.com/iory/scikit-robot/blob/main/examples/pr2_inverse_kinematics.py

What This Example Shows
========================

- Basic IK solving for PR2 right arm
- IK visualization during solving
- Comparison of ``revert_if_fail=True`` vs ``False``
- Unreachable target handling

Running the Example
===================

.. code-block:: bash

   python examples/pr2_inverse_kinematics.py

With different options:

.. code-block:: bash

   # Use pyrender viewer
   python examples/pr2_inverse_kinematics.py --viewer pyrender

   # Disable IK visualization
   python examples/pr2_inverse_kinematics.py --no-ik-visualization

   # Skip revert_if_fail demonstration
   python examples/pr2_inverse_kinematics.py --skip-revert-demo

Key Concepts
============

**revert_if_fail Parameter**

- ``True`` (default): Reverts to original joint angles if IK fails
- ``False``: Keeps partial progress even if target not fully reached

This is useful when you want to get as close as possible to an unreachable target.

Related Documentation
=====================

- :doc:`../reference/robot_model_tips` - Comprehensive IK documentation

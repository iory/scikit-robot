Module Assembly
===============

``skrobot.assembly.module_assembly`` composes robots out of reusable URDF
modules: parse each part once into a :class:`~skrobot.assembly.RobotModule`
(its links become connection ports, each a full SE(3) frame), place
instances in a :class:`~skrobot.assembly.RobotAssembly`, connect ports, and
build a combined URDF or a live :class:`~skrobot.model.RobotModel`.

.. code-block:: python

   from skrobot.assembly import RobotAssembly, RobotModule

   arm = RobotModule.from_urdf('arm', 'arm_module.urdf')
   gripper = RobotModule.from_urdf('gripper', 'gripper_module.urdf')

   assembly = RobotAssembly('my_robot')
   assembly.add_module_instance('arm1', arm)
   assembly.add_module_instance('hand', gripper)
   # mate=True seats the child port onto the parent port
   # (keyed connector: origins coincide, Z opposed, X aligned)
   assembly.connect('arm1', 'flange', 'hand', 'base_link', mate=True)
   assembly.set_root('arm1', 'base_link')
   urdf_path = assembly.build()
   robot = assembly.build_robot_model()

Connections are fixed joints by default; pass
``joint_type='revolute'`` (or ``'continuous'`` / ``'prismatic'``) to
turn the connected port pair itself into an articulation, so linkages
can be assembled from bare links with no module-internal joints.

Closed loops
------------

A URDF is a tree, so a closed linkage (four-bar, parallel mechanism)
declares its cut edge with ``connect(..., loop=True)``: the two ports
must coincide at the zero pose and their common Z axis is the passive
hinge.  ``build()`` writes a ``loop_closures.yaml`` sidecar (the
runtime-IK relay contract) next to the output URDF, injects exact
``<mimic>`` tags when the loop is a parallelogram four-bar (verified
numerically before writing), and keeps the config on
``assembly.loop_closures``.  A general loop is closed in-process with
:class:`~skrobot.kinematics.LoopClosureSolver`:

.. code-block:: python

   from skrobot.kinematics import LoopClosureSolver

   robot = assembly.build_robot_model()
   solver = LoopClosureSolver(robot, assembly.loop_closures)
   robot.crank_joint.joint_angle(0.4)
   solver.solve()  # dependent joints updated so the loop closes

See ``examples/module_assembly_four_bar.py`` for a complete
closed-loop assembly built from bare links.

Classes
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   skrobot.assembly.Port
   skrobot.assembly.RobotModule
   skrobot.assembly.Connection
   skrobot.assembly.RobotAssembly
   skrobot.kinematics.LoopClosureSolver

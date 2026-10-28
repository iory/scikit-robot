=======================
MuJoCo Robot Interface
=======================

**Example script**: ``examples/mujoco_robot_interface.py``

:class:`~skrobot.interfaces.MujocoRobotInterface` is a robot interface
(``ri``) whose robot is simulated in `MuJoCo <https://mujoco.org/>`_. It
takes the same calls as a real robot's interface, so a program written for
the real robot runs against it unchanged.

.. figure:: ../../image/mujoco_robot_interface.png
   :width: 100%
   :target: https://iory.github.io/scikit-robot-web/
   :alt: scikit-robot-web: Python on the left, the simulated Jedy on the right

   The same interface in the browser, in
   `scikit-robot-web <https://github.com/iory/scikit-robot-web>`_: Python
   on the left, the simulated Jedy on the right. Nothing to install.

Source Code
===========

https://github.com/iory/scikit-robot/blob/main/examples/mujoco_robot_interface.py

.. code-block:: bash

    pip install scikit-robot[mujoco] coacd
    python examples/mujoco_robot_interface.py

robot and ri
============

Programs for a real robot in scikit-robot (and in EusLisp before it) keep
two objects apart:

``robot``
    The robot model you plan with: forward and inverse kinematics,
    collision checks. Changing it moves nothing. It changes at once, and
    passes through tables.

``ri``
    The robot interface. It sends commands to the robot and reads back
    what the robot did. Motion takes time, and gravity and contact can
    stop the robot short of the command.

The loop is: plan with ``robot``, move with ``ri``, check with ``ri``:

.. code-block:: python

    import skrobot
    from skrobot.interfaces import MujocoRobotInterface

    robot = skrobot.models.Jedy(light=True)
    ri = MujocoRobotInterface(robot, planar_base=True, viewer=True)

    robot.inverse_kinematics(target, link_list=robot.rarm.link_list,
                             move_target=robot.rarm_end_coords)
    ri.angle_vector(robot.angle_vector(), 2.0)  # send, reach it in 2 s
    ri.wait_interpolation()                     # block until it is done
    robot.angle_vector(ri.angle_vector())       # where it actually is

``ri`` never writes into ``robot``: it works on its own copy, as the ROS
interface does. Only the line that creates ``ri`` differs from a program
for the real robot.

What ri offers
==============

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Call
     - Does
   * - ``ri.angle_vector(av, time)``
     - Send joint angles, reached in ``time`` seconds (raised to the joint
       velocity limits). Returns at once.
   * - ``ri.angle_vector_sequence(avs, times)``
     - Send several, passed through without stopping.
   * - ``ri.wait_interpolation()``
     - Wait until the motion sent last is over.
   * - ``ri.angle_vector()``
     - The robot's joint angles now.
   * - ``ri.go_pos(x, y, yaw)``, ``ri.go_velocity(...)``, ``ri.move_to(coords)``
     - Drive the base (``planar_base=True``), in its own frame for
       ``go_pos``.
   * - ``ri.odom``
     - Where the base is.
   * - ``ri.object_coords(name)``, ``ri.contacts(name)``
     - Where an object is (``objects=``, ``movable_objects=``), and what
       touches it.
   * - ``ri.get_image(camera)``, ``ri.get_depth(camera)``
     - Camera images, from cameras given with ``cameras=``.

``ri.object_coords`` and ``ri.contacts`` give the simulation's ground
truth, which a real robot does not have; there, they come from sensors such
as cameras.

Notes
=====

- Simulated time passes only inside ``ri`` calls (``wait_interpolation``,
  ``go_pos`` and the like); ``time.sleep`` does not move the robot.
- MuJoCo collides meshes by their convex hull, which fills the space
  between a gripper's fingers. ``converter_kwargs={'convex_decompose_links':
  [...]}`` splits those links into convex parts with CoACD; the parts are
  cached, so this happens once per mesh.
- ``Jedy(light=True)`` loads simplified meshes, which convert and simulate
  much faster than the full ones.

Related Documentation
=====================

- :doc:`../reference/interfaces` - Robot interfaces API
- :doc:`pybullet_interface` - The same interface on PyBullet

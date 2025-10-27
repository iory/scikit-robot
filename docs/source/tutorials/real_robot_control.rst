==================
Real Robot Control
==================

For controlling real robots and simulation interfaces, see:

- :doc:`../reference/interfaces` - Robot interfaces API reference

Scikit-robot provides interfaces for:

**ROS Interface**

Control ROS-enabled robots:

.. code-block:: python

   from skrobot.interfaces.ros import ROSRobotInterfaceBase

   class MyRobotInterface(ROSRobotInterfaceBase):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)

   ri = MyRobotInterface(robot)
   ri.angle_vector(target_av, time=3.0)
   ri.wait_interpolation()

**PyBullet Interface**

Simulate robots in PyBullet:

.. code-block:: python

   from skrobot.interfaces._pybullet import PybulletRobotInterface

   ri = PybulletRobotInterface(robot)
   ri.angle_vector(target_av, time=2.0)
   ri.wait_interpolation()

The same code works for both simulation and real robots, making it easy to develop and test in simulation before deploying to hardware.

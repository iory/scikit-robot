===========
Quick Start
===========

This tutorial will get you started with scikit-robot in just a few minutes.

Installation
============

First, install scikit-robot using pip:

.. code-block:: bash

   pip install scikit-robot

For full functionality including PyBullet interface and mesh optimization:

.. code-block:: bash

   pip install scikit-robot[all]

System Dependencies (Linux)
----------------------------

On Ubuntu/Debian, you may need some system libraries:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install libspatialindex-dev freeglut3-dev \
                        libsuitesparse-dev libblas-dev liblapack-dev

Loading Your First Robot
=========================

Let's load and visualize a PR2 robot:

.. code-block:: python

   from skrobot.models import PR2
   from skrobot.viewers import TrimeshSceneViewer

   # Load robot model
   robot = PR2()

   # Create viewer
   viewer = TrimeshSceneViewer(resolution=(800, 600))
   viewer.add(robot)
   viewer.show()

This will open a 3D window showing the PR2 robot. You can:

- **Rotate**: Left mouse button + drag
- **Pan**: Right mouse button + drag
- **Zoom**: Mouse wheel

Moving Robot Joints
====================

Let's move the robot's right arm:

.. code-block:: python

   import numpy as np

   # Set all right arm joints to specific angles
   robot.rarm.angle_vector([0.5, 0.3, -0.2, -1.0, 0.8, -0.5, 0.2])

   # Redraw the viewer to see the changes
   viewer.redraw()

   # Print current end-effector position
   print("End-effector position:", robot.rarm.end_coords.worldpos())

Understanding the Robot Structure
==================================

Explore the robot's structure:

.. code-block:: python

   # Print all link names
   print("Links:", [link.name for link in robot.link_list])

   # Print all joint names
   print("Joints:", [joint.name for joint in robot.joint_list])

   # Access specific links
   print("Right gripper:", robot.r_gripper_palm_link)

   # Get current joint angles
   print("Current joint angles:", robot.angle_vector())

Simple Inverse Kinematics
==========================

Move the end-effector to a target position:

.. code-block:: python

   # Get current end-effector coordinates
   target = robot.rarm.end_coords.copy_worldcoords()

   # Move 10cm forward (in X direction)
   target.translate([0.1, 0, 0])

   # Solve inverse kinematics
   success = robot.rarm.inverse_kinematics(target)

   if success:
       print("IK solved successfully!")
       print("New joint angles:", robot.rarm.angle_vector())
       viewer.redraw()
   else:
       print("IK failed - target may be unreachable")

Animating Robot Motion
=======================

Create a smooth animation:

.. code-block:: python

   import time

   # Define start and end configurations
   start_av = np.array([0, 0, 0, 0, 0, 0, 0])
   end_av = np.array([0.5, 0.3, -0.2, -1.0, 0.8, -0.5, 0.2])

   # Interpolate between start and end
   num_steps = 50
   for i in range(num_steps):
       t = i / (num_steps - 1)  # 0 to 1
       current_av = start_av * (1 - t) + end_av * t

       robot.rarm.angle_vector(current_av)
       viewer.redraw()
       time.sleep(0.05)

Working with Other Robot Models
================================

Scikit-robot includes several built-in robot models:

Fetch Robot
-----------

.. code-block:: python

   from skrobot.models import Fetch

   robot = Fetch()
   viewer = TrimeshSceneViewer()
   viewer.add(robot)
   viewer.show()

Kuka Robot
----------

.. code-block:: python

   from skrobot.models import Kuka

   robot = Kuka()
   viewer = TrimeshSceneViewer()
   viewer.add(robot)
   viewer.show()

Custom URDF Models
------------------

Load your own URDF file:

.. code-block:: python

   from skrobot.model import RobotModel

   robot = RobotModel.from_urdf_file('/path/to/robot.urdf')
   viewer = TrimeshSceneViewer()
   viewer.add(robot)
   viewer.show()

Adding Objects to the Scene
============================

Visualize obstacles and targets:

.. code-block:: python

   import trimesh

   # Create a box obstacle
   box = trimesh.creation.box([0.3, 0.3, 0.5])
   box.apply_translation([0.5, 0, 0.5])
   box.visual.face_colors = [255, 0, 0, 100]  # Semi-transparent red

   # Create a target sphere
   sphere = trimesh.creation.icosphere(radius=0.05)
   sphere.apply_translation([0.6, -0.2, 0.8])
   sphere.visual.face_colors = [0, 255, 0, 255]  # Green

   # Add to viewer
   viewer.add(box)
   viewer.add(sphere)
   viewer.redraw()

Using Coordinate Frames
========================

Understand and manipulate coordinate frames:

.. code-block:: python

   from skrobot.coordinates import Coordinates

   # Create a coordinate frame
   target_coords = Coordinates(pos=[0.6, -0.2, 0.8])

   # Rotate 45 degrees around Z-axis
   target_coords.rotate(np.pi / 4, 'z')

   # Translate relative to current orientation
   target_coords.translate([0.1, 0, 0], wrt='local')

   # Use as IK target
   robot.rarm.inverse_kinematics(target_coords)
   viewer.redraw()

Saving and Loading Robot States
================================

Save current configuration:

.. code-block:: python

   # Save current joint angles
   saved_av = robot.angle_vector().copy()

   # Move robot
   robot.rarm.angle_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
   viewer.redraw()

   # Restore saved configuration
   robot.angle_vector(saved_av)
   viewer.redraw()

Next Steps
==========

Now that you've learned the basics, explore more advanced topics:

- :doc:`inverse_kinematics` - Advanced IK techniques and constraints
- :doc:`motion_planning` - Collision-free trajectory planning
- :doc:`visualization` - Advanced visualization options
- :doc:`urdf_manipulation` - Modifying and optimizing URDF files
- :doc:`real_robot_control` - Controlling real robots via ROS

Complete Example
================

Here's a complete script that demonstrates the basics:

.. code-block:: python

   #!/usr/bin/env python
   import numpy as np
   import time
   from skrobot.models import PR2
   from skrobot.viewers import TrimeshSceneViewer
   from skrobot.coordinates import Coordinates
   import trimesh

   # Load robot
   robot = PR2()

   # Create viewer
   viewer = TrimeshSceneViewer(resolution=(1024, 768))
   viewer.add(robot)

   # Add target sphere
   target = trimesh.creation.icosphere(radius=0.05)
   target_pos = [0.6, -0.2, 0.8]
   target.apply_translation(target_pos)
   target.visual.face_colors = [0, 255, 0, 255]
   viewer.add(target)

   viewer.show()

   # Create target coordinates
   target_coords = Coordinates(pos=target_pos)

   # Reach for target
   print("Reaching for target...")
   success = robot.rarm.inverse_kinematics(target_coords)

   if success:
       print("Target reached!")
       viewer.redraw()
       time.sleep(2)

       # Wave motion
       print("Waving...")
       for i in range(3):
           robot.rarm.angle_vector(
               robot.rarm.angle_vector() + [0, 0, 0, 0, 0.3, 0, 0]
           )
           viewer.redraw()
           time.sleep(0.3)

           robot.rarm.angle_vector(
               robot.rarm.angle_vector() - [0, 0, 0, 0, 0.6, 0, 0]
           )
           viewer.redraw()
           time.sleep(0.3)

           robot.rarm.angle_vector(
               robot.rarm.angle_vector() + [0, 0, 0, 0, 0.3, 0, 0]
           )
           viewer.redraw()
           time.sleep(0.3)
   else:
       print("Could not reach target")

   print("Done! Close the viewer window to exit.")
   viewer.show()

Save this as ``quickstart_demo.py`` and run it:

.. code-block:: bash

   python quickstart_demo.py

Troubleshooting
===============

Viewer doesn't open
-------------------

If the viewer doesn't open, try:

.. code-block:: python

   # Use pyrender instead of trimesh
   from skrobot.viewers import PyrenderViewer

   viewer = PyrenderViewer()
   viewer.add(robot)
   viewer.show()

ImportError for trimesh or pyrender
------------------------------------

Install visualization dependencies:

.. code-block:: bash

   pip install trimesh scikit-robot-pyrender

Performance is slow
-------------------

For better performance:

1. Use optimized meshes (see :doc:`urdf_manipulation`)
2. Reduce viewer resolution
3. Use PyrenderViewer instead of TrimeshSceneViewer

.. code-block:: python

   viewer = TrimeshSceneViewer(resolution=(640, 480))

Getting Help
============

- Documentation: https://scikit-robot.readthedocs.io/
- GitHub Issues: https://github.com/iory/scikit-robot/issues
- Examples: https://github.com/iory/scikit-robot/tree/main/examples

Welcome to scikit-robot!

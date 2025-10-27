===========================
Robot Models Visualization
===========================

**Example script**: ``examples/robot_models.py``

This example demonstrates how to load and visualize multiple built-in robot models simultaneously.

What This Example Shows
========================

- Loading multiple pre-configured robot models
- Arranging robots in a grid layout
- Creating ground planes for visual context
- Setting up camera angles for optimal viewing

Running the Example
===================

Basic usage:

.. code-block:: bash

   python examples/robot_models.py

With different viewers:

.. code-block:: bash

   # Use Trimesh viewer (faster, simpler)
   python examples/robot_models.py --viewer trimesh

   # Use Pyrender viewer (smoother rendering)
   python examples/robot_models.py --viewer pyrender

Non-interactive mode (useful for CI/testing):

.. code-block:: bash

   python examples/robot_models.py --no-interactive

Code Walkthrough
================

Built-in Robot Models
---------------------

The example showcases five built-in robot models:

.. code-block:: python

   robots = [
       skrobot.models.Kuka(),      # Industrial manipulator
       skrobot.models.Fetch(),     # Mobile manipulation robot
       skrobot.models.Nextage(),   # Humanoid torso robot
       skrobot.models.PR2(),       # Research platform robot
       skrobot.models.Panda(),     # Franka Emika Panda arm
   ]

Grid Layout Algorithm
---------------------

The ``_get_tile_shape`` function calculates optimal grid dimensions:

.. code-block:: python

   def _get_tile_shape(num, hw_ratio=1):
       """Calculate grid layout for given number of items"""
       r_num = int(round(np.sqrt(num / hw_ratio)))
       c_num = 0
       while r_num * c_num < num:
           c_num += 1
       while (r_num - 1) * c_num >= num:
           r_num -= 1
       return r_num, c_num

This ensures robots are arranged efficiently, regardless of how many models you want to display.

Adding Ground Planes
--------------------

Each robot gets its own ground plane for visual separation:

.. code-block:: python

   plane = skrobot.model.Box(extents=(row - 0.01, col - 0.01, 0.01))
   plane.translate((row * i, col * j, -0.01))
   viewer.add(plane)

The slight gap (0.01) between planes creates visual separation.

Positioning Robots
------------------

Robots are positioned based on their grid location:

.. code-block:: python

   robot.translate((row * i, col * j, 0))
   viewer.add(robot)

Camera Setup
------------

The camera angle is set for optimal viewing:

.. code-block:: python

   viewer.set_camera(angles=[np.deg2rad(30), 0, 0])

This provides a 30-degree elevated view of the scene.

Key Concepts
============

**RobotModel Loading**

All built-in models are loaded with default configurations:

- URDF files are automatically downloaded and cached
- Joint angles start in default pose
- Visual meshes are loaded for rendering

**Viewer Flexibility**

Two viewer options are provided:

- **TrimeshSceneViewer**: Lightweight, good for development
- **PyrenderViewer**: OpenGL-based, smoother rendering

**Interactive vs Non-Interactive**

In interactive mode, the window stays open:

.. code-block:: python

   while viewer.is_active:
       time.sleep(0.1)
       viewer.redraw()

In non-interactive mode (``--no-interactive``), the script exits immediately after display.

Customization Ideas
===================

Add Your Own Robots
-------------------

.. code-block:: python

   from skrobot.model import RobotModel

   robots = [
       skrobot.models.PR2(),
       skrobot.models.Fetch(),
       RobotModel.from_urdf_file('/path/to/custom.urdf'),
   ]

Change Grid Spacing
-------------------

.. code-block:: python

   row, col = 3, 3  # More space between robots

Animate Robots
--------------

.. code-block:: python

   import time

   for robot in robots:
       for _ in range(100):
           # Random joint motion
           av = robot.angle_vector()
           av += np.random.randn(len(av)) * 0.01
           robot.angle_vector(av)
           viewer.redraw()
           time.sleep(0.05)

Add Coordinate Frames
---------------------

.. code-block:: python

   for robot in robots:
       axis = skrobot.model.Axis(
           axis_radius=0.01,
           axis_length=0.2,
           pos=robot.worldpos(),
           rot=robot.worldrot()
       )
       viewer.add(axis)

Expected Output
===============

You should see a window displaying all five robots arranged in a grid, with each robot standing on its own ground plane with proper spacing.

Troubleshooting
===============

Models Don't Load
-----------------

If models fail to load:

1. Check internet connection (first run downloads models)
2. Verify cache directory: ``~/.skrobot/``
3. Try clearing cache and re-running

Viewer Crashes
--------------

If viewer crashes:

1. Try different viewer: ``--viewer pyrender`` or ``--viewer trimesh``
2. Reduce resolution in code:

.. code-block:: python

   viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(320, 240))

Performance Issues
------------------

For better performance:

1. Use TrimeshSceneViewer instead of PyrenderViewer
2. Reduce number of robots
3. Disable smooth shading

Related Examples
================

- :doc:`pr2_inverse_kinematics` - Detailed IK demonstrations
- :doc:`trimesh_scene_viewer` - Advanced viewer features
- :doc:`pybullet_interface` - Simulating robots in PyBullet

Source Code
===========

Full source: https://github.com/iory/scikit-robot/blob/main/examples/robot_models.py

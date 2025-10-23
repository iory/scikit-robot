Viewers
=======


Examples
========

You can easily try out visualization programs using the examples found in `scikit-robot/examples <https://github.com/iory/scikit-robot/tree/main/examples>`_

.. code-block:: bash

    python robot_models.py --viewer trimesh


.. figure:: ../../image/robot_models.jpg
    :scale: 100%
    :align: center



CommandLine Tools
=================


You can easily visualize a URDF by providing it as an argument to the visualize-urdf command.


.. code-block:: bash

    visualize-urdf ~/.skrobot/pr2_description/pr2.urdf


.. figure:: ../../image/pr2.png
    :scale: 20%
    :align: center


Viewer classes
==============

TrimeshSceneViewer
------------------

**Description:**
  The ``TrimeshSceneViewer`` is an extension of the ``trimesh.viewer.SceneViewer`` tailored for visualizing 3D scenes using the Trimesh library. It is specifically designed for 3D triangle meshes visualization and manipulation in robotic applications.

**Key Functionalities:**

- **Initialization and Configuration:**
  Initializes with options for screen resolution and an update interval. It sets up a scene using Trimesh to manage various geometrical entities.

- **Rendering Control:**
  Manages redraws upon user interactions such as mouse clicks, drags, scrolls, and key presses. It also handles window resizing events to ensure the scene is accurately rendered.

- **Scene Management:**
  Supports dynamic addition and deletion of geometrical entities. It allows management of links and their associated meshes, enabling real-time updates based on robotic movements.

- **Camera Management:**
  Facilitates camera positioning and orientation, allowing for customizable views based on specified angles and transformations reflective of the robotic link configurations.

PyrenderViewer
--------------

**Description:**
  The ``PyrenderViewer`` utilizes the Pyrender library for advanced 3D rendering, ideal for creating realistic visual simulations. This viewer is implemented as a Singleton to ensure only one instance exists throughout the program. It's particularly suited for complex rendering tasks in robotics, including detailed lighting, shading effects, and collision visualization.

**Key Functionalities:**

- **Initialization and Configuration:**
  The viewer is initialized with specified resolution, update interval, and rendering flags. Key parameters include:

  - ``resolution``: Window size (default: ``(640, 480)``)
  - ``update_interval``: Update frequency in seconds (default: ``1.0``)
  - ``enable_collision_toggle``: Enable collision/visual mesh switching (default: ``True``)
  - ``title``: Window title (default: ``'scikit-robot PyrenderViewer'``)

- **Rendering Control:**
  Handles real-time scene updates triggered by user interactions. The viewer automatically manages OpenGL compatibility with fallback support from OpenGL 4.1 → 4.0 → 3.3, ensuring robust operation across different systems including WSL2.

- **Scene Management:**
  Supports dynamic addition and removal of visual and collision meshes linked to robotic models. The viewer maintains real-time synchronization with robot configurations through the ``redraw()`` method.

- **Camera Management:**
  Offers detailed camera setup options through the ``set_camera()`` method:

  - Angle-based positioning with Euler angles
  - Distance and center point configuration
  - Field of view (FOV) adjustment
  - Direct Coordinates object support for precise camera placement

- **Collision/Visual Mesh Toggle:**
  When ``enable_collision_toggle=True``, press the ``v`` key to switch between:

  - **Visual meshes**: Default appearance meshes for rendering (left in figure below)
  - **Collision meshes**: Simplified meshes used for collision detection (displayed in orange/transparent, right in figure below)

  .. figure:: ../_static/visual-collision-comparison.jpg
     :width: 100%
     :align: center
     :alt: Visual mesh (left) vs Collision mesh (right) comparison

     **Visual mesh (left) vs Collision mesh (right).** The visual mesh shows the detailed appearance of the robot with textured wheels. The collision mesh on the right uses simplified cylinder representations for the wheels, which are computationally more efficient for collision detection algorithms.

- **360-Degree Image Capture:**
  The ``capture_360_images()`` method enables automated scene capture from multiple angles:

  - Configurable number of frames and camera elevation
  - Automatic GIF animation generation
  - Transparent background support
  - Custom lighting configuration options


**Keyboard Controls:**

The PyrenderViewer provides extensive keyboard controls for interactive manipulation:

.. list-table:: Keyboard Controls
   :header-rows: 1
   :widths: 10 90

   * - Key
     - Function
   * - ``a``
     - Toggle rotational animation mode
   * - ``c``
     - Toggle backface culling
   * - ``f``
     - Toggle fullscreen mode
   * - ``h``
     - Toggle shadow rendering (may impact performance)
   * - ``i``
     - Cycle through axis display modes (none → world → mesh → all)
   * - ``j``
     - **Toggle joint axes display** (shows/hides joint positions and axes for all robots)
   * - ``l``
     - Cycle lighting modes (scene → Raymond → direct)
   * - ``m``
     - Toggle face normal visualization
   * - ``n``
     - Toggle vertex normal visualization
   * - ``o``
     - Toggle orthographic camera mode
   * - ``q``
     - Quit the viewer
   * - ``r``
     - Start/stop GIF recording (opens file dialog on stop)
   * - ``s``
     - Save current view as image (opens file dialog)
   * - ``v``
     - **Toggle between visual and collision meshes** (if enabled)
   * - ``w``
     - Cycle wireframe modes
   * - ``z``
     - Reset camera to default view

**Mouse Controls:**

- **Left-click + drag**: Rotate camera around scene center
- **Ctrl + Left-click + drag**: Rotate camera around viewing axis
- **Shift + Left-click + drag** or **Middle-click + drag**: Pan camera
- **Right-click + drag** or **Scroll wheel**: Zoom in/out

**Example Usage:**

Basic viewer initialization and robot display:

.. code-block:: python

    from skrobot.viewers import PyrenderViewer
    from skrobot.models import PR2

    # Create viewer instance (Singleton pattern ensures only one instance)
    viewer = PyrenderViewer(resolution=(800, 600), update_interval=1.0/30)
    
    # Load and add robot model
    robot = PR2()
    viewer.add(robot)
    
    # Show the viewer window
    viewer.show()
    
    # Update robot pose and redraw
    robot.reset_manip_pose()
    viewer.redraw()

Collision/Visual mesh toggle example:

.. code-block:: python

    # Enable collision toggle functionality
    viewer = PyrenderViewer(enable_collision_toggle=True)
    
    # Add robot to viewer
    viewer.add(robot)
    viewer.show()
    
    # Press 'v' key in the viewer to toggle between visual and collision meshes
    # Collision meshes will appear in orange/transparent color
    
    # The visual mesh displays the full detailed geometry with textures
    # while collision mesh shows simplified shapes (e.g., cylinders for wheels)
    # optimized for physics calculations

360-degree image capture example:

.. code-block:: python

    # Capture 360-degree rotation images
    viewer.capture_360_images(
        output_dir="./robot_360",
        num_frames=36,  # One image every 10 degrees
        camera_elevation=45,  # Camera elevation angle
        create_gif=True,  # Generate animated GIF
        gif_duration=100,  # 100ms between frames
        transparent_background=True  # Render with transparent background
    )

.. caution::

  To speed up the rendering cycle in **TrimeshSceneViewer** and **PyrenderViewer**, adjust the ``update_interval`` to the reciprocal of the desired frequency. For example, to achieve updates at 30 Hz, set the ``update_interval`` to 1/30. This change will increase the frequency at which the ``redraw()`` function is called, making the rendering process faster.

  Example usage:

  .. code-block:: python

    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480), update_interval=1.0/30)   # Set update interval for 30 Hz
    viewer = skrobot.viewers.PyrenderViewer(resolution=(640, 480), update_interval=1.0/30)      # Same for PyrenderViewer


Color Management
----------------

**Changing Colors:**

To enhance the visibility and distinction of different components in a robot model, users can change the colors of individual links or the entire robot. This can be done using the ``set_color`` method, which applies a specified RGBA color to the link. The ``reset_color`` method restores the original color of the link, allowing for easy toggling between custom and default visualizations.


.. code-block:: python

    import time
    from skrobot.viewers import TrimeshSceneViewer
    from skrobot.models import PR2
    import numpy as np

    viewer = TrimeshSceneViewer()
    robot_model = PR2()
    viewer.add(robot_model)
    viewer.show()

    # Setting the color to red with some transparency
    color = [255, 0, 0, 200]
    for link in robot_model.find_link_path(robot_model.rarm_root_link, robot_model.r_gripper_l_finger_tip_link) + robot_model.find_link_path(robot_model.rarm_root_link, robot_model.r_gripper_r_finger_tip_link):
        link.set_color(color)


.. figure:: ../../image/change-link-color.jpg
    :scale: 100%
    :align: center


.. code-block:: python

    # Resetting the color to default
    for link in robot_model.find_link_path(robot_model.rarm_root_link, robot_model.r_gripper_l_finger_tip_link) + robot_model.find_link_path(robot_model.rarm_root_link, robot_model.r_gripper_r_finger_tip_link):
        link.reset_color()


.. figure:: ../../image/reset-link-color.jpg
    :scale: 100%
    :align: center

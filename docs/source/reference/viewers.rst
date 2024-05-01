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
  The ``PyrenderViewer`` utilizes the Pyrender library for advanced 3D rendering, ideal for creating realistic visual simulations. This viewer is particularly suited for complex rendering tasks in robotics, including detailed lighting and shading effects.

**Key Functionalities:**

- **Initialization and Configuration:**
  The viewer is initialized with specified resolution and rendering flags, creating a scene managed by Pyrender. It supports high-quality rendering features like raymond lighting.

- **Rendering Control:**
  Handles real-time scene updates triggered by user interactions such as mouse events and keyboard inputs, ensuring the scene remains interactive and up-to-date.

- **Scene Management:**
  Similar to ``TrimeshSceneViewer``, it allows for the addition and removal of visual meshes linked to robotic models, supporting dynamic updates to the scene as robotic configurations change.

- **Camera Management:**
  Offers detailed camera setup options, including angle adjustments, distance settings, center positioning, and field of view configuration, providing flexibility in viewing angles for complex scenes.

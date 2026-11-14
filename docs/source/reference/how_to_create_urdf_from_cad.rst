How to Create URDF from CAD Software
====================================

Overview
--------

.. figure:: ../../image/urdf-from-solidworks.png
    :scale: 30%
    :align: center

This document explains how to create a URDF file from SolidWorks models by using:

1. SolidWorks to URDF Exporter
2. scikit-robot's ``convert-urdf-mesh`` command

- **SolidWorks to URDF Exporter**
  This exporter helps convert SolidWorks assemblies into a format (3dxml) which is more easily processed to produce URDF-compatible mesh files.

- **scikit-robot ``convert-urdf-mesh``**
  A tool within the `scikit-robot <https://github.com/iory/scikit-robot>`_ library that can convert 3D model files (like ``.3dxml``, ``.obj``, ``.stl``, etc.) into a URDF or mesh files (e.g., ``.dae``) suitable for ROS.

Recommended: sw2robot (solidworks_urdf_exporter2)
-------------------------------------------------

If you are starting a new project, we recommend using
`sw2robot (solidworks_urdf_exporter2) <https://github.com/jsk-ros-pkg/solidworks_urdf_exporter2>`_
together with scikit-robot. It is the modernized successor of the classic
SolidWorks URDF Exporter and pairs well with ``convert-urdf-mesh``.

Compared with the classic exporter, sw2robot:

- **Infers the kinematic tree automatically** by reading the assembly's existing
  mates (constraints), instead of asking you to lay out the link hierarchy and
  set each joint origin/axis by hand inside SolidWorks.
- **Ships a cross-platform browser editor** so that the *edit / build / export*
  steps run natively on Windows, macOS, and Linux (only the *extract* step, which
  drives SolidWorks over COM, needs Windows + SolidWorks). You can re-root the
  tree, change joint types, edit frames, set materials/densities, and export a
  ROS / robot-compiler package.
- **Works on any URDF**, so you can also open a URDF produced by the classic
  add-in and clean it up in the editor.

**Exports URDF + meshes directly (no ``convert-urdf-mesh`` step needed).**
sw2robot writes the URDF together with its mesh files for you, so with sw2robot
you typically do *not* need to post-process the output with
``convert-urdf-mesh``. In particular it can:

- **Emit visual meshes directly as ``.glb`` (or ``.dae`` / ``.stl``)**, keeping
  per-part materials/colors — no separate 3dxml→dae conversion pass.
- **Generate collision geometry for you** instead of reusing the (heavy, concave)
  visual mesh. You can choose per export:

  - ``copy`` — reuse the visual mesh as-is,
  - ``hull`` — a single convex hull per link,
  - ``coacd`` — approximate **convex decomposition** into a set of convex parts
    (great for physics engines; ``balanced`` / ``fine`` quality presets),
  - ``primitive`` / ``box`` / ``cylinder`` / ``sphere`` — fit a native URDF
    primitive shape per link (no mesh file at all; ``primitive`` auto-picks the
    best-fitting shape).

**Download from the release page.** Prebuilt editors for Windows, macOS
(Apple Silicon), and Linux (x64) are available on the releases page — no
Python or SolidWorks required just to edit:

  `sw2robot releases (latest) <https://github.com/jsk-ros-pkg/solidworks_urdf_exporter2/releases/latest>`__

Click the binary matching your OS on that page to grab the latest prebuilt
editor, then pick the visual mesh format (``.glb`` recommended) and a collision
mode (``coacd`` or ``primitive`` for physics) right in the export dialog. See the
`project README <https://github.com/jsk-ros-pkg/solidworks_urdf_exporter2#readme>`_
for the full workflow.

Tutorial: a quadruped robot from SolidWorks to MuJoCo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sw2robot documentation site has a step-by-step tutorial that builds a
small **4-legged robot** (9 links, 8 revolute joints) from scratch in
SolidWorks, turns it into a URDF, and then makes it walk in MuJoCo.

- `Tutorial: a quadruped from scratch <https://jsk-ros-pkg.github.io/solidworks_urdf_exporter2/tutorial/>`_
- `Make it walk in MuJoCo <https://jsk-ros-pkg.github.io/solidworks_urdf_exporter2/walk/>`_

The overall flow looks like this:

.. code-block:: text

   SolidWorks assembly  --(extract)-->  URDF + meshes  --(edit in browser)-->  ROS 2 / MuJoCo package
   (parts + mates)                      (link tree from mates)                 (--ros2 / --mujoco)

**1. Model the parts and assemble them with mates.**
Three parts (body, thigh, shin) are assembled into four legs. Each joint is
just two ordinary SolidWorks mates — *concentric* (the joint axis) plus
*coincident* (seats the part) — and sw2robot turns that into a ``revolute``
joint. The first inserted part (the body) becomes ``base_link``.

.. figure:: https://jsk-ros-pkg.github.io/solidworks_urdf_exporter2/images/sw_assembly.png
    :width: 480px
    :align: center
    :alt: The tutorial quadruped assembled in SolidWorks

    The finished quadruped assembly in SolidWorks.

.. only:: html

   .. figure:: https://jsk-ros-pkg.github.io/solidworks_urdf_exporter2/images/build_assembly.gif
       :width: 480px
       :align: center
       :alt: Each leg part goes in, then the concentric and coincident mates attach it

       Each leg part is inserted, put on the axle by a concentric mate, and seated by a coincident mate.

**2. Extract the URDF and clean it up in the editor.**
Open the assembly from the sw2robot editor (or run ``sw2urdf``). The link tree
and joint axes are inferred from the mates. In the editor you rename links and
joints, set joint limits, and flip axes where needed (e.g. the right-hand knees).

.. figure:: https://jsk-ros-pkg.github.io/solidworks_urdf_exporter2/images/editor_en.jpg
    :width: 600px
    :align: center
    :alt: The same robot in the sw2robot browser editor

    The same robot opened in the sw2robot browser editor.

**3. Export and use it.**
Export a ROS 2 package (``--ros2``) or an MJCF for MuJoCo (``--mujoco``). The
exported URDF can be loaded directly with scikit-robot:

.. code-block:: python

   import skrobot

   robot = skrobot.models.urdf.RobotModelFromURDF(urdf_file='quadruped.urdf')
   viewer = skrobot.viewers.TrimeshSceneViewer()
   viewer.add(robot)
   viewer.show()

**4. Make it walk in MuJoCo.**
The walk page takes the same quadruped into MuJoCo and trains a walking policy
with reinforcement learning (PPO) on a CPU in under an hour.

.. only:: html

   .. figure:: https://jsk-ros-pkg.github.io/solidworks_urdf_exporter2/images/walk_mujoco.gif
       :width: 600px
       :align: center
       :alt: The tutorial quadruped trotting in MuJoCo

       The tutorial quadruped trotting in MuJoCo
       (from `Make it walk in MuJoCo <https://jsk-ros-pkg.github.io/solidworks_urdf_exporter2/walk/>`_).

.. note::

   The ``convert-urdf-mesh`` workflow described below still applies when you are
   using the **classic** SolidWorks URDF Exporter (which produces ``.3dxml``),
   or when you have existing meshes to reconvert. If you use sw2robot, prefer its
   built-in ``.glb`` + collision export instead.

Coordinate Systems
------------------

- **.3dxml export**: uses the overall assembly coordinate system from SolidWorks.
- **.stl export**: uses each part's or link's local coordinate system.

``convert-urdf-mesh`` provides an option for adjusting coordinates:

- ``--force-zero-origin``: Forces the origin to be the link's local coordinate in the converted mesh files (e.g., ``.dae``).
  - Useful when your source is in the assembly coordinate system (like ``.3dxml``).
  - Highly recommended if you plan to use these meshes in **MuJoCo**.

.. note::

   **glTF (.glb) up axis.** The glTF specification says +Y is up, but the
   ROS tools do not convert: RViz and Gazebo use a ``.glb`` file's vertex
   coordinates as they are, in the URDF's frame, exactly as they do for
   ``.stl`` and ``.dae``. scikit-robot follows the same rule, both when it
   loads a URDF and when ``convert-urdf-mesh --format glb`` writes one, so a
   robot looks the same in scikit-robot, RViz and Gazebo.

   The catch is a ``.glb`` made by a spec-following tool. Blender, for
   example, rotates its Z-up scene to Y-up when it exports, so its ``.glb``
   lies on its side in a URDF (RViz, Gazebo and scikit-robot alike). Export
   such a file with the tool's Y-up conversion turned off (Blender: uncheck
   *+Y Up* under *Transform*). Conversely, a ``.glb`` written by
   scikit-robot shows up rotated by 90 degrees about X when opened in
   Blender or a three.js viewer; the file is fine for ROS.

Installation
------------

1. **Install scikit-robot**

.. code-block:: bash

   pip install scikit-robot

or clone directly from GitHub if you need the latest updates:

.. code-block:: bash

   git clone https://github.com/iory/scikit-robot.git
   cd scikit-robot
   pip install -e .

2. **Install a SolidWorks URDF Exporter**

- **Recommended: sw2robot (solidworks_urdf_exporter2).** Download the prebuilt
  editor for your OS from the release page — no separate installation into
  SolidWorks is required to edit / build / export:

  `sw2robot releases (latest) <https://github.com/jsk-ros-pkg/solidworks_urdf_exporter2/releases/latest>`__

- **Classic exporter (alternative).** Obtain the plugin from the following link:

  `SolidWorks URDF Exporter Plugin <https://drive.google.com/file/d/1iJ1jx8uAQsnmTtEBv4zEJnCgSbWJ3Ho2/view?usp=drive_link>`_

  Follow the official instructions to install it into your SolidWorks environment.

  `Installation Instructions <https://github.com/ros/solidworks_urdf_exporter>`_

Workflow
--------

1. **Export from SolidWorks**

   - In SolidWorks, open your assembly.
   - Use the "SolidWorks to URDF Exporter" plugin to generate a ``.3dxml`` file.

2. **Convert to DAE (or STL) and Generate URDF**

   - Run the scikit-robot command to convert ``.3dxml`` (or other mesh formats) to ``.dae`` and generate a URDF automatically.
   - Example usage:

   .. code-block:: bash

      convert-urdf-mesh <URDF_PATH> --output <OUTPUT_URDF_PATH>

   - This command outputs:
     - A set of mesh files (e.g., ``.dae``)
     - A URDF file referencing those meshes

3. **Verify URDF in ROS**

   - Copy the generated URDF and mesh files into your ROS package.
   - Test in Rviz or another ROS-compatible viewer:

   .. code-block:: bash

      roslaunch urdf_tutorial display.launch model:=path/to/generated.urdf

   - Confirm the model loads and displays properly.

4. **Usage with MuJoCo**

   - MuJoCo typically requires meshes to be centered at their local origin.
   - Therefore, always use the ``--force-zero-origin`` option when converting to ensure proper alignment.

   .. code-block:: bash

      convert-urdf-mesh <URDF_PATH> --output <OUTPUT_URDF_PATH> --force-zero-origin

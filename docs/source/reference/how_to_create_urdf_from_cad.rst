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

Coordinate Systems
------------------

- **.3dxml export**: uses the overall assembly coordinate system from SolidWorks.
- **.stl export**: uses each part's or link's local coordinate system.

``convert-urdf-mesh`` provides an option for adjusting coordinates:

- ``--force-zero-origin``: Forces the origin to be the link's local coordinate in the converted mesh files (e.g., ``.dae``).
  - Useful when your source is in the assembly coordinate system (like ``.3dxml``).
  - Highly recommended if you plan to use these meshes in **MuJoCo**.

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

2. **Install SolidWorks URDF Exporter**

- Obtain the plugin from the following link:

  `SolidWorks URDF Exporter Plugin <https://drive.google.com/file/d/1iJ1jx8uAQsnmTtEBv4zEJnCgSbWJ3Ho2/view?usp=drive_link>`_

- Follow the official instructions to install it into your SolidWorks environment.

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

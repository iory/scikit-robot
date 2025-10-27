==================
URDF Manipulation
==================

For URDF manipulation tools and techniques, see:

- :doc:`../reference/how_to_create_urdf_from_cad` - Creating URDF from CAD software
- :doc:`../cli` - Command-line URDF tools

Scikit-robot provides comprehensive URDF manipulation tools:

**modularize-urdf**: Convert monolithic URDF to reusable xacro macros

.. code-block:: bash

   skr modularize-urdf robot.urdf --output robot_module.xacro

**change-urdf-root**: Dynamically reconfigure kinematic hierarchy

.. code-block:: bash

   skr change-urdf-root robot.urdf new_root_link output.urdf

**convert-urdf-mesh**: Optimize 3D meshes

.. code-block:: bash

   skr convert-urdf-mesh robot.urdf --output optimized.urdf --quality 0.5

**visualize-urdf**: Interactive 3D preview

.. code-block:: bash

   skr visualize-urdf robot.urdf --viewer trimesh

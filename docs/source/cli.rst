Command Line Interface (CLI)
============================

Scikit-robot provides a unified command-line interface through the ``skr`` command, which consolidates all robot-related tools into a single entry point. This design makes it easy to discover and use various robot manipulation tools.

Installation
------------

The CLI tools are automatically installed when you install scikit-robot:

.. code-block:: bash

   pip install scikit-robot

Getting Started
---------------

To see all available commands:

.. code-block:: bash

   skr --help

This will display a list of all available subcommands along with their descriptions.

Available Commands
------------------

visualize-urdf
~~~~~~~~~~~~~~

Visualize URDF robot models in an interactive 3D viewer.

.. code-block:: bash

   # Basic usage
   skr visualize-urdf robot.urdf
   
   # With specific viewer
   skr visualize-urdf robot.urdf --viewer trimesh
   skr visualize-urdf robot.urdf --viewer pyrender

convert-urdf-mesh
~~~~~~~~~~~~~~~~~

Convert mesh files referenced in URDF to different formats or simplify them.

.. code-block:: bash

   # Convert meshes
   skr convert-urdf-mesh robot.urdf --output converted_robot.urdf
   
   # Simplify meshes with voxel size
   skr convert-urdf-mesh robot.urdf --voxel-size 0.001
   
   # Convert to STL format
   skr convert-urdf-mesh robot.urdf --output robot_stl.urdf -f stl

change-urdf-root
~~~~~~~~~~~~~~~~

Change the root link of a URDF file to a different link.

.. code-block:: bash

   # Change root link
   skr change-urdf-root robot.urdf new_root_link output.urdf
   
   # List available links
   skr change-urdf-root robot.urdf --list
   
   # Verbose output
   skr change-urdf-root robot.urdf new_root output.urdf --verbose

modularize-urdf
~~~~~~~~~~~~~~~

Modularize URDF files by breaking them into reusable components.

.. code-block:: bash

   skr modularize-urdf robot.urdf --output modular_robot.urdf

urdf-hash
~~~~~~~~~

Calculate a hash value for URDF files to track changes and versions.

.. code-block:: bash

   skr urdf-hash robot.urdf

visualize-mesh
~~~~~~~~~~~~~~

Visualize individual mesh files in 3D.

.. code-block:: bash

   skr visualize-mesh mesh_file.stl
   skr visualize-mesh mesh_file.obj

convert-wheel-collision
~~~~~~~~~~~~~~~~~~~~~~~

Convert wheel collision models in URDF files.

.. code-block:: bash

   skr convert-wheel-collision robot.urdf --output converted.urdf

generate-robot-class
~~~~~~~~~~~~~~~~~~~~

Generate Python robot class from URDF geometry. This tool automatically detects
kinematic chains (arms, legs, head, torso) and generates a Python class with
appropriate properties and end-effector coordinates.

No LLM or API keys required - uses only URDF structure and geometry.

.. code-block:: bash

   # Generate robot class and print to stdout
   skr generate-robot-class robot.urdf

   # Save to file
   skr generate-robot-class robot.urdf --output MyRobot.py

   # Specify custom class name
   skr generate-robot-class robot.urdf --class-name MyCustomRobot --output MyRobot.py

   # Show detected groups without generating code
   skr generate-robot-class robot.urdf --show-groups

Backward Compatibility
----------------------

For backward compatibility, all original individual commands are still available:

.. code-block:: bash

   # These commands work the same as their skr equivalents
   visualize-urdf robot.urdf
   convert-urdf-mesh robot.urdf --output converted.urdf
   change-urdf-root robot.urdf new_root output.urdf
   modularize-urdf robot.urdf --output modular.urdf
   urdf-hash robot.urdf
   visualize-mesh mesh_file.stl
   convert-wheel-collision robot.urdf --output converted.urdf
   generate-robot-class robot.urdf --output MyRobot.py

Getting Help
------------

Each subcommand provides its own help information:

.. code-block:: bash

   # General help
   skr --help
   
   # Help for specific commands
   skr visualize-urdf --help
   skr convert-urdf-mesh --help
   skr change-urdf-root --help

Examples
--------

Here are some common usage examples:

Visualizing a Robot Model
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Download a sample robot model and visualize it
   skr visualize-urdf ~/.skrobot/pr2_description/pr2.urdf --viewer trimesh

Converting Mesh Formats
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Convert all meshes in a URDF to STL format
   skr convert-urdf-mesh robot.urdf --output robot_stl.urdf -f stl
   
   # Simplify meshes by decimation
   skr convert-urdf-mesh robot.urdf -d 0.98 --output simplified.urdf

Changing Robot Structure
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # First, see what links are available
   skr change-urdf-root robot.urdf --list
   
   # Then change the root to a specific link
   skr change-urdf-root robot.urdf base_link new_robot.urdf

Architecture
------------

The CLI system is designed to be extensible. New commands can be added by:

1. Creating a new Python module in ``skrobot/apps/`` with a ``main()`` function
2. The CLI will automatically discover and register the new command
3. Command names are derived from the module filename (underscores become hyphens)

This modular design makes it easy to add new functionality while maintaining a consistent interface.
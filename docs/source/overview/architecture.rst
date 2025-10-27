============
Architecture
============

This page explains the core architecture and main components of scikit-robot.

System Architecture Overview
=============================

Scikit-robot is built around three core principles: **modularity**, **efficiency**, and **accessibility**.

The system consists of three main layers:

1. **User Python Code**: Write simple Python scripts using scikit-robot API
2. **Scikit-Robot Framework**: Core components including URDF tools, RobotModel, Viewers, Coordinates management, Hash system, and Planners
3. **External Systems**: Integration with PyBullet, ROS, Blender, and real robots

Core Components
===============

1. Coordinates and CascadedCoords
----------------------------------

The ``Coordinates`` class is the foundation of scikit-robot's geometric computation:

**Coordinates Class**

- Represents a 3D position and orientation as a unified entity
- Provides transformation methods: ``translate``, ``rotate``, ``transform_vector``
- Supports multiple rotation representations: rotation matrices, quaternions, RPY angles
- Includes inverse transformation, composition, and interpolation

**CascadedCoords Class**

- Extends ``Coordinates`` with hierarchical parent-child relationships
- Implements lazy evaluation: transformations computed only when needed
- Incremental updates: only affected branches recalculated on changes
- Memory efficient: shared coordinate objects within the hierarchy

Example usage:

.. code-block:: python

   from skrobot.coordinates import Coordinates, CascadedCoords

   # Create coordinate frames
   world = Coordinates()
   base = CascadedCoords(parent=world, pos=[1, 0, 0])
   end_effector = CascadedCoords(parent=base, pos=[0, 0, 0.5])

   # Move base -> end_effector automatically updates
   base.translate([0.1, 0, 0])
   print(end_effector.worldpos())  # Reflects parent's movement

2. RobotModel
-------------

The ``RobotModel`` class provides a unified interface for robot manipulation:

**Key Features**

- Automatic generation from URDF files
- Forward and inverse kinematics (IK/FK)
- Jacobian computation
- Forward and inverse dynamics
- Self-collision detection
- Joint limit handling

**Flexible IK Constraints**

The inverse kinematics solver supports various geometric constraints:

- Position-only (move end-effector to target position)
- Orientation-only (achieve target orientation)
- Full 6-DOF (position + orientation)
- Custom constraints (e.g., keep tool vertical)

Example usage:

.. code-block:: python

   from skrobot.models import PR2

   robot = PR2()

   # Move right arm to target position
   target_coords = robot.rarm.end_coords.copy_worldcoords()
   target_coords.translate([0.1, -0.1, 0])

   robot.rarm.inverse_kinematics(
       target_coords,
       rotation_axis=True  # Full 6-DOF constraint
   )

3. URDF Toolchain
-----------------

Scikit-robot provides comprehensive command-line tools for URDF manipulation:

**modularize-urdf**

Converts monolithic URDF files into reusable xacro macros:

- Namespace management (prefix addition)
- Parameter extraction
- Connection point definition
- Macro generation

.. code-block:: bash

   skr modularize-urdf robot.urdf \
       --output robot_modular.xacro \
       --prefix myrobot

**change-urdf-root**

Reconfigures URDF hierarchical structures:

- Path discovery using depth-first search
- Coordinate inversion with joint reversal
- Topology reconstruction
- Preserves kinematic equivalence

.. code-block:: bash

   skr change-urdf-root robot.urdf \
       new_root_link \
       output.urdf

**convert-urdf-mesh**

Optimizes 3D meshes:

- Texture-preserving decimation
- Format conversion (STL, OBJ, DAE, PLY)
- Batch processing
- Quality control

.. code-block:: bash

   skr convert-urdf-mesh robot.urdf \
       --output optimized_robot.urdf \
       --quality 0.5

**visualize-urdf**

Interactive 3D visualization:

.. code-block:: bash

   skr visualize-urdf robot.urdf --viewer trimesh

4. Hash-based Model Management
-------------------------------

**Comprehensive Content Hashing**

Computes SHA-256 hashes including:

- URDF XML content (using W3C XML Canonicalization)
- All referenced mesh files
- All texture files

**Hash-based URI Scheme**

.. code-block:: python

   from skrobot.models import RobotModel

   # Load model by hash (auto-downloads if needed)
   robot = RobotModel.from_urdf_file(
       "hash://sha256:abc123..."
   )

**Benefits**

- Reliable model identity
- Automatic distribution
- Version management
- Simulation-to-hardware transfer

5. Visualization
----------------

Multiple visualization backends:

**TrimeshSceneViewer**

.. code-block:: python

   from skrobot.viewers import TrimeshSceneViewer

   viewer = TrimeshSceneViewer()
   viewer.add(robot)
   viewer.show()

**PyrenderViewer**

For smoother rendering with OpenGL:

.. code-block:: python

   from skrobot.viewers import PyrenderViewer

   viewer = PyrenderViewer()
   viewer.add(robot)
   viewer.show()

**JupyterNotebookViewer**

Interactive visualization in Jupyter notebooks and Google Colab:

.. code-block:: python

   from skrobot.viewers import JupyterNotebookViewer

   viewer = JupyterNotebookViewer(height=600)
   viewer.add(robot)
   viewer.show()

   # Animate without flickering
   robot.rarm.angle_vector([0.1, 0.2, 0.3])
   viewer.redraw()

6. Motion Planning
------------------

**SQP-based Trajectory Optimization**

.. code-block:: python

   from skrobot.planner import sqp_plan_trajectory
   from skrobot.planner import SweptSphereSdfCollisionChecker

   # Create collision checker
   collision_checker = SweptSphereSdfCollisionChecker(
       robot_model=robot,
       obstacles=obstacle_sdf
   )

   # Plan collision-free trajectory
   trajectory = sqp_plan_trajectory(
       robot,
       start_av,
       goal_av,
       collision_checker=collision_checker
   )

**Features**

- Swept sphere collision detection
- Signed Distance Field (SDF) for obstacles
- Sequential Quadratic Programming optimization
- Smooth, collision-free paths

7. ROS Interface
----------------

**ROSRobotInterface**

Connects scikit-robot to real robots via ROS:

.. code-block:: python

   from skrobot.interfaces.ros import ROSRobotInterfaceBase

   class MyRobotInterface(ROSRobotInterfaceBase):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)

   ri = MyRobotInterface(robot)

   # Subscribe to joint states
   ri.update_robot_state()

   # Send angle vector
   ri.angle_vector(target_av, time=5.0)
   ri.wait_interpolation()

**Features**

- JointState topic subscription
- FollowJointTrajectory action client
- Synchronization between model and real robot
- Consistent API for sim and real robots

Integration with External Tools
================================

**PyBullet Simulation**

.. code-block:: python

   from skrobot.interfaces._pybullet import PybulletRobotInterface

   ri = PybulletRobotInterface(robot)
   ri.angle_vector(target_av)
   ri.wait_interpolation()

**Blender Visualization**

FormaMotus Blender plugin uses scikit-robot as backend:

- High-quality rendering
- Joint structure diagrams
- Animation export

**CAD Software Integration**

Compatible with CAD exporters:

- Onshape-to-robot
- SolidWorks-to-URDF
- Fusion 360 URDF Exporter

Data Flow
=========

1. **Model Loading**: URDF → RobotModel (with hash verification)
2. **Computation**: User commands → Coordinates/RobotModel → Transformations
3. **Visualization**: RobotModel → Viewer → Display
4. **Control**: Commands → Interface → Simulator/Real Robot
5. **Planning**: Start/Goal → Planner → Trajectory → Execution

Performance Considerations
===========================

**Lazy Evaluation**

CascadedCoords computes transformations only when ``worldcoords()`` or ``worldpos()`` is called.

**Mesh Caching**

Hash-based caching provides up to 5× speedup for models with shared meshes.

**Incremental Updates**

Only affected coordinate branches recalculated on changes.

**Efficient Data Structures**

Tree-based link hierarchy enables O(log n) lookups.

This architecture provides a solid foundation for robot development, from simple scripts to complex reconfigurable systems.

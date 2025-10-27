==========================
Signed Distance Functions
==========================

**Example script**: ``examples/signed_distance_functions.py``

This example demonstrates creating and using Signed Distance Functions (SDFs) for collision detection.

Source Code
===========

https://github.com/iory/scikit-robot/blob/main/examples/signed_distance_functions.py

What This Example Shows
========================

- Creating primitive SDFs (Box, Sphere, Cylinder)
- Combining multiple SDFs with UnionSDF
- Converting meshes to SDFs
- Using SDFs for collision checking

Key Concepts
============

SDFs represent the distance from any point in space to the nearest surface:

- Negative values: inside the object
- Positive values: outside the object
- Zero: on the surface

This makes them efficient for collision detection and motion planning.

Related Documentation
=====================

- :doc:`../reference/sdfs` - SDF API reference
- :doc:`collision_free_trajectory` - Using SDFs for planning

"""Collision detection primitives and utilities.

This module provides backend-agnostic collision detection for robotics
applications, supporting both NumPy and JAX for differentiable collision
avoidance in trajectory optimization.

Geometry Primitives
-------------------
- Sphere: Spherical collision geometry
- Capsule: Capsule (line segment + radius) collision geometry
- Box: Axis-aligned or oriented bounding box
- HalfSpace: Infinite half-space (plane)

Distance Functions
------------------
- collision_distance: Automatic dispatch based on geometry types
- sphere_sphere_distance, capsule_capsule_distance, etc.
- colldist_from_sdf: Smooth activation for optimization

Robot Collision
---------------
- RobotCollisionChecker: Manages collision geometries for robot links
- LinkCollisionGeometry: Geometry attached to a robot link

Example
-------
>>> from skrobot.collision import Sphere, Capsule, collision_distance
>>> import numpy as np
>>> s1 = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=0.5)
>>> s2 = Sphere(center=np.array([2.0, 0.0, 0.0]), radius=0.5)
>>> dist = collision_distance(s1, s2)  # returns 1.0
"""

# Geometry primitives
# Distance functions
from skrobot.collision.distance import box_halfspace_distance
from skrobot.collision.distance import capsule_box_distance
from skrobot.collision.distance import capsule_capsule_distance
from skrobot.collision.distance import capsule_halfspace_distance
from skrobot.collision.distance import colldist_from_sdf
from skrobot.collision.distance import collision_distance
from skrobot.collision.distance import sphere_box_distance
from skrobot.collision.distance import sphere_capsule_distance
from skrobot.collision.distance import sphere_halfspace_distance
from skrobot.collision.distance import sphere_sphere_distance
from skrobot.collision.geometry import Box
from skrobot.collision.geometry import Capsule
from skrobot.collision.geometry import CollisionGeometry
from skrobot.collision.geometry import HalfSpace
from skrobot.collision.geometry import Sphere

# Robot collision
from skrobot.collision.robot_collision import LinkCollisionGeometry
from skrobot.collision.robot_collision import RobotCollisionChecker


__all__ = [
    # Geometry primitives
    'CollisionGeometry',
    'Sphere',
    'Capsule',
    'Box',
    'HalfSpace',
    # Distance functions
    'collision_distance',
    'sphere_sphere_distance',
    'capsule_capsule_distance',
    'sphere_capsule_distance',
    'sphere_box_distance',
    'capsule_box_distance',
    'sphere_halfspace_distance',
    'capsule_halfspace_distance',
    'box_halfspace_distance',
    'colldist_from_sdf',
    # Robot collision
    'RobotCollisionChecker',
    'LinkCollisionGeometry',
]

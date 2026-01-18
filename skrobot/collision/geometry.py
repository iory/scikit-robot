"""Collision geometry primitives.

This module provides collision geometry primitives (Sphere, Capsule, Box, HalfSpace)
that work with both NumPy and JAX backends.

Example
-------
>>> from skrobot.collision import Sphere, Capsule, collision_distance
>>> import numpy as np
>>> s1 = Sphere(center=np.array([0, 0, 0]), radius=0.5)
>>> s2 = Sphere(center=np.array([2, 0, 0]), radius=0.5)
>>> dist = collision_distance(s1, s2)  # returns 1.0
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CollisionGeometry:
    """Base class for collision geometries.

    All collision geometries support:
    - Transformation (position + rotation)
    - Backend-agnostic operations (numpy or jax.numpy)
    """

    def transform(self, position, rotation, xp=np):
        """Transform geometry to world frame.

        Parameters
        ----------
        position : array (3,)
            Translation vector.
        rotation : array (3, 3)
            Rotation matrix.
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        CollisionGeometry
            Transformed geometry in world frame.
        """
        raise NotImplementedError

    def get_batch_axes(self):
        """Get batch axes for this geometry.

        Returns
        -------
        tuple
            Shape of batch dimensions.
        """
        return ()


@dataclass
class Sphere(CollisionGeometry):
    """Sphere collision geometry.

    Parameters
    ----------
    center : array (3,) or (..., 3)
        Center position(s).
    radius : float or array
        Radius (scalar or batched).
    """
    center: np.ndarray
    radius: float

    def transform(self, position, rotation, xp=np):
        """Transform sphere to world frame."""
        new_center = position + xp.matmul(rotation, self.center)
        return Sphere(center=new_center, radius=self.radius)

    def get_batch_axes(self):
        if self.center.ndim > 1:
            return self.center.shape[:-1]
        return ()

    @classmethod
    def from_center_and_radius(cls, center, radius):
        """Create sphere from center and radius.

        Parameters
        ----------
        center : array-like (3,)
            Center position.
        radius : float
            Sphere radius.

        Returns
        -------
        Sphere
            Sphere instance.
        """
        return cls(center=np.asarray(center), radius=float(radius))


@dataclass
class Capsule(CollisionGeometry):
    """Capsule collision geometry (line segment + radius).

    A capsule is defined by two endpoints and a radius.
    It's the Minkowski sum of a line segment and a sphere.

    Parameters
    ----------
    p1 : array (3,) or (..., 3)
        First endpoint.
    p2 : array (3,) or (..., 3)
        Second endpoint.
    radius : float or array
        Capsule radius.
    """
    p1: np.ndarray
    p2: np.ndarray
    radius: float

    def transform(self, position, rotation, xp=np):
        """Transform capsule to world frame."""
        new_p1 = position + xp.matmul(rotation, self.p1)
        new_p2 = position + xp.matmul(rotation, self.p2)
        return Capsule(p1=new_p1, p2=new_p2, radius=self.radius)

    def get_batch_axes(self):
        if self.p1.ndim > 1:
            return self.p1.shape[:-1]
        return ()

    @property
    def height(self):
        """Capsule height (distance between endpoints)."""
        return np.linalg.norm(self.p2 - self.p1)

    @property
    def center(self):
        """Capsule center (midpoint of segment)."""
        return (self.p1 + self.p2) / 2

    @property
    def axis(self):
        """Capsule axis (unit vector from p1 to p2)."""
        diff = self.p2 - self.p1
        length = np.linalg.norm(diff)
        if length < 1e-10:
            return np.array([0, 0, 1])
        return diff / length

    @classmethod
    def from_center_height_axis(cls, center, height, axis, radius):
        """Create capsule from center, height, axis and radius.

        Parameters
        ----------
        center : array-like (3,)
            Center position.
        height : float
            Capsule height (distance between endpoints).
        axis : array-like (3,)
            Capsule axis direction (will be normalized).
        radius : float
            Capsule radius.

        Returns
        -------
        Capsule
            Capsule instance.
        """
        center = np.asarray(center)
        axis = np.asarray(axis)
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        half_height = height / 2
        p1 = center - half_height * axis
        p2 = center + half_height * axis
        return cls(p1=p1, p2=p2, radius=float(radius))

    @classmethod
    def from_endpoints(cls, p1, p2, radius):
        """Create capsule from two endpoints and radius.

        Parameters
        ----------
        p1 : array-like (3,)
            First endpoint.
        p2 : array-like (3,)
            Second endpoint.
        radius : float
            Capsule radius.

        Returns
        -------
        Capsule
            Capsule instance.
        """
        return cls(p1=np.asarray(p1), p2=np.asarray(p2), radius=float(radius))


@dataclass
class Box(CollisionGeometry):
    """Axis-aligned bounding box collision geometry.

    Parameters
    ----------
    center : array (3,) or (..., 3)
        Box center.
    half_extents : array (3,) or (..., 3)
        Half-extents (half width, half height, half depth).
    rotation : array (3, 3) or (..., 3, 3), optional
        Rotation matrix. If None, box is axis-aligned.
    """
    center: np.ndarray
    half_extents: np.ndarray
    rotation: Optional[np.ndarray] = None

    def transform(self, position, rot, xp=np):
        """Transform box to world frame."""
        new_center = position + xp.matmul(rot, self.center)
        if self.rotation is not None:
            new_rotation = xp.matmul(rot, self.rotation)
        else:
            new_rotation = rot
        return Box(center=new_center, half_extents=self.half_extents,
                   rotation=new_rotation)

    def get_batch_axes(self):
        if self.center.ndim > 1:
            return self.center.shape[:-1]
        return ()

    @classmethod
    def from_center_and_extents(cls, center, extents, rotation=None):
        """Create box from center and full extents.

        Parameters
        ----------
        center : array-like (3,)
            Box center.
        extents : array-like (3,)
            Full extents (width, height, depth).
        rotation : array-like (3, 3), optional
            Rotation matrix.

        Returns
        -------
        Box
            Box instance.
        """
        center = np.asarray(center)
        half_extents = np.asarray(extents) / 2
        if rotation is not None:
            rotation = np.asarray(rotation)
        return cls(center=center, half_extents=half_extents, rotation=rotation)


@dataclass
class HalfSpace(CollisionGeometry):
    """Half-space (infinite plane) collision geometry.

    Defined by a point on the plane and an outward normal.
    Points are "inside" the half-space if (x - point) . normal < 0.

    Parameters
    ----------
    point : array (3,)
        A point on the plane.
    normal : array (3,)
        Outward normal of the plane (should be unit vector).
    """
    point: np.ndarray
    normal: np.ndarray

    def transform(self, position, rotation, xp=np):
        """Transform half-space to world frame."""
        new_point = position + xp.matmul(rotation, self.point)
        new_normal = xp.matmul(rotation, self.normal)
        return HalfSpace(point=new_point, normal=new_normal)

    @classmethod
    def from_point_and_normal(cls, point, normal):
        """Create half-space from point and normal.

        Parameters
        ----------
        point : array-like (3,)
            A point on the plane.
        normal : array-like (3,)
            Outward normal (will be normalized).

        Returns
        -------
        HalfSpace
            HalfSpace instance.
        """
        point = np.asarray(point)
        normal = np.asarray(normal)
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        return cls(point=point, normal=normal)

    @classmethod
    def ground_plane(cls, height=0.0):
        """Create a ground plane (z = height).

        Parameters
        ----------
        height : float
            Height of the ground plane.

        Returns
        -------
        HalfSpace
            Ground plane half-space.
        """
        return cls(point=np.array([0, 0, height]), normal=np.array([0, 0, 1]))

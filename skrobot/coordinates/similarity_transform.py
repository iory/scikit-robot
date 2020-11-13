import copy

import numpy as np

from skrobot.coordinates.base import transform_coords
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import quaternion_multiply


def similarity_transform_coords(c1, c2, out=None):
    """Return Coordinates by applying c1 to c2 from the left

    Parameters
    ----------
    c1 : SimilarityTransformCoordinates
        input Coordinates
    c2 : SimilarityTransformCoordinates
        input Coordinates
    out : SimilarityTransformCoordinates or None
        Output argument. If this value is specified, the results will be
        in-placed.

    Returns
    -------
    out : SimilarityTransformCoordinates
        new coordinates
    """
    if out is None:
        out = SimilarityTransformCoordinates()
    elif not isinstance(out, SimilarityTransformCoordinates):
        raise TypeError("Input type should be "
                        "skrobot.coordinates.SimilarityTransformCoordinates")
    out.translation = c1.translation + c1.scale * np.dot(
        c1.rotation, c2.translation)
    out.rotation = quaternion_multiply(c1.quaternion, c2.quaternion)
    out.scale = c1.scale * c2.scale
    return out


class SimilarityTransformCoordinates(CascadedCoords):

    def __init__(self, *args, **kwargs):
        scale = kwargs.pop('scale', 1.0)
        super(SimilarityTransformCoordinates, self).__init__(
            *args, **kwargs)
        self.scale = scale

    def inverse_transformation(self, dest=None):
        """Return a invese transformation of this coordinate system.

        Create a new coordinate with inverse transformation of this
        coordinate system.

        """
        inv_scale = 1.0 / self.scale
        if dest is None:
            dest = SimilarityTransformCoordinates(scale=inv_scale)
        dest.rotation = self.rotation.T
        dest.translation = np.matmul(dest.rotation, self.translation)
        dest.translation = -1.0 * dest.translation * inv_scale
        return dest

    def T(self):
        """Return 4x4 homogeneous transformation matrix.

        Returns
        -------
        matrix : np.ndarray
            homogeneous transformation matrix shape of (4, 4)
        """
        matrix = np.r_[np.c_[self.rotation, self.translation],
                       [[0, 0, 0, 1]]]
        scale_mat = np.eye(4)
        scale_mat[:3, :3] = np.diag(self.scale * np.ones(3))
        matrix = matrix.dot(scale_mat)
        return matrix

    def transform(self, c, wrt='local'):
        """Transform this coordinate by coords based on wrt

        Note that this function changes this coordinate's
        translation and rotation.
        If you would like not to change this coordinate,
        Please use copy_worldcoords()

        Parameters
        ----------
        c : skrobot.coordinates.Coordinates
            coordinate
        wrt : string or skrobot.coordinates.Coordinates
            If wrt is 'local' or self, multiply c from left.
            If wrt is 'world', transform c with respect to worldcoord.
            If wrt is Coordinates, transform c with respect to c.

        Returns
        -------
        self : skrobot.coordinates.Coordinates
            return this coordinate
        """
        if isinstance(c, SimilarityTransformCoordinates):
            transform_func = similarity_transform_coords
        else:
            transform_func = transform_coords
        if wrt == 'local' or wrt == self:
            transform_func(self, c, self)
        elif wrt == 'parent' or wrt == self.parent \
                or wrt == 'world':
            transform_func(c, self, self)
        elif isinstance(wrt, Coordinates):
            transform_func(wrt.inverse_transformation(), self, self)
            transform_func(c, self, self)
            transform_func(wrt.worldcoords(), self, self)
        else:
            raise ValueError('transform wrt {} is not supported'.format(wrt))
        return self

    def transform_vector(self, v):
        """"Return vector represented at world frame.

        Vector v given in the local coords is converted to world
        representation.

        Parameters
        ----------
        v : numpy.ndarray
            3d vector.
            We can take batch of vector like (batch_size, 3)
        Returns
        -------
        transformed_point : numpy.ndarray
            transformed point
        """
        v = np.array(v, dtype=np.float64)
        if v.ndim == 2:
            return (self.scale * np.matmul(self.rotation, v.T)
                    + self.translation.reshape(3, -1)).T
        return self.scale * np.matmul(self.rotation, v) + self.translation

    def inverse_transform_vector(self, vec):
        """Transform vector in world coordinates to local coordinates

        Parameters
        ----------
        vec : numpy.ndarray
            3d vector.
            We can take batch of vector like (batch_size, 3)
        Returns
        -------
        transformed_point : numpy.ndarray
            transformed point
        """
        vec = np.array(vec, dtype=np.float64)
        inv_scale = 1.0 / self.scale
        if vec.ndim == 2:
            return inv_scale \
                * (np.matmul(self.rotation.T, vec.T) - np.matmul(
                    self.rotation.T, self.translation).reshape(3, -1)).T
        return inv_scale \
            * (np.matmul(self.rotation.T, vec)
               - np.matmul(self.rotation.T, self.translation))

    def copy_coords(self):
        """Return a deep copy of the Coordinates."""
        return SimilarityTransformCoordinates(
            pos=copy.deepcopy(self.worldpos()),
            rot=copy.deepcopy(self.worldrot()),
            scale=self.scale)

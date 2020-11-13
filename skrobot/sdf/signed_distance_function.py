from numbers import Number

import numpy as np
import pysdfgen

from skrobot.coordinates.math import normalize_vector
from skrobot.coordinates.similarity_transform import \
    SimilarityTransformCoordinates


class SDF(SimilarityTransformCoordinates):

    def __init__(self, sdf_data, origin, resolution,
                 use_abs=True,
                 *args, **kwargs):
        super(SDF, self).__init__(*args, **kwargs)
        self.num_interpolants = 8
        self.min_coords_x = [0, 2, 3, 5]
        self.max_coords_x = [1, 4, 6, 7]
        self.min_coords_y = [0, 1, 3, 6]
        self.max_coords_y = [2, 4, 5, 7]
        self.min_coords_z = [0, 1, 2, 4]
        self.max_coords_z = [3, 5, 6, 7]

        self._data = sdf_data
        self._origin = origin
        self._dims = self.data.shape
        self.resolution = resolution

        spts, _ = self.surface_points()
        self._center = 0.5 * (np.min(spts, axis=0) + np.max(spts, axis=0))

        self.sdf_to_grid_transform = SimilarityTransformCoordinates(
            pos=self.origin,
            scale=self.resolution)

        # buffer
        self._points_buf = np.zeros([self.num_interpolants, 3], dtype=np.int)

        # optionally use only the absolute values
        # (useful for non-closed meshes in 3D)
        self.use_abs = use_abs
        if use_abs:
            self._data = np.abs(self.data)

    @property
    def dimensions(self):
        """SDF dimension information.

        Returns
        -------
        self._dims : numpy.ndarray
            dimension of this sdf.
        """
        return self._dims

    @property
    def origin(self):
        """Return the location of the origin in the SDF grid.

        Returns
        -------
        self._origin : numpy.ndarray
            The 3-ndarray that contains the location of
            the origin of the mesh grid in real space.
        """
        return self._origin

    @property
    def resolution(self):
        """The grid resolution (how wide each grid cell is).

        Resolution is max dist from a surface when the surface
        is orthogonal to diagonal grid cells

        Returns
        -------
        self._resolution : float
            The width of each grid cell.
        """
        return self._resolution

    @resolution.setter
    def resolution(self, res):
        """Setter of resolution.

        Parameters
        ----------
        res : float
            new resolution.
        """
        self._resolution = res
        self._surface_threshold = res * np.sqrt(2) / 2.0

    @property
    def surface_threshold(self):
        """Threshold of surface value.

        Returns
        -------
        self._surface_threshold : float
            threshold
        """
        return self._surface_threshold

    @property
    def center(self):
        """Center of grid.

        This basically transforms the world frame to grid center.

        Returns
        -------
        :obj:`numpy.ndarray`
        """
        return self._center

    def on_surface(self, coords):
        """Determines whether or not a point is on the object surface.

        Parameters
        ----------
        coords : :obj:`numpy.ndarray` of int
            A 2- or 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        :obj:`tuple` of bool, float
            Is the point on the object's surface, and what
            is the signed distance at that point?
        """
        sdf_val = self[coords]
        if np.abs(sdf_val) < self.surface_threshold:
            return True, sdf_val
        return False, sdf_val

    def is_out_of_bounds(self, grid_coords):
        """Returns True if coords is an out of bounds access.

        Parameters
        ----------
        grid_coords : numpy.ndarray or list of int
            3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        is_out : bool
            If coords is in grid, return True.
        """
        coords = np.array(grid_coords)
        if coords.ndim == 1:
            return np.array(coords < 0).any() or \
                np.array(coords >= self.dimensions).any()
        elif coords.ndim == 2:
            return np.logical_or(
                (coords < 0).any(axis=1),
                (coords >= np.array(self.dimensions)).any(axis=1))
        else:
            raise ValueError

    @property
    def data(self):
        """The SDF data.

        Returns
        -------
        self._data : numpy.ndarray
            The 3-dimensional ndarray that holds the grid of signed distances.
        """
        return self._data

    def _signed_distance(self, grid_coords):
        """Returns the signed distance at the given coordinates

        Interpolating if necessary.

        Parameters
        ----------
        grid_coords : numpy.ndarray
            A 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        float or numpy.ndarray
            The signed distance at the given coords (interpolated).
        """
        grid_coords = np.array(grid_coords)
        if grid_coords.ndim == 1:
            if len(grid_coords) != 3:
                raise IndexError('Indexing must be 3 dimensional')
            if self.is_out_of_bounds(grid_coords):
                # logging.debug('Out of bounds access. Snapping to SDF dims')
                pass

            # snap to grid dims
            coords_buf = np.zeros(3)
            coords_buf[0] = max(0, min(grid_coords[0], self.dimensions[0] - 1))
            coords_buf[1] = max(0, min(grid_coords[1], self.dimensions[1] - 1))
            coords_buf[2] = max(0, min(grid_coords[2], self.dimensions[2] - 1))

            # regular indexing if integers
            if type(grid_coords[0]) is int and \
                    type(grid_coords[1]) is int and \
                    type(grid_coords[2]) is int:
                coords_buf = coords_buf.astype(np.int)
                return self.data[coords_buf[0], coords_buf[1], coords_buf[2]]

            # otherwise interpolate
            min_coords = np.floor(coords_buf)
            max_coords = min_coords + 1  # assumed to be on grid
            self._points_buf[self.min_coords_x, 0] = min_coords[0]
            self._points_buf[self.max_coords_x, 0] = max_coords[0]
            self._points_buf[self.min_coords_y, 1] = min_coords[1]
            self._points_buf[self.max_coords_y, 1] = max_coords[1]
            self._points_buf[self.min_coords_z, 2] = min_coords[2]
            self._points_buf[self.max_coords_z, 2] = max_coords[2]

            # bilinearly interpolate points
            sd = 0.0
            for i in range(self.num_interpolants):
                p = self._points_buf[i, :]
                if self.is_out_of_bounds(p):
                    v = 0.0
                else:
                    v = self.data[p[0], p[1], p[2]]
                w = np.prod(-np.abs(p - coords_buf) + 1)
                sd = sd + w * v

            return sd
        elif grid_coords.ndim == 2:
            # for batch input
            coords_buf = np.maximum(
                0, np.minimum(grid_coords, np.array(self.dimensions) - 1))
            sd = np.zeros(len(grid_coords), dtype=np.float64)
            no_interpolating = (
                coords_buf == np.array(coords_buf, dtype=np.int32)).all(axis=1)
            no_interpolating_coords = np.array(
                coords_buf[no_interpolating], dtype=np.int32)
            if len(no_interpolating_coords) > 0:
                sd[no_interpolating] = self.data[
                    no_interpolating_coords[:, 0],
                    no_interpolating_coords[:, 1],
                    no_interpolating_coords[:, 2],
                ]

            interpolating_coords = coords_buf[np.logical_not(no_interpolating)]
            if len(interpolating_coords) == 0:
                return sd

            min_coords = np.floor(interpolating_coords)
            max_coords = min_coords + 1  # assumed to be on grid

            n = len(interpolating_coords)
            points_buf = np.zeros([n, self.num_interpolants, 3], dtype=np.int)
            points_buf[:, self.min_coords_x, 0] = np.repeat(
                min_coords[:, 0][None, ], 4, axis=0).T
            points_buf[:, self.max_coords_x, 0] = np.repeat(
                max_coords[:, 0][None, ], 4, axis=0).T
            points_buf[:, self.min_coords_y, 1] = np.repeat(
                min_coords[:, 1][None, ], 4, axis=0).T
            points_buf[:, self.max_coords_y, 1] = np.repeat(
                max_coords[:, 1][None, ], 4, axis=0).T
            points_buf[:, self.min_coords_z, 2] = np.repeat(
                min_coords[:, 2][None, ], 4, axis=0).T
            points_buf[:, self.max_coords_z, 2] = np.repeat(
                max_coords[:, 2][None, ], 4, axis=0).T

            # bilinearly interpolate points
            interpolating_sd = sd[np.logical_not(no_interpolating)]
            for i in range(self.num_interpolants):
                p = points_buf[:, i, :]
                valid = np.logical_not(self.is_out_of_bounds(p))
                p = p[valid]
                v = self.data[p[:, 0], p[:, 1], p[:, 2]]
                w = np.prod(-np.abs(p - interpolating_coords[valid]) + 1,
                            axis=1)
                interpolating_sd[valid] = interpolating_sd[valid] + w * v
            sd[np.logical_not(no_interpolating)] = interpolating_sd
            return sd
        else:
            raise ValueError

    def __getitem__(self, grid_coords):
        """Returns the signed distance at the given coordinates.

        Parameters
        ----------
        grid_coords : numpy.ndarray
            A or 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        sd : float
            The signed distance at the given grid_coords (interpolated).
        """
        return self._signed_distance(grid_coords)

    def surface_normal(self, grid_coords, delta=1.5):
        """Returns the sdf surface normal at the given coordinates

        Returns the sdf surface normal at the given coordinates by
        computing the tangent plane using SDF interpolation.

        Parameters
        ----------
        grid_coords : numpy.ndarray
            A 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        delta : float
            A radius for collecting surface points near the target coords
            for calculating the surface normal.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 3-dimensional ndarray that represents the surface normal.

        Raises
        ------
        IndexError
            If the coords vector does not have three entries.
        """
        grid_coords = np.array(grid_coords)
        if grid_coords.ndim == 1:
            if len(grid_coords) != 3:
                raise IndexError('Indexing must be 3 dimensional')

            # log warning if out of bounds access
            if self.is_out_of_bounds(grid_coords):
                # print('Out of bounds access. Snapping to SDF dims')
                pass

            # snap to grid dims
            grid_coords[0] = max(
                0, min(grid_coords[0], self.dimensions[0] - 1))
            grid_coords[1] = max(
                0, min(grid_coords[1], self.dimensions[1] - 1))
            grid_coords[2] = max(
                0, min(grid_coords[2], self.dimensions[2] - 1))
            index_coords = np.zeros(3)

            # check points on surface
            sdf_val = self[grid_coords]
            if np.abs(sdf_val) >= self.surface_threshold:
                return None

            # collect all surface points within the delta sphere
            X = []
            d = np.zeros(3)
            dx = -delta
            while dx <= delta:
                dy = -delta
                while dy <= delta:
                    dz = -delta
                    while dz <= delta:
                        d = np.array([dx, dy, dz])
                        if dx != 0 or dy != 0 or dz != 0:
                            d = delta * normalize_vector(d)
                        index_coords[0] = grid_coords[0] + d[0]
                        index_coords[1] = grid_coords[1] + d[1]
                        index_coords[2] = grid_coords[2] + d[2]
                        sdf_val = self[index_coords]
                        if np.abs(sdf_val) < self.surface_threshold:
                            X.append([index_coords[0], index_coords[1],
                                      index_coords[2], sdf_val])
                        dz += delta
                    dy += delta
                dx += delta

            # fit a plane to the surface points
            X.sort(key=lambda x: x[3])
            X = np.array(X)[:, :3]
            A = X - np.mean(X, axis=0)
            try:
                U, S, V = np.linalg.svd(A.T)
                n = U[:, 2]
            except np.linalg.LinAlgError:
                return None
            return n
        elif grid_coords.ndim == 2:
            invalid_normals = self.is_out_of_bounds(grid_coords)
            valid_normals = np.logical_not(invalid_normals)
            n = len(grid_coords)
            indices = np.arange(n)[valid_normals]
            normals = np.nan * np.ones((n, 3))
            grid_coords = grid_coords[valid_normals]

            if len(grid_coords) == 0:
                return normals
            grid_coords = np.maximum(
                0, np.minimum(grid_coords, np.array(self.dimensions) - 1))

            # check points on surface
            sdf_val = self[grid_coords]
            valid_surfaces = np.abs(sdf_val) < self.surface_threshold
            indices = indices[valid_surfaces]

            grid_coords = grid_coords[valid_surfaces]

            if len(grid_coords) == 0:
                return normals

            # collect all surface points within the delta sphere
            X = np.inf * np.ones((len(grid_coords), 27, 4), dtype=np.float64)
            dx = - delta
            for i in range(3):
                dy = - delta
                for j in range(3):
                    dz = - delta
                    for k in range(3):
                        d = np.array([dx, dy, dz])
                        if dx != 0 or dy != 0 or dz != 0:
                            d = delta * normalize_vector(d)
                        index_coords = grid_coords + d
                        sdf_val = self[index_coords]
                        flags = np.abs(sdf_val) < self.surface_threshold
                        X[flags, (i * 9) + (j * 3) + k, :3] = index_coords[
                            flags]
                        X[flags, (i * 9) + (j * 3) + k, 3] = sdf_val[flags]
                        dz += delta
                    dy += delta
                dx += delta

            # fit a plane to the surface points
            for i, x in enumerate(X):
                x = x[~np.isinf(x[:, 3])]
                if len(x) != 0:
                    x = x[np.argsort(x[:, 3])]
                    x = x[:, :3]
                    A = x - np.mean(x, axis=0)
                    try:
                        U, S, V = np.linalg.svd(A.T)
                        normal = U[:, 2]
                    except np.linalg.LinAlgError:
                        normal = np.nan * np.ones(3)
                else:
                    normal = np.nan * np.ones(3)
                normals[indices[i]] = normal
            return normals
        else:
            raise ValueError

    def surface_points(self, grid_basis=True):
        """Returns the points on the surface.

        Parameters
        ----------
        grid_basis : bool
            If False, the surface points are transformed to the world frame.
            If True (default), the surface points are left in grid coordinates.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of int, :obj:`numpy.ndarray` of
            float. The points on the surface and the signed distances at
            those points.
        """
        surface_points = np.where(np.abs(self.data) < self.surface_threshold)
        x = surface_points[0]
        y = surface_points[1]
        z = surface_points[2]
        surface_points = np.c_[x, np.c_[y, z]]
        surface_values = self.data[surface_points[:, 0],
                                   surface_points[:, 1],
                                   surface_points[:, 2]]

        print('start')
        if not grid_basis:
            surface_points = self.transform_pt_grid_to_obj(surface_points.T)
            surface_points = surface_points.T
        print('end')

        return surface_points, surface_values

    def transform_pt_obj_to_grid(self, x_sdf, direction=False):
        """Converts a point in sdf coords to the grid basis.

        If direction is True, don't translate.

        Parameters
        ----------
        x_sdf : numpy 3xN ndarray or numeric scalar
            points to transform from sdf basis in meters to grid basis

        Returns
        -------
        x_grid : numpy 3xN ndarray or scalar
            points in grid basis
        """
        if isinstance(x_sdf, Number):
            return self.copy_worldcoords().transform(
                self.sdf_to_grid_transform
            ).inverse_transformation().scale * x_sdf
        if direction is True:
            # 1 / s [R^T v - R^Tp] p == 0 case
            x_grid = np.dot(x_sdf, self.copy_worldcoords().transform(
                self.sdf_to_grid_transform).worldrot().T)
        else:
            x_grid = self.copy_worldcoords().transform(
                self.sdf_to_grid_transform).inverse_transform_vector(x_sdf.T)
        return x_grid

    def transform_pt_grid_to_obj(self, x_grid, direction=False):
        """Converts a point in grid coords to the obj basis.

        If direction is True, then don't translate.

        Parameters
        ----------
        x_grid : numpy.ndarray or numbers.Number
            3xN ndarray or numeric scalar
            points to transform from grid basis to sdf basis in meters
        direction : bool
            If this value is True, x_grid treated as normal vectors.

        Returns
        -------
        x_sdf : numpy.ndarray
            3xN ndarray. points in sdf basis (meters)
        """
        if isinstance(x_grid, Number):
            return self.copy_worldcoords().transform(
                self.sdf_to_grid_transform).scale * x_grid

        if direction:
            x_sdf = np.dot(x_grid, self.copy_worldcoords().transform(
                self.sdf_to_grid_transform).worldrot().T)
        else:
            x_sdf = self.copy_worldcoords().transform(
                self.sdf_to_grid_transform).transform_vector(
                    x_grid.astype(np.float32).T)
        return x_sdf

    @staticmethod
    def from_file(filepath):
        """Return SDF instance from .sdf file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            path of .sdf file

        Returns
        -------
        sdf_instance : skrobot.exchange.sdf.SDF
            instance of sdf
        """
        with open(filepath, 'r') as f:
            # dimension of each axis should all be equal for LSH
            nx, ny, nz = [int(i) for i in f.readline().split()]
            ox, oy, oz = [float(i) for i in f.readline().split()]
            dims = np.array([nx, ny, nz])
            origin = np.array([ox, oy, oz])

            # resolution of the grid cells in original mesh coords
            resolution = float(f.readline())
            sdf_data = np.zeros(dims)

            # loop through file, getting each value
            count = 0
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        sdf_data[i][j][k] = float(f.readline())
                        count += 1
        return SDF(sdf_data, origin, resolution)

    @staticmethod
    def from_objfile(obj_filepath, dim=100, padding=5):
        """Return SDF instance from .obj file.

        This file Internally create .sdf file from .obj file.
        Converting obj to SDF tooks a some time.

        Parameters
        ----------
        obj_filepath : str or pathlib.Path
            path of objfile
        dim : int
            dim of sdf
        padding : int
            number of padding

        Returns
        -------
        sdf_instance : skrobot.exchange.sdf.SDF
            instance of sdf
        """
        sdf_filepath = pysdfgen.obj2sdf(str(obj_filepath), dim, padding)
        return SDF.from_file(sdf_filepath)

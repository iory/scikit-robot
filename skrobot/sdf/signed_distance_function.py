from __future__ import division

from logging import getLogger
from math import floor
import os

import numpy as np
import pysdfgen
from scipy.interpolate import RegularGridInterpolator

from skrobot.coordinates import CascadedCoords
from skrobot.coordinates import make_cascoords
from skrobot.coordinates import Transform
from skrobot.data import get_cache_dir
from skrobot.utils import checksum_md5


logger = getLogger(__name__)


def trimesh2sdf(mesh, dim_grid=100, padding_grid=5):
    """Convert trimesh to signed distance function.

    Parameters
    ----------
    mesh : trimesh.base.Trimesh
        mesh object.
    dim_grid : int
        dimension of the GridSDF.
        This value is used for a not primitive mesh.
    padding_grid : int
        number of padding.

    Returns
    -------
    sdf : skrobot.sdf.SignedDistanceFunction
        converted signed distance function.
    """
    if not ('file_path' in mesh.metadata
            or 'shape' in mesh.metadata):
        raise ValueError("Input mesh doesn't contain valid metadata"
                         " for converting SDF.")
    is_loaded_mesh = 'file_path' in mesh.metadata
    if is_loaded_mesh:
        file_path = mesh.metadata['file_path']
        sdf = GridSDF.from_objfile(file_path, dim_grid=dim_grid,
                                   padding_grid=padding_grid)
    else:
        # process primtives
        shape = mesh.metadata['shape']
        if shape == 'box':
            extents = mesh.metadata['extents']
            sdf = BoxSDF([0, 0, 0], extents)
        elif shape == 'cylinder':
            height = mesh.metadata['height']
            radius = mesh.metadata['radius']
            sdf = CylinderSDF([0, 0, 0], radius=radius, height=height)
        elif shape == 'sphere':
            radius = mesh.metadata['radius']
            sdf = SphereSDF([0, 0, 0], radius)
        else:
            msg = "primtive type {0} is not supported".format(shape)
            raise ValueError(msg)

    origin = mesh.metadata["origin"]
    rotation_matrix = origin[:3, :3]
    translation = origin[:3, 3]
    sdf.coords = make_cascoords(pos=translation, rot=rotation_matrix)
    return sdf


def link2sdf(link, urdf_path, dim_grid=30):
    """Convert Link to corresponding sdf

    Parameters
    ----------
    link : skrobot.model.Link
        link object
    urdf_path : str
        urdf path of the robot model that the link belongs to
    dim_grid : int
        dimension of the GridSDF

    Returns
    -------
    sdf : skrobot.sdf.SignedDistanceFunction
        corresponding signed distance function to the link type.
        e.g. if Link has geometry of urdf.Box, then BoxSDF is
        created.
    """

    sdf = trimesh2sdf(link.collision_mesh, dim_grid=dim_grid)
    link.assoc(sdf.coords, relative_coords=sdf.coords)
    return sdf


class SignedDistanceFunction(object):
    """A base class for signed distance functions (SDFs).

    Suffixes `_obj` and `_sdf` (e.g. in points_sdf) indicate that
    points are expressed in the sdf's object coordinate (`self.coords`)
    and sdf-specfic coordinate respectively. Each SDF performs the
    signed-distance computation in its own sdf-specific coordinate.
    For example, the origin of GridSDF's sdf-specific coordinate is
    the tip of the precomputed-gridded-box.

    Usually, a child class implements the computation in the sdf-specific
    coordinates and SignedDistanceFunction wraps them so that the user can
    pass and get points and values expressed in an object coordinate. Thus,
    it is less likely that a user directly calls a method in a child class.
    """

    def __init__(self, origin, coords=None, use_abs=False):
        if coords is None:
            coords = CascadedCoords()

        self.coords = coords
        self.sdf_to_obj_transform = Transform(origin, np.eye(3))
        self._origin = np.array(origin)
        self.use_abs = use_abs

    def __call__(self, points_obj):
        """Compute signed distances of input points to the implicit surface.

        Parameters
        -------
        points_obj : numpy.ndarray[float](n_point, 3)
            2 dim point array w.r.t. object coordinate.

        Returns
        -------
        signed_distances : numpy.ndarray[float]
            1 dim (n_point,) array of signed distance.
        """
        points_sdf = self._transform_pts_obj_to_sdf(points_obj)
        sd_vals = self._signed_distance(points_sdf)
        if self.use_abs:
            return np.abs(sd_vals)
        return sd_vals

    def on_surface(self, points_obj):
        """Check if points are on the surface.

        Parameters
        ----------
        points_obj : numpy.ndarray[float](n_point, 3)
            2 dim point array w.r.t. an object coordinate.

        Returns
        -------
        logicals : numpy.ndarray[bool](n_point,)
            boolean vector of the on-surface predicate for `points_obj`.
        sd_vals : numpy.ndarray[float](n_point,)
            signed distances corresponding to `logicals`.
        """
        sd_vals = self.__call__(points_obj)
        logicals = np.abs(sd_vals) < self._surface_threshold
        return logicals, sd_vals

    def surface_points(self, n_sample=1000):
        """Sample points from the implicit surface of the sdf.

        Parameters
        ----------
        n_sample : int
            number of sample points.

        Returns
        -------
        points_obj : numpy.ndarray[float](n_point, 3)
            sampled points w.r.t object coordinate.
        dists : numpy.ndarray[float](n_point,)
            signed distances corresponding to points_obj.
        """
        points_, dists = self._surface_points(n_sample=n_sample)
        points_obj = self._transform_pts_sdf_to_obj(points_)
        return points_obj, dists

    def _transform_pts_obj_to_sdf(self, points_obj):
        """Transform points from to an object coordinate to a sdf-specific coordinate.

        Parameters
        ----------
        points_obj : numpy.ndarray[float](n_point, 3)
            2 dim point array w.r.t. an object coordinate.

        Returns
        -------
        points_sdf : numpy.ndarray[float](n_point, 3)
            2 dim point array w.r.t. a sdf-specific coordinate.
        """
        tf_world_to_local =\
            self.coords.get_transform().inverse_transformation()
        tf_local_to_sdf = self.sdf_to_obj_transform.inverse_transformation()
        tf_world_to_sdf = tf_world_to_local * tf_local_to_sdf
        points_sdf = tf_world_to_sdf.transform_vector(points_obj)
        return points_sdf

    def _transform_pts_sdf_to_obj(self, points_sdf):
        """Transform points from to a sdf-specific coordinate to an object coordinate.

        Parameters
        ----------
        points_sdf : numpy.ndarray[float](n_point, 3)
            2 dim point array w.r.t. a sdf-specific coordinate.

        Returns
        -------
        points_obj : numpy.ndarray[float](n_point, 3)
            2 dim point array w.r.t. an object coordinate.
        """
        tf_local_to_world = self.coords.get_transform()
        tf_sdf_to_local = self.sdf_to_obj_transform
        tf_sdf_to_world = tf_sdf_to_local * tf_local_to_world
        points_obj = tf_sdf_to_world.transform_vector(points_sdf)
        return points_obj


class UnionSDF(SignedDistanceFunction):
    """One can concat multiple SDFs `sdf_list` by using this class.

    For consistency in the concatenation, it is required that
    the all SDFs to be concated are with `use_abs=False`.
    """

    def __init__(self, sdf_list, coords=None):
        origin = np.zeros(3)
        use_abs = False
        super(UnionSDF, self).__init__(origin, coords=coords, use_abs=use_abs)

        use_abs_list = [sdf.use_abs for sdf in sdf_list]
        all_false = np.all(~np.array(use_abs_list))
        assert all_false, "use_abs for each sdf must be consistent"

        self.sdf_list = sdf_list

        threshold_list = [sdf._surface_threshold for sdf in sdf_list]
        self._surface_threshold = max(threshold_list)

    def _signed_distance(self, points_sdf):
        sd_vals_list = np.array([sdf(points_sdf) for sdf in self.sdf_list])
        sd_vals_union = np.min(sd_vals_list, axis=0)
        return sd_vals_union

    def _surface_points(self, n_sample=1000):
        # equaly asign sample number to each sdf.surface_points()
        n_list = len(self.sdf_list)
        n_sample_each = int(floor(n_sample / n_list))
        n_sample_last = n_sample - n_sample_each * (n_list - 1)
        num_list = [n_sample_each] * (n_list - 1) + [n_sample_last]

        points = np.vstack([sdf.surface_points(n_sample=n_sample_)[0]
                            for sdf, n_sample_
                            in zip(self.sdf_list, num_list)])
        logicals, sd_vals = self.on_surface(points)
        return points[logicals], sd_vals[logicals]

    @classmethod
    def from_robot_model(cls, robot_model, dim_grid=50):
        """Create union sdf from a robot model

        Parameters
        ----------
        robot_model : skrobot.model.RobotModel
            Using the links of the robot_model this creates
            the UnionSDF instance.

        Returns
        -------
        union_sdf : skrobot.sdf.UnionSDF
            union sdf of robot_model
        """
        sdf_list = []
        for link in robot_model.link_list:
            if link.collision_mesh is not None:
                sdf = link2sdf(link, robot_model.urdf_path, dim_grid=dim_grid)
                sdf_list.append(sdf)
        return cls(sdf_list)


class BoxSDF(SignedDistanceFunction):
    """SDF for a box specified by `origin` and `width`."""

    def __init__(self, origin, width, coords=None, use_abs=False):
        super(BoxSDF, self).__init__(origin, coords=coords, use_abs=use_abs)
        self._width = np.array(width)
        self._surface_threshold = np.min(self._width) * 1e-2

    def _signed_distance(self, points_sdf):
        n_pts, _ = points_sdf.shape

        half_extent = self._width * 0.5
        pts_from_center = points_sdf - self._origin[None, :]
        sd_vals_each_axis = np.abs(pts_from_center) - half_extent[None, :]

        positive_dists_each_axis = np.maximum(sd_vals_each_axis, 0.0)
        positive_dists = np.sqrt(np.sum(positive_dists_each_axis**2, axis=1))

        negative_dists_each_axis = np.max(sd_vals_each_axis, axis=1)
        negative_dists = np.minimum(negative_dists_each_axis, 0.0)

        sd_vals = positive_dists + negative_dists
        return sd_vals

    def _surface_points(self, n_sample=1000):
        # surface points by raymarching
        vecs = np.random.randn(n_sample, 3)
        ray_tips = np.zeros((n_sample, 3))
        return ray_marching(ray_tips, vecs,
                            self._signed_distance,
                            self._surface_threshold)


class SphereSDF(SignedDistanceFunction):
    """SDF for a sphere specified by `origin` and `radius`."""

    def __init__(self, origin, radius, coords=None, use_abs=False):
        super(SphereSDF, self).__init__(origin, coords=coords, use_abs=use_abs)
        self._radius = radius
        self._surface_threshold = radius * 1e-2

    def _signed_distance(self, points_sdf):
        n_pts, _ = points_sdf.shape
        c = self._origin

        diffs = points_sdf - c[None, :]
        dists_from_origin = np.sqrt(np.sum(diffs**2, axis=1))
        sd_vals = dists_from_origin - self._radius
        return sd_vals

    def _surface_points(self, n_sample=1000):
        # surface points by raymarching
        vecs = np.random.randn(n_sample, 3)
        ray_tips = np.zeros((n_sample, 3))
        return ray_marching(ray_tips, vecs,
                            self._signed_distance,
                            self._surface_threshold)


class CylinderSDF(SignedDistanceFunction):
    """SDF for a cylinder specified by `origin`,`radius` and `height`"""

    def __init__(self, origin, height, radius, coords=None, use_abs=False):
        super(CylinderSDF, self).__init__(origin,
                                          coords=coords, use_abs=use_abs)
        self._height = height
        self._radius = radius
        self._surface_threshold = min(radius, height) * 1e-2

    def _signed_distance(self, points_sdf):
        n_pts, _ = points_sdf.shape
        c = self._origin
        height_half = 0.5 * self._height

        pts_from_center_3d = points_sdf - c[None, :]
        radius_from_center = np.sqrt(
            pts_from_center_3d[:, 0]**2 + pts_from_center_3d[:, 1]**2)
        height_from_center = pts_from_center_3d[:, 2]

        # Now the problem is reduced to 2 dim [radius, height] box sdf
        # so the algorithm from now is the same as the box sdf computation
        half_extent_2d = np.array([self._radius, height_half])
        pts_from_center_2d = np.vstack(
            [radius_from_center, height_from_center]).T
        sd_vals_each_axis = np.abs(pts_from_center_2d)\
            - half_extent_2d[None, :]

        positive_dists_each_axis = np.maximum(sd_vals_each_axis, 0.0)
        positive_dists = np.sqrt(np.sum(positive_dists_each_axis**2, axis=1))

        negative_dists_each_axis = np.max(sd_vals_each_axis, axis=1)
        negative_dists = np.minimum(negative_dists_each_axis, 0.0)

        sd_vals = positive_dists + negative_dists
        return sd_vals

    def _surface_points(self, n_sample=1000):
        # surface points by raymarching
        vecs = np.random.randn(n_sample, 3)
        ray_tips = np.zeros((n_sample, 3))
        return ray_marching(ray_tips, vecs,
                            self._signed_distance,
                            self._surface_threshold)


class GridSDF(SignedDistanceFunction):
    """SDF using precopmuted signed distances for gridded points."""

    def __init__(self, sdf_data, origin, resolution,
                 fill_value=np.inf, coords=None, use_abs=False):

        super(GridSDF, self).__init__(origin, coords=coords, use_abs=use_abs)
        # optionally use only the absolute values
        # (useful for non-closed meshes in 3D)
        self._data = np.abs(sdf_data) if use_abs else sdf_data
        self._dims = np.array(self._data.shape)
        self._resolution = resolution
        self._surface_threshold = resolution * np.sqrt(2) / 2.0

        # create regular grid interpolator
        xlin, ylin, zlin = [
            np.array(range(d)) * resolution for d in self._data.shape]
        self.itp = RegularGridInterpolator(
            (xlin, ylin, zlin),
            self._data,
            bounds_error=False,
            fill_value=fill_value)

        spts, _ = self._surface_points()

        self.sdf_to_obj_transform = Transform(origin, np.eye(3))

    def is_out_of_bounds(self, points_obj):
        """check if the the input points is out of bounds

        This method checks if the the input points is out of bounds
        of RegularGridInterpolator.

        Parameters
        ----------
        points_obj : numpy.ndarray[float](n_points, 3)
            points w.r.t. object to be checked.

        Returns
        -------
        is_out_arr : numpy.ndarray[bool](n_points,)
            If points is out of the interpolator's boundary,
            the correspoinding element of is_out_arr is True
        """
        points_sdf = super(GridSDF, self)._transform_pts_obj_to_sdf(points_obj)
        points_grid = np.array(points_sdf) / self._resolution
        is_out_arr = np.logical_or(
            (points_grid < 0).any(axis=1),
            (points_grid >= np.array(self._dims)).any(axis=1))
        return is_out_arr

    def _signed_distance(self, points_sdf):
        points_sdf = np.array(points_sdf)
        sd_vals = self.itp(points_sdf)
        return sd_vals

    def _surface_points(self, n_sample=None):
        surface_points = np.where(np.abs(self._data) < self._surface_threshold)
        x = surface_points[0]
        y = surface_points[1]
        z = surface_points[2]
        surface_points = np.c_[x, np.c_[y, z]]
        surface_values = self._data[surface_points[:, 0],
                                    surface_points[:, 1],
                                    surface_points[:, 2]]
        if n_sample is not None:
            # somple points WITHOUT duplication
            n_pts = len(surface_points)
            n_sample = min(n_sample, n_pts)
            idxes = np.random.permutation(n_pts)[:n_sample]

            # update points and sds
            surface_points = surface_points[idxes]
            surface_values = surface_values[idxes]

        return surface_points * self._resolution, surface_values

    @staticmethod
    def from_file(filepath):
        """Return GridSDF instance from a .sdf file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            path of .sdf file

        Returns
        -------
        sdf_instance : skrobot.esdf.GridSDF
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
            sdf_data = np.fromstring(f.read(), dtype=float, sep='\n').reshape(
                *dims).transpose(2, 1, 0)
        return GridSDF(sdf_data, origin, resolution)

    @staticmethod
    def from_objfile(obj_filepath, dim_grid=100, padding_grid=5):
        """Return GridSDF instance from an .obj file.

        In the initial call of this method for an .obj file,
        the pre-process takes some time to converting it to
        a .sdf file. However, because a cache of .sdf file is
        created in the initial call, there is no overhead
        from the next call for the same .obj file.

        Parameters
        ----------
        obj_filepath : str or pathlib.Path
            path of objfile
        dim_grid : int
            dim of sdf
        padding_grid : int
            number of padding

        Returns
        -------
        sdf_instance : skrobot.sdf.GridSDF
            instance of sdf
        """
        sdf_cache_dir = os.path.join(get_cache_dir(), 'sdf')
        if not os.path.exists(sdf_cache_dir):
            os.makedirs(sdf_cache_dir)

        hashed_filename = '{}_{}_{}'.format(
            checksum_md5(obj_filepath), dim_grid, padding_grid)

        sdf_cache_path = os.path.join(sdf_cache_dir, hashed_filename + '.sdf')
        if not os.path.exists(sdf_cache_path):
            logger.info(
                'pre-computing sdf and making a cache at {0}.'
                .format(sdf_cache_path))
            pysdfgen.obj2sdf(str(obj_filepath), dim_grid, padding_grid,
                             output_filepath=sdf_cache_path)
            logger.info('finish pre-computation')
        return GridSDF.from_file(sdf_cache_path)


def ray_marching(pts_starts, direction_arr, f_sdf, threshold):
    norms = np.linalg.norm(direction_arr, axis=1)
    direction_arr_unit = direction_arr / norms[:, None]
    ray_tips = pts_starts
    while True:
        sd_vals = f_sdf(ray_tips)
        ray_tips += direction_arr_unit * sd_vals[:, None]
        if np.all(np.abs(sd_vals) < threshold):
            break
    tips_final = ray_tips
    sd_vals_final = f_sdf(ray_tips)
    return tips_final, sd_vals_final

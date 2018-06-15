import copy

import numpy as np

from robot.math import _wrap_axis
from robot.math import matrix2quaternion
from robot.math import matrix_log
from robot.math import normalize_vector
from robot.math import quaternion2matrix
from robot.math import random_rotation
from robot.math import random_translation
from robot.math import rotate_matrix
from robot.math import rotation_angle
from robot.math import rotation_matrix
from robot.math import rotation_matrix_from_rpy
from robot.math import rpy2quaternion
from robot.math import rpy_angle
from robot.math import rpy_matrix
from robot.math import quaternion_multiply


def transform_coords(c1, c2):
    pos = c1.pos + np.dot(c1.rotation, c2.pos)
    rot = quaternion2matrix(quaternion_multiply(c1._q, c2._q))
    return Coordinates(pos=pos, rot=rot)


class Coordinates(object):

    def __init__(self, pos=None,
                 rot=np.eye(3),
                 dimension=3,
                 euler=None,
                 rpy=None,
                 axis=None,
                 angle=None,
                 wrt='local',
                 name=None):
        self.rotation = rot
        if pos is None:
            pos = np.zeros(3)
        if rpy is None:
            rpy = np.zeros(3)
        else:
            self.newcoords(rpy_matrix(rpy[0],
                                      rpy[1],
                                      rpy[2],
                                      pos))
        self.translation = pos
        self.rpy = rpy
        self.name = name
        self.parent_link = None

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        rotation = np.array(rotation)
        # Convert quaternions
        if rotation.shape == (4,):
            self._q = np.array([q for q in rotation])
            if np.abs(np.linalg.norm(self._q) - 1.0) > 1e-3:
                raise ValueError('Invalid quaternion. Must be '
                                 'norm 1.0, get {}'.
                                 format(np.linalg.norm(self._q)))
            rotation = quaternion2matrix(self._q)
        elif rotation.shape == (3,):
            # Convert [yaw-pitch-roll] to rotation matrix
            self._q = rpy2quaternion(rotation)
            rotation = quaternion2matrix(self._q)
        else:
            self._q = matrix2quaternion(rotation)

        # Convert lists and tuples
        if type(rotation) in (list, tuple):
            rotation = np.array(rotation).astype(np.float32)

        self._check_valid_rotation(rotation)
        self._rotation = rotation * 1.

    @property
    def translation(self):
        return self.pos

    @translation.setter
    def translation(self, translation):
        # Convert lists to translation arrays
        if type(translation) in (list, tuple) and len(translation) == 3:
            translation = np.array([t for t in translation]).astype(np.float32)

        self._check_valid_translation(translation)
        self.pos = translation.squeeze() * 1.

    def _check_valid_rotation(self, rotation):
        """Checks that the given rotation matrix is valid.
        """
        if not isinstance(rotation, np.ndarray) or \
           not np.issubdtype(rotation.dtype, np.number):
            raise ValueError('Rotation must be specified \
                              as numeric numpy array')

        if len(rotation.shape) != 2 or \
           rotation.shape[0] != 3 or \
           rotation.shape[1] != 3:
            raise ValueError('Rotation must be specified as a 3x3 ndarray')

        if np.abs(np.linalg.det(rotation) - 1.0) > 1e-3:
            raise ValueError('Illegal rotation. Must have '
                             'determinant == 1.0, get {}'.
                             format(np.linalg.det(rotation)))

    def _check_valid_translation(self, translation):
        """Checks that the translation vector is valid.
        """
        if not isinstance(translation, np.ndarray) or \
           not np.issubdtype(translation.dtype, np.number):
            raise ValueError('Translation must be specified \
                              as numeric numpy array')
        t = translation.squeeze()
        if len(t.shape) != 1 or t.shape[0] != 3:
            raise ValueError('Translation must be specified as a 3-vector, \
                              3x1 ndarray, or 1x3 ndarray')

    @property
    def dimension(self):
        return len(self.pos)

    def translate(self, vec, wrt='local'):
        """translate this coordinates. unit is [mm]"""
        vec = np.array(vec, dtype=np.float64)
        vec /= 1000.0
        return self.newcoords(self.rotation,
                              self.parent_orientation(vec, wrt) + self.pos)

    def transform_vector(self, v):
        """"Vector v given in the local coords
        is converted to world representation"""
        return np.matmul(self.rotation, v) + self.pos

    def inverse_transform_vector(self, vec):
        """vec in world coordinates -> local"""
        return np.matmul(self.rotation.T, vec) - np.matmul(self.rotation.T, self.pos)

    def inverse_transformation(self, dest=None):
        """Create a new coordinate
        with inverse transformation of
        this coordinate system."""
        if dest is None:
            dest = Coordinates()
        dest.rotation = self.rotation.T
        dest.translation = np.matmul(dest.rotation, self.translation)
        dest.translation = -1.0 * dest.translation
        return dest

    def transformation(self, c2, wrt='local'):
        c2 = c2.worldcoords()
        c1 = self.worldcoords()
        inv = c1.inverse_transformation()
        if wrt == 'local' or wrt == self:
            inv = transform_coords(inv, c2)
        elif wrt == 'parent' or \
                wrt == self.parent_link or \
                wrt == 'world':
            inv = transform_coords(c2, inv)
        elif isinstance(wrt, Coordinates):
            xw = wrt.worldcoords()
            inv = transform_coords(c2, inv)
            inv = transform_coords(xw.inverse_transformation(),
                                   inv)
            inv = transform_coords(inv, xw)
        else:
            raise ValueError("wrt {} not supported".format(wrt))
        return inv

    def T(self):
        """Return 4x4 transformation matrix"""
        matrix = np.zeros((4, 4), dtype=np.float64)
        matrix[3, 3] = 1.0
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.pos
        return matrix

    @property
    def quaternion(self):
        return matrix2quaternion(self.rotation)

    def parent_orientation(self, v, wrt):
        if wrt == 'local' or wrt == self:
            return np.matmul(self.rotation, v)
        if wrt == 'parent' \
           or wrt == self.parent_link \
           or wrt == 'world':
            return v
        raise ValueError('wrt {} not supported'.format(wrt))

    def rotate_vector(self, v):
        return np.matmul(self.rotation, v)

    def transform(self, c, wrt='local'):
        if wrt == 'local' or wrt == self:
            tmp_coords = transform_coords(self, c)
            self.rotation = tmp_coords.rotation
            self.translation = tmp_coords.translation
        elif wrt == 'parent' or wrt == self.parent_link \
                or wrt == 'world':
            tmp_coords = transform_coords(c, self)
            self.rotation = tmp_coords.rotation
            self.translation = tmp_coords.translation
        elif isinstance(wrt, Coordinates):
            tmp_coords = transform_coords(wrt.inverse_transformation, self)
            tmp_coords = transform_coords(c, tmp_coords)
            tmp_coords = transform_coords(wrt.worldcoords(), tmp_coords)
            self.rotation = tmp_coords.rotation
            self.translation = tmp_coords.translation
        else:
            raise ValueError("transform wrt {} is not supported".format(wrt))
        return self.newcoords(self.rotation, self.pos)

    def rpy_angle(self):
        return rpy_angle(self.rotation)

    def axis(self, ax):
        ax = _wrap_axis(ax)
        return self.rotate_vector(ax)

    def difference_position(self, coords,
                            translation_axis=True):
        """return diffece in positoin of given coords, translation-axis can take (:x, :y, :z, :xy, :yz, :zx)."""
        dif_pos = self.inverse_transform_vector(coords.worldpos())
        translation_axis = _wrap_axis(translation_axis)
        dif_pos[translation_axis == 1] = 0.0
        return dif_pos

    def difference_rotation(self, coords,
                            rotation_axis=True):
        """return diffece in rotation of given coords, rotation-axis can take
        (:x, :y, :z, :xx, :yy, :zz, :xm, :ym, :zm)"""
        def need_mirror_for_nearest_axis(coords0, coords1, ax):
            a0 = coords0.axis(ax)
            a1 = coords1.axis(ax)
            a1_mirror = - a1
            dr1 = np.arccos(np.dot(a0, a1)) * \
                normalize_vector(np.cross(a0, a1))
            dr1m = np.arccos(np.dot(a0, a1_mirror)) * \
                normalize_vector(np.cross(a0, a1_mirror))
            return np.linalg.norm(dr1) < np.linalg.norm(dr1m)

        if rotation_axis in ['x', 'y', 'z']:
            a0 = self.axis(rotation_axis)
            a1 = coords.axis(rotation_axis)
            dif_rot = np.matmul(self.worldrot().T,
                                np.arccos(np.dot(a0, a1)) *
                                normalize_vector(np.cross(a0, a1)))
        elif rotation_axis in ['xx', 'yy', 'zz']:
            ax = rotation_axis[0]
            a0 = self.axis(ax)
            a2 = coords.axis(ax)
            if need_mirror_for_nearest_axis(self, coords, ax) is False:
                a2 = - a2
            dif_rot = np.matmul(self.worldrot().T,
                                np.arccos(np.dot(a0, a2)) *
                                normalize_vector(np.cross(a0, a2)))
        elif rotation_axis is False:
            dif_rot = np.array([0, 0, 0])
        elif rotation_axis is True:
            dif_rotmatrix = np.matmul(self.worldrot().T,
                                      coords.worldrot())
            dif_rot = matrix_log(dif_rotmatrix)
        else:
            raise ValueError
        return dif_rot

    def rotate_with_matrix(self, mat, wrt='local'):
        if wrt == 'local' or wrt == self:
            self.rotation = np.matmul(self.rotation, mat)
            self.newcoords(self.rotation, self.pos)
        elif wrt == 'parent' or wrt == self.parent_link:
            self.rotation = np.matmul(mat, self.rotation)
            self.newcoords(self.rotation, self.pos)
        elif isinstance(wrt, Coordinates):
            r2 = wrt.worldrot()
            r2t = r2.T
            r2t = np.matmul(mat, r2t)
            r2t = np.matmul(r2, r2t)
            self.rotation = np.matmul(r2t, self.rotation)
        else:
            raise ValueError('wrt {} is not supported'.format(wrt))

    def rotate(self, theta, axis=None, wrt="local"):
        if isinstance(axis, list) or isinstance(axis, np.ndarray):
            self.rotation = self.rotate_with_matrix(
                rotation_matrix(theta, axis), wrt)
        elif axis is None or axis is False:
            self.rotation = self.rotate_with_matrix(theta, wrt)
        elif wrt == 'local' or wrt == self:
            self.rotation = rotate_matrix(self.rotation, theta, axis)
        elif wrt == 'parent' or wrt == 'world':
            self.rotation = rotate_matrix(self.rotation, theta,
                                          axis, True)
        elif isinstance(wrt, Coordinates):  # C1'=C2*R*C2(-1)*C1
            self.rotation = self.rotate_with_matrix(
                rotation_matrix(theta, axis), wrt)
        else:
            raise ValueError('wrt {} not supported'.format(wrt))
        return self.newcoords(self.rotation, self.pos)

    def copy(self):
        return self.copy_coords()

    def copy_coords(self):
        """Returns a deep copy of the Coordinates.
        """
        return Coordinates(pos=copy.deepcopy(self.worldpos()),
                           rot=copy.deepcopy(self.worldrot()))

    def coords(self):
        return self.copy_coords()

    def worldcoords(self):
        return self

    def copy_worldcoords(self):
        return self.coords()

    def update(self):
        pass

    def worldrot(self):
        return self.rotation

    def worldpos(self):
        return self.pos

    def newcoords(self, c, pos=None):
        """
        Update of coords is always done through newcoords
        """
        if pos is not None:
            self.rotation = copy.deepcopy(c)
            self.translation = copy.deepcopy(pos)
        else:
            self.rotation = copy.deepcopy(c.rotation)
            self.translation = copy.deepcopy(c.translation)
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        rot = self.worldrot()
        pos = self.worldpos()
        self.rpy = rpy_angle(self.rotation)[0]
        if self.name:
            prefix = self.__class__.__name__ + ' ' + self.name
        else:
            prefix = self.__class__.__name__

        return '#<%s %.1lf %.1lf %.1lf / %.1lf %.1lf %.1lf>' % (prefix,
                                                                pos[0] *
                                                                1000.0,
                                                                pos[1] *
                                                                1000.0,
                                                                pos[2] *
                                                                1000.0,
                                                                self.rpy[0],
                                                                self.rpy[1],
                                                                self.rpy[2])


class CascadedCoords(Coordinates):

    def __init__(self, parent=None, *args, **kwargs):
        super(CascadedCoords, self).__init__(*args, **kwargs)
        self.manager = self
        self._changed = True
        # self.worldcoords = Coordinates(rot=rot, pos=pos)

        self.child_links = []
        if parent:
            self.parent_link = parent
            self.parent_link.add_child(self)
        self._worldcoords = Coordinates(pos=self.pos,
                                        rot=self.rotation)
        self.descendants = []

    def newcoords(self, c, pos=None):
        super(CascadedCoords, self).newcoords(c, pos)
        self.changed()
        return self

    def changed(self):
        if self._changed is False:
            self._changed = True
            return [c.changed() for c in self.child_links]
        return [False]

    def parentcoords(self):
        if self.parent_link:
            return self.parent_link.worldcoords()

    def transform_vector(self, v):
        return self.worldcoords().transform_vector(v)

    def inverse_transform_vector(self, v):
        return self.worldcoords().inverse_transform_vector(v)

    def rotate_with_matrix(self, matrix, wrt):
        if wrt == 'local' or wrt == self:
            self.rotation = np.dot(self.rotation, matrix)
            return self.newcoords(self.rotation, self.pos)
        elif wrt == 'parent' or wrt == self.parent_link:
            self.rotation = np.matmul(matrix, self.rotation)
            return self.newcoords(self.rotation, self.pos)
        else:
            parent_coords = self.parentcoords()
            parent_rot = parent_coords.rotation
            if isinstance(wrt, Coordinates):
                wrt_rot = wrt.worldrot()
                matrix = np.matmul(wrt_rot, matrix)
                matrix = np.matmul(matrix, wrt_rot.T)
            matrix = np.matmul(matrix, parent_rot)
            matrix = np.matmul(parent_rot.T, matrix)
            self.rotation = np.matmul(matrix, self.rotation)
            return self.newcoords(self.rotation, self.pos)

    def rotate(self, theta, axis, wrt='local'):
        if isinstance(axis, list) or isinstance(axis, np.ndarray):
            return self.rotate_with_matrix(
                rotation_matrix(theta, axis), wrt)
        if isinstance(axis, np.ndarray) and axis.shape == (3, 3):
            return self.rotate_with_matrix(theta, wrt)

        if wrt == 'local' or wrt == self:
            self.rotation = rotate_matrix(self.rotation, theta, axis)
            return self.newcoords(self.rotation, self.pos)
        elif wrt == 'parent' or wrt == self.parent_link:
            self.rotation = rotate_matrix(self.rotation, theta, axis)
            return self.newcoords(self.rotation, self.pos)
        else:
            return self.rotate_with_matrix(
                rotation_matrix(theta, axis), wrt)

    def transform(self, c, wrt='local'):
        if isinstance(wrt, Coordinates):
            raise NotImplementedError
        elif wrt == 'local' or wrt == self:  # multiply c from the left
            tmp_coords = transform_coords(self, c)
            self.rotation = tmp_coords.rotation
            self.translation = tmp_coords.translation
        else:
            raise NotImplementedError
        return self.newcoords(self.rotation, self.pos)

    def worldcoords(self):
        """Calculate rotation and position in the world."""
        if self._changed:
            if self.parent_link:
                self._worldcoords = transform_coords(
                    self.parent_link.worldcoords(),
                    self)
            else:
                self._worldcoords.rotation = self.rotation
                self._worldcoords.pos = self.pos
            self.update()
            self._changed = False
        return self._worldcoords

    def worldrot(self):
        return self.worldcoords().rotation

    def worldpos(self):
        return self.worldcoords().pos

    def add_child(self, child_link):
        self.child_links.append(child_link)

    @property
    def parent(self):
        return self.parent_link

    def update(self):
        pass


def make_coords(*args, **kwargs):
    return Coordinates(*args, **kwargs)


def make_cascoords(*args, **kwargs):
    return CascadedCoords(*args, **kwargs)


def random_coords():
    return Coordinates(pos=random_translation(),
                       rot=random_rotation())


def wrt(coords, vec):
    return coords.transform_vector(vec)


def coordinates_distance(c1, c2, c=None):
    if c is None:
        c = c1.transformation(c2)
    return np.linalg.norm(c.worldpos()), rotation_angle(c.worldrot())[0]

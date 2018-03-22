import copy

import numpy as np

from robot.math import _wrap_axis
from robot.math import matrix2quaternion
from robot.math import matrix_log
from robot.math import normalize_vector
from robot.math import random_rotation
from robot.math import random_translation
from robot.math import rotate_matrix
from robot.math import rotation_angle
from robot.math import rotation_matrix
from robot.math import rpy_angle
from robot.math import rpy_matrix


def transform_coords(c1, c2):
    pos = c1.pos + np.dot(c1.rot, c2.pos)
    rot = np.dot(c1.rot, c2.rot)
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
        self.rot = rot
        if pos is None:
            pos = np.zeros(3)
        if rpy is None:
            rpy = np.zeros(3)
        else:
            self.newcoords(rpy_matrix(rpy[0],
                                      rpy[1],
                                      rpy[2],
                                      pos))
        self.pos = pos
        self.rpy = rpy
        self.name = name
        self.parent_link = None

    @property
    def rotation(self):
        return self.rot

    @property
    def translation(self):
        return self.pos

    @property
    def dimension(self):
        return len(self.pos)

    def translate(self, vec, wrt='local'):
        """translate this coordinates. unit is [mm]"""
        vec = np.array(vec, dtype=np.float64)
        vec /= 1000.0
        return self.newcoords(self.rot,
                              self.parent_orientation(vec, wrt) + self.pos)

    def transform_vector(self, v):
        """"Vector v given in the local coords
        is converted to world representation"""
        return np.matmul(self.rot, v) + self.pos

    def inverse_transform_vector(self, vec):
        """vec in world coordinates -> local"""
        return np.matmul(self.rot.T, vec) - np.matmul(self.rot.T, self.pos)

    def inverse_transformation(self, dest=None):
        """Create a new coordinate
        with inverse transformation of
        this coordinate system."""
        if dest is None:
            dest = Coordinates()
        dest.rot = self.rot.T
        dest.pos = np.matmul(dest.rot, self.pos)
        dest.pos = -1.0 * dest.pos
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
        matrix[:3, :3] = self.rot
        matrix[:3, 3] = self.pos
        return matrix

    def quaternion(self):
        return matrix2quaternion(self.rot)

    def parent_orientation(self, v, wrt):
        if wrt == 'local' or wrt == self:
            return np.matmul(self.rot, v)
        if wrt == 'parent' \
           or wrt == self.parent_link \
           or wrt == 'world':
            return v
        raise ValueError('wrt {} not supported'.format(wrt))

    def rotate_vector(self, v):
        return np.matmul(self.rot, v)

    def transform(self, c, wrt='local'):
        if wrt == 'local' or wrt == self:
            self = transform_coords(self, c)
        elif wrt == 'parent' or wrt == self.parent_link \
             or wrt == 'world':
            self = transform_coords(c, self)
        elif isinstance(wrt, Coordinates):
            self = transform_coords(wrt.inverse_transformation, self)
            self = transform_coords(c, self)
            self = transform_coords(wrt.worldcoords(), self)
        else:
            raise ValueError("transform wrt {} is not supported".format(wrt))
        return self.newcoords(self.rot, self.pos)

    def rpy_angle(self):
        return rpy_angle(self.rot)

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
                                np.arccos(np.dot(a0, a1)) * \
                                normalize_vector(np.cross(a0, a1)))
        elif rotation_axis in ['xx', 'yy', 'zz']:
            ax = rotation_axis[0]
            a0 = self.axis(ax)
            a2 = coords.axis(ax)
            if need_mirror_for_nearest_axis(self, coords, ax) is False:
                a2 = - a2
            dif_rot = np.matmul(self.worldrot().T,
                                np.arccos(np.dot(a0, a2)) * \
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
            self.rot = np.matmul(self.rot, mat)
            self.newcoords(self.rot, self.pos)
        elif wrt == 'parent' or wrt == self.parent_link:
            self.rot = np.matmul(mat, self.rot)
            self.newcoords(self.rot, self.pos)
        elif isinstance(wrt, Coordinates):
            r2 = wrt.worldrot()
            r2t = r2.T
            r2t = np.matmul(mat, r2t)
            r2t = np.matmul(r2, r2t)
            self.rot = np.matmul(r2t, self.rot)
        else:
            raise ValueError('wrt {} is not supported'.format(wrt))

    def rotate(self, theta, axis=None, wrt="local"):
        if isinstance(axis, list) or isinstance(axis, np.ndarray):
            self.rot = self.rotate_with_matrix(
                rotation_matrix(theta, axis), wrt)
        elif axis is None or axis is False:
            self.rot = self.rotate_with_matrix(theta, wrt)
        elif wrt == 'local' or wrt == self:
            self.rot = rotate_matrix(self.rot, theta, axis,
                                     True)
        elif wrt == 'parent' or wrt == 'world':
            self.rot = rotate_matrix(self.rot, theta,
                                     axis)
        elif isinstance(wrt, Coordinates):  # C1'=C2*R*C2(-1)*C1
            self.rot = self.rotate_with_matrix(
                rotation_matrix(theta, axis), wrt)
        else:
            raise ValueError('wrt {} not supported'.format(wrt))
        return self.newcoords(self.rot, self.pos)

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

    def worldrot(self):
        return self.rot

    def worldpos(self):
        return self.pos

    def newcoords(self, c, pos=None):
        if isinstance(c, Coordinates):
            self.rot = copy.deepcopy(c.rot)
            self.pos = copy.deepcopy(c.pos)
        elif pos is not None:
            c = np.array(c)
            if not c.shape == (3, 3):
                c = rpy_matrix(c[0], c[1], c[2])
            self.rot = copy.deepcopy(c)
            self.pos = copy.deepcopy(pos)
        else:
            raise NotImplementedError
        return self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        rot = self.worldrot()
        pos = self.worldpos()
        self.rpy = rpy_angle(self.rot)[0]
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
                                        rot=self.rot)
        self.descendants = []

    def newcoords(self, c, pos=None):
        super(CascadedCoords, self).newcoords(c, pos)
        self.changed()
        return self

    def changed(self):
        if self._changed is False:
            self._changed = True
            for child_link in self.child_links:
                child_link.changed()

    def parentcoords(self):
        if self.parent_link:
            return self.parent_link.worldcoords()

    def transform_vector(self, v):
        return self.worldcoords().transform_vector(v)

    def inverse_transform_vector(self, v):
        return self.worldcoords().inverse_transform_vector(v)

    def rotate_with_matrix(self, matrix, wrt):
        if wrt == 'local' or wrt == self:
            self.rot = np.dot(self.rot, matrix)
            return self.newcoords(self.rot, self.pos)
        elif wrt == 'parent' or wrt == self.parent_link:
            self.rot = np.matmul(matrix, self.rot)
            return self.newcoords(self.rot, self.pos)
        else:
            parent_coords = self.parentcoords()
            parent_rot = parent_coords.rot
            if isinstance(wrt, Coordinates):
                wrt_rot = wrt.worldrot()
                matrix = np.matmul(wrt_rot, matrix)
                matrix = np.matmul(matrix, wrt_rot.T)
            matrix = np.matmul(matrix, parent_rot)
            matrix = np.matmul(parent_rot.T, matrix)
            self.rot = np.matmul(matrix, self.rot)
            return self.newcoords(self.rot, self.pos)

    def rotate(self, theta, axis, wrt='local'):
        if isinstance(axis, list) or isinstance(axis, np.ndarray):
            return self.rotate_with_matrix(
                rotation_matrix(theta, axis), wrt)
        if isinstance(axis, np.ndarray) and axis.shape == (3, 3):
            return self.rotate_with_matrix(theta, wrt)

        if wrt == 'local' or wrt == self:
            self.rot = rotate_matrix(self.rot, theta, axis)
            return self.newcoords(self.rot, self.pos)
        elif wrt == 'parent' or wrt == self.parent_link:
            self.rot = rotate_matrix(self.rot, theta, axis)
            return self.newcoords(self.rot, self.pos)
        else:
            return self.rotate_with_matrix(
                rotation_matrix(theta, axis), wrt)

    def transform(self, c, wrt='local'):
        if isinstance(wrt, Coordinates):
            raise NotImplementedError
        elif wrt == 'local' or wrt == self:  # multiply c from the left
            self = transform_coords(self, c)
        else:
            raise NotImplementedError
        return self.newcoords(self.rot, self.pos)

    def worldcoords(self):
        """Calculate rotation and position in the world."""
        if self._changed:
            if self.parent_link:
                self._worldcoords = transform_coords(
                    self.parent_link.worldcoords(),
                    self)
            else:
                self._worldcoords.rot = self.rot
                self._worldcoords.pos = self.pos
            self._changed = False
        return self._worldcoords

    def worldrot(self):
        return self.worldcoords().rot

    def worldpos(self):
        return self.worldcoords().pos

    def add_child(self, child_link):
        self.child_links.append(child_link)

    @property
    def parent(self):
        return self.parent_link


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

import copy

import numpy as np

from skrobot.dual_quaternion import DualQuaternion
from skrobot.math import _check_valid_rotation
from skrobot.math import _check_valid_translation
from skrobot.math import _wrap_axis
from skrobot.math import matrix2quaternion
from skrobot.math import matrix_log
from skrobot.math import normalize_vector
from skrobot.math import quaternion2matrix
from skrobot.math import quaternion_multiply
from skrobot.math import random_rotation
from skrobot.math import random_translation
from skrobot.math import rotate_matrix
from skrobot.math import rotation_angle
from skrobot.math import rotation_matrix
from skrobot.math import rpy2quaternion
from skrobot.math import rpy_angle


def transform_coords(c1, c2):
    translation = c1.translation + np.dot(c1.rotation, c2.translation)
    q = quaternion_multiply(c1.quaternion, c2.quaternion)
    return Coordinates(pos=translation, rot=q)


class Coordinates(object):

    def __init__(self,
                 pos=[0, 0, 0],
                 rot=np.eye(3),
                 name=None):
        """Initialization of Coordinates

        Parameters
        ----------
        pos : list or np.ndarray
            shape of (3,) translation vector
        rot : list or np.ndarray
            we can take 3x3 rotation matrix or
            [yaw, pitch, roll] or
            quaternion [w, x, y, z] order
        name : string or None
            name of this coordinates
        """
        self.rotation = rot
        self.translation = pos
        if name is None:
            name = ''
        self.name = name
        self.parent = None

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

        _check_valid_rotation(rotation)
        self._rotation = rotation * 1.

    @property
    def translation(self):
        return self._translation

    @translation.setter
    def translation(self, translation):
        # Convert lists to translation arrays
        if type(translation) in (list, tuple) and len(translation) == 3:
            translation = np.array([t for t in translation]).astype(np.float64)

        _check_valid_translation(translation)
        self._translation = translation.squeeze() * 1.


    @property
    def dimension(self):
        return len(self.translation)

    def changed(self):
        return False

    def translate(self, vec, wrt='local'):
        """translate this coordinates.

        unit is [m]
        """
        vec = np.array(vec, dtype=np.float64)
        return self.newcoords(
            self.rotation,
            self.parent_orientation(vec, wrt) + self.translation)

    def transform_vector(self, v):
        """"Return vector represented at world frame.

        Vector v given in the local coords is converted to world
        representation.

        """
        return np.matmul(self.rotation, v) + self.translation

    def inverse_transform_vector(self, vec):
        """Transform vector in world coordinates to local coordinates"""
        return np.matmul(self.rotation.T, vec) - \
            np.matmul(self.rotation.T, self.translation)

    def inverse_transformation(self, dest=None):
        """Return a invese transformation of this coordinate system.

        Create a new coordinate with inverse transformation of this
        coordinate system.

        """
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
                wrt == self.parent or \
                wrt == 'world':
            inv = transform_coords(c2, inv)
        elif isinstance(wrt, Coordinates):
            xw = wrt.worldcoords()
            inv = transform_coords(c2, inv)
            inv = transform_coords(xw.inverse_transformation(),
                                   inv)
            inv = transform_coords(inv, xw)
        else:
            raise ValueError('wrt {} not supported'.format(wrt))
        return inv

    def T(self):
        """Return 4x4 transformation matrix."""
        matrix = np.zeros((4, 4), dtype=np.float64)
        matrix[3, 3] = 1.0
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    @property
    def quaternion(self):
        return self._q

    @property
    def dual_quaternion(self):
        qr = normalize_vector(self.quaternion)
        x, y, z = self.translation
        qd = quaternion_multiply(np.array([0, x, y, z]), qr) * 0.5
        return DualQuaternion(qr, qd)

    def parent_orientation(self, v, wrt):
        if wrt == 'local' or wrt == self:
            return np.matmul(self.rotation, v)
        if wrt == 'parent' \
           or wrt == self.parent \
           or wrt == 'world':
            return v
        raise ValueError('wrt {} not supported'.format(wrt))

    def rotate_vector(self, v):
        return np.matmul(self.rotation, v)

    def inverse_rotate_vector(self, v):
        return np.matmul(v, self.rotation)

    def transform(self, c, wrt='local'):
        if wrt == 'local' or wrt == self:
            tmp_coords = transform_coords(self, c)
        elif wrt == 'parent' or wrt == self.parent \
                or wrt == 'world':
            tmp_coords = transform_coords(c, self)
        elif isinstance(wrt, Coordinates):
            tmp_coords = transform_coords(wrt.inverse_transformation, self)
            tmp_coords = transform_coords(c, tmp_coords)
            tmp_coords = transform_coords(wrt.worldcoords(), tmp_coords)
        else:
            raise ValueError('transform wrt {} is not supported'.format(wrt))
        return self.newcoords(tmp_coords)

    def rpy_angle(self):
        return rpy_angle(self.rotation)

    def axis(self, ax):
        ax = _wrap_axis(ax)
        return self.rotate_vector(ax)

    def difference_position(self, coords,
                            translation_axis=True):
        """Return differences in positoin of given coords.

        """
        dif_pos = self.inverse_transform_vector(coords.worldpos())
        translation_axis = _wrap_axis(translation_axis)
        dif_pos[translation_axis == 1] = 0.0
        return dif_pos

    def difference_rotation(self, coords,
                            rotation_axis=True):
        """Return differences in rotation of given coords.

        """
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
            if np.abs(np.linalg.norm(np.array(a0) - np.array(a1))) < 0.001:
                dif_rot = np.array([0, 0, 0], 'f')
            else:
                dif_rot = np.matmul(self.worldrot().T,
                                    np.arccos(np.dot(a0, a1)) *
                                    normalize_vector(np.cross(a0, a1)))
        elif rotation_axis in ['xx', 'yy', 'zz']:
            ax = rotation_axis[0]
            a0 = self.axis(ax)
            a2 = coords.axis(ax)
            if need_mirror_for_nearest_axis(self, coords, ax) is False:
                a2 = - a2
            if np.abs(np.linalg.norm(np.array(a0) - np.array(a1))) < 0.001:
                dif_rot = np.array([0, 0, 0], 'f')
            else:
                dif_rot = np.matmul(self.worldrot().T,
                                    np.arccos(np.dot(a0, a2)) *
                                    normalize_vector(np.cross(a0, a2)))
        elif rotation_axis is False or rotation_axis is None:
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
            rot = np.matmul(self.rotation, mat)
            self.newcoords(rot, self.translation)
        elif wrt == 'parent' or wrt == self.parent or \
                wrt == 'world' or wrt is None or \
                wrt == worldcoords:
            rot = np.matmul(mat, self.rotation)
            self.newcoords(rot, self.translation)
        elif isinstance(wrt, Coordinates):
            r2 = wrt.worldrot()
            r2t = r2.T
            r2t = np.matmul(mat, r2t)
            r2t = np.matmul(r2, r2t)
            self.rotation = np.matmul(r2t, self.rotation)
        else:
            raise ValueError('wrt {} is not supported'.format(wrt))
        return self

    def rotate(self, theta, axis=None, wrt='local'):
        if isinstance(axis, list) or isinstance(axis, np.ndarray):
            self.rotate_with_matrix(
                rotation_matrix(theta, axis), wrt)
        elif axis is None or axis is False:
            self.rotate_with_matrix(theta, wrt)
        elif wrt == 'local' or wrt == self:
            self.rotation = rotate_matrix(self.rotation, theta, axis)
        elif wrt == 'parent' or wrt == 'world':
            self.rotation = rotate_matrix(self.rotation, theta,
                                          axis, True)
        elif isinstance(wrt, Coordinates):  # C1'=C2*R*C2(-1)*C1
            self.rotate_with_matrix(
                rotation_matrix(theta, axis), wrt)
        else:
            raise ValueError('wrt {} not supported'.format(wrt))
        return self.newcoords(self.rotation, self.translation)

    def copy(self):
        return self.copy_coords()

    def copy_coords(self):
        """Returns a deep copy of the Coordinates."""
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
        return self.translation

    def newcoords(self, c, pos=None):
        """Update of coords is always done through newcoords."""
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
        self.worldrot()
        pos = self.worldpos()
        self.rpy = rpy_angle(self.rotation)[0]
        if self.name:
            prefix = self.__class__.__name__ + ':' + self.name
        else:
            prefix = self.__class__.__name__

        return "#<{0} {1} "\
            "{2:.3f} {3:.3f} {4:.3f} / {5:.1f} {6:.1f} {7:.1f}>".\
            format(prefix,
                   hex(id(self)),
                   pos[0],
                   pos[1],
                   pos[2],
                   self.rpy[0],
                   self.rpy[1],
                   self.rpy[2])


class CascadedCoords(Coordinates):

    def __init__(self, parent=None, *args, **kwargs):
        super(CascadedCoords, self).__init__(*args, **kwargs)
        self.manager = self
        self._changed = True
        self._descendants = []

        self.parent = parent
        if parent is not None:
            self.parent.assoc(self)
        self._worldcoords = Coordinates(pos=self.translation,
                                        rot=self.rotation)

    @property
    def descendants(self):
        return self._descendants

    def assoc(self, child, c=None):
        if not (child in self.descendants):
            if c is None:
                c = self.worldcoords().transformation(
                    child.worldcoords())
            child.obey(self)
            child.newcoords(c)
            self._descendants.append(child)
            return child

    def obey(self, mother):
        if self.parent is not None:
            self.parent.dissoc(self)
        self.parent = mother

    def dissoc(self, child):
        if child in self.descendants:
            c = child.worldcoords().copy_coords()
            self._descendants.remove(child)
            child.disobey(self)
            child.newcoords(c)

    def disobey(self, mother):
        if self.parent == mother:
            self.parent = None
        return self.parent

    def newcoords(self, c, pos=None):
        super(CascadedCoords, self).newcoords(c, pos)
        self.changed()
        return self

    def changed(self):
        if self._changed is False:
            self._changed = True
            return [c.changed() for c in self.descendants]
        return [False]

    def parentcoords(self):
        if self.parent:
            return self.parent.worldcoords()

    def transform_vector(self, v):
        return self.worldcoords().transform_vector(v)

    def inverse_transform_vector(self, v):
        return self.worldcoords().inverse_transform_vector(v)

    def rotate_with_matrix(self, matrix, wrt):
        if wrt == 'local' or wrt == self:
            self.rotation = np.dot(self.rotation, matrix)
            return self.newcoords(self.rotation, self.translation)
        elif wrt == 'parent' or wrt == self.parent:
            self.rotation = np.matmul(matrix, self.rotation)
            return self.newcoords(self.rotation, self.translation)
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
            return self.newcoords(self.rotation, self.translation)

    def rotate(self, theta, axis, wrt='local'):
        """Rotate this coordinate.

        Rotate this coordinate relative to axis by theta radians
        with respect to wrt.

        Parameters
        ----------
        theta : float
            radian
        axis : string or numpy.ndarray
            'x', 'y', 'z' or vector
        wrt : string or Coordinates

        Returns
        -------
        self
        """
        if isinstance(axis, list) or isinstance(axis, np.ndarray):
            return self.rotate_with_matrix(
                rotation_matrix(theta, axis), wrt)
        if isinstance(axis, np.ndarray) and axis.shape == (3, 3):
            return self.rotate_with_matrix(theta, wrt)

        if wrt == 'local' or wrt == self:
            self.rotation = rotate_matrix(self.rotation, theta, axis)
            return self.newcoords(self.rotation, self.translation)
        elif wrt == 'parent' or wrt == self.parent:
            self.rotation = rotate_matrix(self.rotation, theta, axis)
            return self.newcoords(self.rotation, self.translation)
        else:
            return self.rotate_with_matrix(
                rotation_matrix(theta, axis), wrt)

    def rotate_vector(self, v):
        return self.worldcoords().rotate_vector(v)

    def inverse_rotate_vector(self, v):
        return self.worldcoords().inverse_rotate_vector(v)

    def transform(self, c, wrt='local'):
        if isinstance(wrt, Coordinates):
            raise NotImplementedError
        elif wrt == 'local' or wrt == self:  # multiply c from the left
            tmp_coords = transform_coords(self, c)
            self.rotation = tmp_coords.rotation
            self.translation = tmp_coords.translation
        else:
            raise NotImplementedError
        return self.newcoords(self.rotation, self.translation)

    def worldcoords(self):
        """Calculate rotation and position in the world."""
        if self._changed:
            if self.parent:
                self._worldcoords = transform_coords(
                    self.parent.worldcoords(),
                    self)
            else:
                self._worldcoords.rotation = self.rotation
                self._worldcoords.translation = self.translation
            self.update()
            self._changed = False
        return self._worldcoords

    def worldrot(self):
        return self.worldcoords().rotation

    def worldpos(self):
        return self.worldcoords().translation

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, c):
        if not (c is None or coordinates_p(c)):
            raise ValueError('parent should be None or Coordinates. '
                             'get type=={}'.format(type(c)))
        self._parent = c

    def update(self):
        pass


def coordinates_p(x):
    return isinstance(x, Coordinates)


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


worldcoords = CascadedCoords(name='worldcoords')

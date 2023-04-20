from numbers import Number

import numpy as np

from skrobot.coordinates.math import quaternion_absolute_distance
from skrobot.coordinates.math import quaternion_multiply
from skrobot.coordinates.math import quaternion_normalize
from skrobot.coordinates.quaternion import Quaternion


class DualQuaternion(object):
    """Class for handling dual quaternions and their interpolations.

    Parameters
    ----------
    qr : list or numpy.ndarray
    qd : list or numpy.ndarray
        element of dual quaternion
    enforce_unit_norm : bool (optional)
        if True, norm should be 1.0.
    """

    def __init__(self,
                 qr=[1, 0, 0, 0],
                 qd=[0, 0, 0, 0],
                 enforce_unit_norm=False):
        if (isinstance(qd, list) or isinstance(qd, np.ndarray)) and \
           len(qd) == 3:
            x, y, z = qd
            qr = quaternion_normalize(qr)
            qd = 0.5 * quaternion_multiply([0, x, y, z], qr)
        self.qr = qr
        self.qd = qd

        if enforce_unit_norm:
            norm = self.norm
            if not np.allclose(norm[0], [1]):
                raise ValueError("Dual quaternoin's norm "
                                 'should be 1, but gives {}'.format(norm[0]))

    @property
    def translation(self):
        """Return translation of this dual quaternion.

        Returns
        -------
        q_translation.xyz : numpy.ndarray
            vector shape of (3, ). unit is [m]

        Examples
        --------
        >>> from skrobot.coordinates import Coordinates
        >>> c = Coordinates()
        >>> c.dual_quaternion.translation
        array([0., 0., 0.])
        >>> c.translate([0.1, 0.2, 0.3])
        >>> c.dual_quaternion.translation
        array([0.1, 0.2, 0.3])
        """
        dq = self.normalized
        q_rot = dq.qr
        if (q_rot.w < 0.0):
            q_rot = - q_rot
        q_translation = (2.0 * dq.qd) * dq.qr.conjugate
        return q_translation.xyz

    @property
    def rotation(self):
        """Return rotation matrix of this dual quaternion

        Returns
        -------
        dq.qr.rotation : numpy.ndarray
            3x3 rotation matrix

        Examples
        --------
        >>> import numpy as np
        >>> from skrobot.coordinates import Coordinates
        >>> c = Coordinates()
        >>> c.dual_quaternion.rotation
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        >>> c.rotate(np.pi / 2.0, 'y')
        >>> c.dual_quaternion.rotation
        array([[ 2.22044605e-16,  0.00000000e+00,  1.00000000e+00],
               [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
               [-1.00000000e+00,  0.00000000e+00,  2.22044605e-16]])
        """
        dq = self.normalized
        return dq.qr.rotation

    @property
    def quaternion(self):
        """Return this dual quaternion's qr (rotation)

        Returns
        -------
        dq.qr : skrobot.coordinates.quaternion.Quaternion
            rotation quaternion
        """
        dq = self.normalized
        return dq.qr

    @property
    def dq(self):
        """Return flatten vector of this dual quaternion

        Returns
        -------
        np.concatenate([self.qr.q, self.qd.q]) : numpy.ndarray
            (1x8) vector of this dual quaternion

        Examples
        --------
        >>> from skrobot.coordinates.dual_quaternion import DualQuaternion
        >>> dq = DualQuaternion()
        >>> dq.dq
        array([1., 0., 0., 0., 0., 0., 0., 0.])
        """
        return np.concatenate([self.qr.q, self.qd.q])

    @dq.setter
    def dq(self, dq):
        """Setter of dq

        Parameters
        ----------
        dq : numpy.ndarray
            (1x8) vector

        Examples
        --------
        >>> import numpy as np
        >>> from skrobot.coordinates.dual_quaternion import DualQuaternion
        >>> dq = DualQuaternion()
        >>> dq.dq
        array([1., 0., 0., 0., 0., 0., 0., 0.])
        >>> dq.dq = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        """
        self.qr = dq[0:4]
        self.qd = dq[4:8]

    @property
    def qr(self):
        """Return orientation

        Returns
        -------
        self._qr : numpy.narray
            [w, x, y, z] order
        """
        return self._qr

    @qr.setter
    def qr(self, qr_wxyz):
        """Setter of qr

        Parameters
        ----------
        qr_wxyz : list or numpy.ndarray or
                  skrobot.coordinates.quaternion.Quaternion
            new qr
        """
        if isinstance(qr_wxyz, Quaternion):
            self._qr = qr_wxyz
        else:
            self._qr = Quaternion(q=qr_wxyz)

    @property
    def qd(self):
        """Return translation quaternion

        Returns
        -------
        self._qd : skrobot.coordinates.quaternion.Quaternion
            quaternion indicating translation
        """
        return self._qd

    @qd.setter
    def qd(self, qd_wxyz):
        """Setter of qd

        Parameters
        ----------
        qr_wxyz : list or numpy.ndarray or
                  skrobot.coordinates.quaternion.Quaternion
            new qd
        """
        if isinstance(qd_wxyz, Quaternion):
            self._qd = qd_wxyz
        else:
            if len(qd_wxyz) == 3:
                qd_wxyz = [0, qd_wxyz[0], qd_wxyz[1], qd_wxyz[2]]
            elif len(qd_wxyz) != 4:
                raise ValueError
            self._qd = Quaternion(q=qd_wxyz)

    @property
    def conjugate(self):
        """Return conjugate of this dual quaternion

        Returns
        -------
        DualQuaternion : skrobot.coordinates.dual_quaternion.DualQuaternion
            new DualQuaternion class has this dual quaternion's conjugate
        """
        qr_c = self._qr.conjugate
        qd_c = self._qd.conjugate
        return DualQuaternion(qr_c, qd_c)

    @property
    def norm(self):
        """Return pair of norm of this dual quaternion

        Returns
        -------
        qr_norm, qd_norm : tuple(float, float)
            qr and qd's norm

        Examples
        --------
        >>> from skrobot.coordinates.dual_quaternion import DualQuaternion
        >>> dq = DualQuaternion()
        >>> dq.norm
        (1.0, 0.0)
        """
        qr_norm = self.qr.norm
        qd_norm = np.dot(self.qr.q, self.qd.q) / qr_norm
        return (qr_norm, qd_norm)

    def normalize(self):
        """Normalize this dual quaternion

        Note that this function changes property.

        Returns
        -------
        self : skrobot.coordinates.dual_quaternion.DualQuaternion
            return self
        """
        real_norm = self.qr.norm
        self.qr = self.qr / real_norm
        self.qd = self.qd / real_norm
        return self

    @property
    def normalized(self):
        """Return normalized this dual quaternion

        Returns
        -------
        DualQuaternion(qr, qd, True) :
                skrobot.coordinates.dual_quaternion.DualQuaternion
            normalized dual quaternion
        """
        real_norm = self.qr.norm
        qr = self.qr / real_norm
        qd = self.qd / real_norm
        return DualQuaternion(qr, qd, True)

    @property
    def scalar(self):
        """The scalar part of the dual quaternion.

        Returns
        -------
        scalar : float
            scalar
        """
        scalar = (self + self.conjugate) * 0.5
        return scalar

    def copy(self):
        """Return a copy of this quaternion.

        Returns
        -------
        : DualQuaternion
            copied DualQuaternion instance
        """
        return DualQuaternion(self.qr.copy(), self.qd.copy())

    @staticmethod
    def interpolate(dq0, dq1, t):
        """Return interpolated dual quaternion

        Parameters
        ----------
        dq0 : skrobot.coordinates.dual_quaternion.DualQuaternion
        dq1 : skrobot.coordinates.dual_quaternion.DualQuaternion
            dual quaternion
        t : float
            ratio of interpolation. Must be 0 <= t <= 1.0.
        """
        if not 0 <= t <= 1:
            raise ValueError('Interpolation step must be between 0 and 1, '
                             'but gives {}'.format(t))

        dqt = dq0 * (1 - t) + dq1 * t
        return dqt.normalized

    def enforce_positive_q_rot_w(self):
        assert self.norm[0] > 1e-8
        q_rot_w = self.qr.w
        if q_rot_w < 0.0:
            self.dq = -self.dq

    @property
    def axis(self):
        """Return axis of this dual quaternion

        Returns
        -------
        self.qr.axis : numpy.ndarray
            this dual quaternion's axis.
            See See skrobot.coordinates.quaternion.Quaternion.axis.
        """
        return self.qr.axis

    @property
    def angle(self):
        """Return rotation angle of this dual quaternion

        Returns
        -------
        self.qr.angle : float
            this dual quaternion's rotation angle with respect to self.axis.
            See skrobot.coordinates.quaternion.Quaternion.angle.
        """
        return self.qr.angle

    def screw_axis(self):
        """Return screw axis

        Calculates rotation, translation and screw axis from dual
        quaternion.

        Returns
        -------
        screw_axis, theta, translation : tuple(numpy.ndarray, float, float)
            screw axis of this dual quaternion.
            rotation angle in radian.
            translation
        """
        qr_w = self.qr.w
        theta = 2.0 * np.arccos(qr_w)
        theta = np.mod(theta, np.pi * 2.0)

        qd_w = self.qd.w
        if theta > 1.0e-12:
            s = np.sin(theta / 2.0)
            translation = -2.0 * qd_w / s
            screw_axis = self.qr.xyz / s
        else:
            translation = 2.0 * np.sqrt(np.sum(self.qd.xyz ** 2))
            if translation > 1.0e-12:
                screw_axis = 2.0 * self.qd.xyz / translation
            else:
                screw_axis = np.zeros(3, dtype=np.float64)
        return screw_axis, theta, translation

    @property
    def inverse(self):
        """Return inverse of this dual quaternion

        Returns
        -------
        dq : skrobot.coordinates.dual_quaternion.DualQuaternion
            new DualQuaternion class has inverse of this dual quaternion
        """
        if self.norm[0] < 1.0e-8:
            return None
        inv_qr = self.qr.inverse
        return DualQuaternion(
            inv_qr, - inv_qr * self.qd * inv_qr)

    def T(self):
        """Return 4x4 homogeneous transformation matrix.

        Returns
        -------
        matrix : numpy.ndarray
            homogeneous transformation matrix shape of (4, 4)

        Examples
        --------
        >>> from numpy import pi
        >>> from skrobot.coordinates import Coordinates
        >>> from skrobot.coordinates.dual_quaternion import DualQuaternion
        >>> dq = DualQuaternion()
        >>> dq.T()
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
        >>> dq = Coordinates().rotate(pi / 2.0, 'y').\
        ...                    translate((0.1, 0.2, 0.3)).\
        ...                    dual_quaternion
        >>> dq.T()
        array([[ 2.22044605e-16,  0.00000000e+00,  1.00000000e+00,
                 3.00000000e-01],
               [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
                 2.00000000e-01],
               [-1.00000000e+00,  0.00000000e+00,  2.22044605e-16,
                -1.00000000e-01],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                 1.00000000e+00]])
        """
        matrix = np.zeros((4, 4), dtype=np.float64)
        matrix[3, 3] = 1.0
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def __add__(self, val):
        if not isinstance(val, DualQuaternion):
            raise TypeError('Cannot add dual quaternion with '
                            'object of type {}'.format(type(val)))
        dq = DualQuaternion()
        dq.dq = self.dq + val.dq
        return dq

    def __mul__(self, val):
        if isinstance(val, DualQuaternion):
            new_qr = self._qr * val._qr
            new_qd = self._qr * val._qd + self._qd * val._qr
            return DualQuaternion(new_qr, new_qd)
        elif isinstance(val, Number):
            new_qr = val * self.qr
            new_qd = val * self.qd
            return DualQuaternion(new_qr, new_qd, False)
        raise TypeError('Cannot multiply dual quaternion '
                        'with object of type {}'.format(type(val)))

    def __rmul__(self, val):
        return self.__mul__(val)

    def __str__(self):
        return '{0}+{1}e'.format(self.qr, self.qd)

    def __repr__(self):
        return 'DualQuaternion({0},{1})'.format(
            repr(self.qr), repr(self.qd))

    def difference_position(self, other_dq):
        """Return difference position

        Parameters
        ----------
        other_dq : skrobot.coordinates.dual_quaternion.DualQuaternion
            dual quaternion

        Returns
        -------
        dif_pos : float
            difference position's norm
        """
        trans = self.qd * self.qr.conjugate
        other_trans = other_dq.qd * other_dq.qr.conjugate
        return 2.0 * np.linalg.norm(trans.xyz - other_trans.xyz, ord=2)

    def difference_rotation(self, other_dq):
        """Return difference rotation distance

        Parameters
        ----------
        other_dq : skrobot.coordinates.dual_quaternion.DualQuaternion
            dual quaternion

        Returns
        -------
        dif_rot : float
            angle distance in radian.
        """
        return quaternion_absolute_distance(
            self.qr.q,
            other_dq.qr.q)

    def pose(self):
        """Return [x, y, z, wx, wy, wz, wq] elements.

        Returns
        -------
        pose : numpy.ndarray
            [x, y, z, wx, wy, wz, wq] pose
        """
        self.normalize()
        dq = self

        pose = np.zeros(7, dtype=np.float64)
        q_rot = dq.qr
        if (q_rot.w < 0.0):
            q_rot = -q_rot
        q_translation = 2.0 * dq.qd * dq.qr.conjugate

        pose[0:3] = q_translation.xyz.copy()
        pose[3:7] = q_rot.q.copy()
        return pose.copy()

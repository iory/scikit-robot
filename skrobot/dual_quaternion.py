from numbers import Number

import numpy as np

from skrobot.math import quaternion_multiply
from skrobot.math import quaternion_normalize
from skrobot.quaternion import Quaternion


class DualQuaternion(object):
    """
    Class for handling dual quaternions and their interpolations.

    Attributes
    ----------
    qr : :obj:`numpy.ndarray` of float
        A 4-entry quaternion in wxyz format.

    qd : :obj:`numpy.ndarray` of float
        A 4-entry quaternion in wxyz format.
        to represent the translation.

    conjugate : :obj:`DualQuaternion`
        The conjugate of this DualQuaternion.

    norm : :obj:`tuple` of :obj:`numpy.ndarray`
        The normalized vectors for qr and qd, respectively.

    normalized : :obj:`DualQuaternion`
        This quaternion with qr normalized.
    """

    def __init__(self,
                 qr=[1, 0, 0, 0],
                 qd=[0, 0, 0, 0],
                 enforce_unit_norm=False):
        """
        Initialize a dual quaternion.

        Parameters
        ----------

        """
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
                                 "should be 1, but gives {}".format(norm[0]))

    @property
    def translation(self):
        dq = self.normalized
        q_rot = dq.qr
        if (q_rot.w < 0.0):
            q_rot = - q_rot
        q_translation = (2.0 * dq.qd) * dq.qr.conjugate
        return q_translation.xyz

    @property
    def rotation(self):
        dq = self.normalized
        return dq.qr.rotation

    @property
    def quaternion(self):
        dq = self.normalized
        return dq.qr

    @property
    def dq(self):
        return np.concatenate([self.qr.q, self.qd.q])

    @dq.setter
    def dq(self, dq):
        self.qr = dq[0:4]
        self.qd = dq[4:8]

    @property
    def qr(self):
        """
        Returns
        -------
        self._qr : np.narray
            [w, x, y, z] order
        """
        return self._qr

    @qr.setter
    def qr(self, qr_wxyz):
        if isinstance(qr_wxyz, Quaternion):
            self._qr = qr_wxyz
        else:
            self._qr = Quaternion(q=qr_wxyz)

    @property
    def qd(self):
        """
        Returns
        -------
        self._qd : np.narray
            [w, x, y, z] order
        """
        return self._qd

    @qd.setter
    def qd(self, qd_wxyz):
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
        qr_c = self._qr.conjugate
        qd_c = self._qd.conjugate
        return DualQuaternion(qr_c, qd_c)

    @property
    def norm(self):
        qr_norm = self.qr.norm
        qd_norm = np.dot(self.qr, self.qd) / qr_norm
        return (qr_norm, qd_norm)

    def normalize(self):
        real_norm = self.qr.norm
        self.qr = self.qr / real_norm
        self.qd = self.qd / real_norm
        return self

    @property
    def normalized(self):
        real_norm = self.qr.norm
        qr = self.qr / real_norm
        qd = self.qd / real_norm
        return DualQuaternion(qr, qd, True)

    @property
    def scalar(self):
        """

        The scalar part of the dual quaternion.

        """
        scalar = (self + self.conjugate) * 0.5
        return scalar

    def copy(self):
        """
        Return a copy of this quaternion.

        Returns
        -------
        : DualQuaternion
            copied DualQuaternion instance
        """
        return DualQuaternion(self.qr.copy(), self.qd.copy())

    @staticmethod
    def interpolate(dq0, dq1, t):
        if not 0 <= t <= 1:
            raise ValueError("Interpolation step must be between 0 and 1, "
                             "but gives {}".format(t))

        dqt = dq0 * (1 - t) + dq1 * t
        return dqt.normalized

    def enforce_positive_q_rot_w(self):
        assert(self.norm[0] > 1e-8)
        q_rot_w = self.qr.w
        if q_rot_w < 0.0:
            self.dq = -self.dq

    @property
    def axis(self):
        return self.qr.axis

    @property
    def angle(self):
        return self.qr.angle

    def screw_axis(self):
        """

        Calculates rotation, translation and screw axis from dual quaternion.

        Returns:
        screw_axis (~numpy.ndarray) : screw axis of this dual quaternion
        theta (~float) : radian
        translation (~float) :

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

    def inverse(self):
        if self.norm[0] < 1.0e-8:
            return None
        inv_qr = self.qr.inverse()
        return DualQuaternion(
            inv_qr, - inv_qr * self.qd * inv_qr)

    def T(self):
        """Return 4x4 transformation matrix"""
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
        return "DualQuaternion({0},{1})".format(
            repr(self.qr), repr(self.qd))

    def pose(self):
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

from numbers import Number

import numpy as np

from robot.math import quaternion_conjugate
from robot.math import quaternion_multiply


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
                 enforce_unit_norm=True):
        """
        Initialize a dual quaternion.

        Parameters
        ----------

        """
        self.qr = qr
        self.qd = qd

        if enforce_unit_norm:
            norm = self.norm
            if not np.allclose(norm[0], [1]):
                raise ValueError("Dual quaternoin's norm "
                                 "should be 1, but gives {}".format(norm[0]))

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
        self._qr = np.array(qr_wxyz, dtype=np.float64)

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
        if len(qd_wxyz) == 3:
            qd_wxyz = [0, qd_wxyz[0], qd_wxyz[1], qd_wxyz[2]]
        elif len(qd_wxyz) != 4:
            raise ValueError
        if qd_wxyz[0] != 0:
            raise ValueError('First value of Qd must be 0, but gives {}'.
                             format(qd_wxyz))
        self._qd = np.array(qd_wxyz, dtype=np.float64)

    @property
    def conjugate(self):
        qr_c = quaternion_conjugate(self._qr)
        qd_c = quaternion_conjugate(self._qd)
        return DualQuaternion(qr_c, qd_c)

    @property
    def norm(self):
        qr_c = quaternion_conjugate(self._qr)
        qd_c = quaternion_conjugate(self._qd)

        qr_norm = np.linalg.norm(quaternion_multiply(self._qr, qr_c))
        qd_norm = np.linalg.norm(quaternion_multiply(self._qr, qd_c) +
                                 quaternion_multiply(self._qd, qr_c))

        return (qr_norm, qd_norm)

    @property
    def normalized(self):
        qr = self.qr / 1.0 / np.linalg.norm(self.qr)
        return DualQuaternion(qr, self.qd, True)

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

    def screw_axis(self):
        """

        Calculates rotation, translation and screw axis from dual quaternion.

        """
        qr_w = self.qr[0]
        rotation = 2.0 * np.rad2deg(np.arccos(qr_w))
        rotation = np.mod(rotation, 360.0)

        qd_w = self.qd[0]
        if rotation > 1.0e-12:
            s = np.sin(rotation / 2.0 * np.pi / 180.0)
            translation = -2.0 * qd_w / s
            screw_axis = self.qr[1:] / s
        else:
            translation = 2.0 * np.sqrt(np.sum(self.qd[1:] ** 2))
            if translation > 1.0e-12:
                screw_axis = 2.0 * self.qd[1:] / translation
            else:
                screw_axis = np.zeros(3, dtype=np.float64)
        return screw_axis, rotation, translation

    def __add__(self, val):
        if not isinstance(val, DualQuaternion):
            raise TypeError('Cannot add dual quaternion with '
                            'object of type {}'.format(type(val)))

        new_qr = self.qr + val.qr
        new_qd = self.qd + val.qd
        new_qr = new_qr / np.linalg.norm(new_qr)

        return DualQuaternion(new_qr, new_qd, False)

    def __mul__(self, val):
        if isinstance(val, DualQuaternion):
            new_qr = quaternion_multiply(self._qr, val._qr)
            new_qd = quaternion_multiply(self._qr, val._qd) + \
                quaternion_multiply(self._qd, val._qr)
            return DualQuaternion(new_qr, new_qd)
        elif isinstance(val, Number):
            new_qr = val * self.qr
            new_qd = val * self.qd
            return DualQuaternion(new_qr, new_qd, False)
        raise TypeError('Cannot multiply dual quaternion '
                        'with object of type {}'.format(type(val)))

    def __str__(self):
        return '{0}+{1}e'.format(self.qr, self.qd)

    def __repr__(self):
        return "DualQuaternion({0},{1})".format(
            repr(self.qr), repr(self.qd))

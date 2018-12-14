from numbers import Number

import numpy as np

from robot.math import quaternion_inverse
from robot.math import quaternion_multiply
from robot.math import quaternion_norm


class Quaternion(object):

    def __init__(self,
                 w=1.0,
                 x=0.0,
                 y=0.0,
                 z=0.0,
                 q=None):
        if q is None:
            self.q = [w, x, y, z]
        else:
            self.q = q

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, quaternion):
        quaternion = np.array(quaternion, dtype=np.float32)
        if not (quaternion.shape == (4,)):
            raise ValueError("quaternion should be of shape (4,)."
                             " get {}".format(quaternion.shape))
        self._q = quaternion

    @property
    def x(self):
        return self.q[1]

    @property
    def y(self):
        return self.q[2]

    @property
    def z(self):
        return self.q[3]

    @property
    def w(self):
        return self.q[0]

    def norm(self):
        return quaternion_norm(self.q)

    def normalize(self):
        """Normalize this quaternion"""
        norm = self.norm()
        if norm > 1e-8:
            self.q = self.q / norm

    def normalized(self):
        """Return Normalized quaternion"""
        norm = self.norm()
        q = self.q.copy()
        if norm > 1e-8:
            q = q / norm
        return Quaternion(q=q)

    def copy(self):
        return Quaternion(q=self.q.copy())

    def conjugate(self):
        new_q = [self.w, -self.x, -self.y, -self.z]
        return Quaternion(q=new_q)

    def inverse(self):
        return Quaternion(q=quaternion_inverse(self.q))

    def __add__(self, cls):
        new_q = self.q + cls.q
        return Quaternion(q=new_q)

    def __sub__(self, cls):
        new_q = self.q - cls.q
        return Quaternion(q=new_q)

    def __mul__(self, cls):
        if isinstance(cls, Quaternion):
            q = quaternion_multiply(self.q, cls.q)
            return Quaternion(q=q)
        elif isinstance(cls, Number):
            q = self.q.copy()
            return Quaternion(q=q * cls)
        else:
            raise TypeError("Quaternion's multiplication is only supported "
                            "Number or Quaternion. get {}".format(type(cls)))

    def __rmul__(self, cls):
        if isinstance(cls, Number):
            q = self.q.copy()
            return Quaternion(q=cls * q)
        else:
            raise TypeError("Quaternion's multiplication is only supported "
                            "Number or Quaternion. get {}".format(type(cls)))

    def __truediv__(self, cls):
        if isinstance(cls, Quaternion):
            return self * cls.inverse()
        elif isinstance(cls, Number):
            q = self.q.copy()
            return Quaternion(q=q / cls)
        else:
            raise TypeError("Quaternion's division is only supported "
                            "Number or Quaternion. get {}".format(type(cls)))

    def __div__(self, cls):
        return self.__truediv__(cls)

    def __neg__(self):
        return Quaternion(q=-self.q)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        prefix = self.__class__.__name__

        return '#<{} {} w: {} x: {} y: {} z: {}>'.format(
            prefix,
            hex(id(self)),
            self.w,
            self.x,
            self.y,
            self.z)

from numbers import Number

import numpy as np

from skrobot.coordinates.math import normalize_vector
from skrobot.coordinates.math import quaternion2matrix
from skrobot.coordinates.math import quaternion_inverse
from skrobot.coordinates.math import quaternion_multiply
from skrobot.coordinates.math import quaternion_norm


class Quaternion(object):

    """Class for handling Quaternion.

    Parameters
    ----------
    w : float or numpy.ndarray
    x : float
    y : float
    z : float
    q : None or numpy.ndarray
        if q is not specified, use w, x, y, z.

    Examples
    --------
    >>> from skrobot.coordinates.quaternion import Quaternion
    >>> q = Quaternion()
    >>> q
    #<Quaternion 0x1283bde48 w: 1.0 x: 0.0 y: 0.0 z: 0.0>
    >>> q = Quaternion([1, 2, 3, 4])
    >>> q
    #<Quaternion 0x1283cad68 w: 1.0 x: 2.0 y: 3.0 z: 4.0>
    >>> q = Quaternion(q=[1, 2, 3, 4])
    >>> q
    #<Quaternion 0x1283bd438 w: 1.0 x: 2.0 y: 3.0 z: 4.0>
    >>> q = Quaternion(1, 2, 3, 4)
    >>> q
    #<Quaternion 0x128400198 w: 1.0 x: 2.0 y: 3.0 z: 4.0>
    >>> q = Quaternion(w=0.0, x=1.0, y=0.0, z=0.0)
    >>> q
    #<Quaternion 0x1283cc2e8 w: 0.0 x: 1.0 y: 0.0 z: 0.0>
    """
    def __init__(self,
                 w=1.0,
                 x=0.0,
                 y=0.0,
                 z=0.0,
                 q=None):
        if q is None:
            if (isinstance(w, list) or isinstance(w, np.ndarray)) and \
               len(w) == 4:
                self.q = w
            else:
                self.q = [w, x, y, z]
        else:
            self.q = q

    @property
    def q(self):
        """Return quaternion

        Returns
        -------
        self._q : numpy.ndarray
            [w, x, y, z] quaternion

        Examples
        --------
        >>> from skrobot.coordinates.quaternion import Quaternion
        >>> q = Quaternion()
        >>> q.q
        array([1., 0., 0., 0.])
        >>> q = Quaternion(w=0.0, x=1.0, y=0.0, z=0.0)
        >>> q.q
        array([0., 1., 0., 0.])
        """
        return self._q

    @q.setter
    def q(self, quaternion):
        """Set quaternion

        Parameters
        ----------
        quaternion : list or numpy.ndarray
            [w, x, y, z] quaternion

        Examples
        --------
        >>> from skrobot.coordinates.quaternion import Quaternion
        >>> q = Quaternion()
        >>> q
        #<Quaternion 0x1267d7198 w: 1.0 x: 0.0 y: 0.0 z: 0.0>
        >>> q.q = [0.0, 1.0, 0.0, 0.0]
        >>> q
        #<Quaternion 0x1267d7198 w: 0.0 x: 1.0 y: 0.0 z: 0.0>
        """
        quaternion = np.array(quaternion, dtype=np.float64)
        if not (quaternion.shape == (4,)):
            raise ValueError('quaternion should be of shape (4,).'
                             ' get {}'.format(quaternion.shape))
        self._q = quaternion

    @property
    def x(self):
        """Return x element

        Returns
        -------
        self.q[1] : float
            x element of this quaternion
        """
        return self.q[1]

    @property
    def y(self):
        """Return y element

        Returns
        -------
        self.q[2] : float
            y element of this quaternion
        """
        return self.q[2]

    @property
    def z(self):
        """Return z element

        Returns
        -------
        self.q[3] : float
            z element of this quaternion
        """
        return self.q[3]

    @property
    def w(self):
        """Return w element

        Returns
        -------
        self.q[0] : float
            w element of this quaternion
        """
        return self.q[0]

    @property
    def xyz(self):
        """Return xyz vector of this quaternion

        Returns
        -------
        quaternion_xyz : numpy.ndarray
            xyz elements of this quaternion
        """
        return self.q[1:]

    @property
    def rotation(self):
        """Return rotation matrix.

        Note that this property internally normalizes quaternion.

        Returns
        -------
        quaternion2matrix(self.q) : numpy.ndarray
            3x3 rotation matrix

        Examples
        --------
        >>> from skrobot.coordinates.quaternion import Quaternion
        >>> q = Quaternion()
        >>> q
        #<Quaternion 0x12f1aa6a0 w: 1.0 x: 0.0 y: 0.0 z: 0.0>
        >>> q.rotation
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        >>> q.q = [0, 1, 0, 0]
        >>> q
        #<Quaternion 0x12f1aa6a0 w: 0.0 x: 1.0 y: 0.0 z: 0.0>
        >>> q.rotation
        array([[ 1.,  0.,  0.],
               [ 0., -1.,  0.],
               [ 0.,  0., -1.]])
        >>> q.q = [1, 2, 3, 4]
        >>> q
        #<Quaternion 0x12f1aa6a0 w: 1.0 x: 2.0 y: 3.0 z: 4.0>
        >>> q.rotation
        array([[-0.66666667,  0.13333333,  0.73333333],
               [ 0.66666667, -0.33333333,  0.66666667],
               [ 0.33333333,  0.93333333,  0.13333333]])
        """
        return quaternion2matrix(self.normalized.q)

    @property
    def axis(self):
        """Return axis of this quaternion.

        Note that this property return normalized axis.

        Returns
        -------
        axis : numpy.ndarray
            normalized axis vector
        """
        if self.w > 1.0:
            q = self.normalized
        else:
            q = self

        # quaternion is normalized
        # q.w is less than 1.0, so term always positive.
        s = np.sqrt(1 - q.w ** 2)

        if s < 0.001:
            axis = q.xyz
        else:
            axis = q.xyz / s
        axis = normalize_vector(axis)
        return axis

    @property
    def angle(self):
        """Return rotation angle of this quaternion

        Returns
        -------
        theta : float
            rotation angle with respect to self.axis
        """
        q = self.normalized
        theta = 2.0 * np.arccos(q.w)
        return theta

    @property
    def norm(self):
        """Return norm of this quaternion

        Returns
        -------
        quaternion_norm(self.q) : float
            norm of this quaternion

        Examples
        --------
        >>> from skrobot.coordinates.quaternion import Quaternion
        >>> q = Quaternion()
        >>> q.norm
        1.0
        >>> q = Quaternion([1, 2, 3, 4])
        >>> q.norm
        5.477225575051661
        >>> q.normalized.norm
        0.9999999999999999
        """
        return quaternion_norm(self.q)

    def normalize(self):
        """Normalize this quaternion.

        Note that this function changes wxyz property.

        Examples
        --------
        >>> from skrobot.coordinates.quaternion import Quaternion
        >>> q = Quaternion([1, 2, 3, 4])
        >>> q.q
        array([1., 2., 3., 4.])
        >>> q.normalize()
        >>> q.q
        array([0.18257419, 0.36514837, 0.54772256, 0.73029674])
        """
        norm = self.norm
        if norm > 1e-8:
            self.q = self.q / norm

    @property
    def normalized(self):
        """Return Normalized quaternion.

        Returns
        -------
        normalized quaternion : skrobot.coordinates.quaternion.Quaternion
            return quaternion which is norm == 1.0.

        Examples
        --------
        >>> from skrobot.coordinates.quaternion import Quaternion
        >>> q = Quaternion([1, 2, 3, 4])
        >>> normalized_q = q.normalized
        >>> normalized_q.q
        array([0.18257419, 0.36514837, 0.54772256, 0.73029674])
        >>> q.q
        array([1., 2., 3., 4.])
        """
        norm = self.norm
        q = self.q.copy()
        if norm > 1e-8:
            q = q / norm
        return Quaternion(q=q)

    def copy(self):
        """Return copy of this Quaternion

        Returns
        -------
        Quaternion(q=self.q.copy()) : skrobot.coordinates.quaternion.Quaternion
            copy of this quaternion
        """
        return Quaternion(q=self.q.copy())

    @property
    def conjugate(self):
        """Return conjugate of this quaternion

        Returns
        -------
        Quaternion : skrobot.coordinates.quaternion.Quaternion
            new Quaternion class has this quaternion's conjugate

        Examples
        --------
        >>> from skrobot.coordinates.quaternion import Quaternion
        >>> q = Quaternion()
        >>> q.conjugate
        #<Quaternion 0x12f2dfb38 w: 1.0 x: -0.0 y: -0.0 z: -0.0>
        >>> q.q = [0, 1, 0, 0]
        >>> q.conjugate
        #<Quaternion 0x12f303c88 w: 0.0 x: -1.0 y: -0.0 z: -0.0>
        """
        new_q = [self.w, -self.x, -self.y, -self.z]
        return Quaternion(q=new_q)

    @property
    def inverse(self):
        """Return inverse of this quaternion

        Returns
        -------
        q : skrobot.coordinates.quaternion.Quaternion
            new Quaternion class has inverse of this quaternion

        Examples
        --------
        >>> from skrobot.coordinates.quaternion import Quaternion
        >>> q = Quaternion()
        >>> q
        #<Quaternion 0x127e6da58 w: 1.0 x: 0.0 y: 0.0 z: 0.0>
        >>> q.inverse
        #<Quaternion 0x1281bbda0 w: 1.0 x: -0.0 y: -0.0 z: -0.0>
        >>> q.q = [0, 1, 0, 0]
        >>> q.inverse
        #<Quaternion 0x1282b0cc0 w: 0.0 x: -1.0 y: -0.0 z: -0.0>
        """
        return Quaternion(q=quaternion_inverse(self.q))

    def T(self):
        """Return 4x4 homogeneous transformation matrix.

        Returns
        -------
        matrix : numpy.ndarray
            homogeneous transformation matrix shape of (4, 4)

        Examples
        --------
        >>> from skrobot.coordinates.quaternion import Quaternion
        >>> q = Quaternion()
        >>> q.T()
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
        >>> q.q = [1, 2, 3, 4]
        >>> q.T()
        array([[-0.66666667,  0.13333333,  0.73333333,  0.        ],
               [ 0.66666667, -0.33333333,  0.66666667,  0.        ],
               [ 0.33333333,  0.93333333,  0.13333333,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])
        """
        matrix = np.zeros((4, 4), dtype=np.float64)
        matrix[3, 3] = 1.0
        matrix[:3, :3] = self.rotation
        return matrix

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
                            'Number or Quaternion. get {}'.format(type(cls)))

    def __rmul__(self, cls):
        if isinstance(cls, Number):
            q = self.q.copy()
            return Quaternion(q=cls * q)
        else:
            raise TypeError("Quaternion's multiplication is only supported "
                            'Number or Quaternion. get {}'.format(type(cls)))

    def __truediv__(self, cls):
        if isinstance(cls, Quaternion):
            return self * cls.inverse
        elif isinstance(cls, Number):
            q = self.q.copy()
            return Quaternion(q=q / cls)
        else:
            raise TypeError("Quaternion's division is only supported "
                            'Number or Quaternion. get {}'.format(type(cls)))

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

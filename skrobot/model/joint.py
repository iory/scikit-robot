import numpy as np
from logging import getLogger

from skrobot.coordinates import _wrap_axis
from skrobot.coordinates import normalize_vector

_default_max_joint_velocity = 1.0
_default_max_joint_torque = 1.0

logger = getLogger(__name__)


def calc_angle_speed_gain_scalar(joint, dav, i, periodic_time):
    """Calculate angle speed gain

    Parameters
    ----------
    joint_list : list[skrobot.model.Joint]

    Returns
    -------
    n : int
        total Degrees of Freedom
    """
    if dav[i] == 0 or periodic_time == 0:
        dav_gain = 1.0
    else:
        dav_gain = abs(joint.max_joint_velocity / (dav[i] / periodic_time))
    return min(dav_gain, 1.0)


def calc_angle_speed_gain_vector(joint, dav, i, periodic_time):
    """Calculate angle speed gain for multiple Degrees of Freedom

    """
    if periodic_time == 0:
        return 1.0
    dav_gain = 1.0
    for joint_dof_index in range(joint.joint_dof):
        if dav[i + joint_dof_index] == 0:
            dav_gain = min(dav_gain, 1.0)
        else:
            tmp_gain = abs(joint.max_joint_velocity[joint_dof_index]
                           / (dav[i + joint_dof_index] / periodic_time))
            dav_gain = min(tmp_gain, dav_gain)
    return dav_gain


def calc_target_joint_dimension(joint_list):
    """Calculate Total Degrees of Freedom from joint list

    Parameters
    ----------
    joint_list : list[skrobot.model.Joint]

    Returns
    -------
    n : int
        total Degrees of Freedom
    """
    n = 0
    for j in joint_list:
        n += j.joint_dof
    return n


def calc_target_joint_dimension_from_link_list(link_list):
    """Calculate Total Degrees of Freedom from link list

    Parameters
    ----------
    link_list : list[skrobot.model.Link]

    Returns
    -------
    n : int
        total Degrees of Freedom
    """
    n = 0
    for link in link_list:
        if hasattr(link, 'joint'):
            n += link.joint.joint_dof
    return n


def calc_dif_with_axis(dif, axis):
    """Return diff with respect to axis.

    Parameters
    ----------
    dif : list[float] or numpy.ndarray
        difference vector
    axis : str or bool or None
        if axis is False or None, return numpy.array([]).
        if axis is True, return dif.

    Returns
    -------
    ret : numpy.ndarray
        difference with respect to axis.
    """
    if axis in ['x', 'xx']:
        ret = np.array([dif[1], dif[2]])
    elif axis in ['y', 'yy']:
        ret = np.array([dif[0], dif[2]])
    elif axis in ['z', 'zz']:
        ret = np.array([dif[0], dif[1]])
    elif axis in ['xy', 'yx']:
        ret = np.array([dif[2]])
    elif axis in ['yz', 'zy']:
        ret = np.array([dif[0]])
    elif axis in ['zx', 'xz']:
        ret = np.array([dif[1]])
    elif axis is None or axis is False:
        ret = np.array([])
    elif axis in ['xm', 'ym', 'zm']:
        ret = dif
    elif axis is True:
        ret = dif
    else:
        raise ValueError('axis {} is not supported'.format(axis))
    return ret


class Joint(object):

    def __init__(self, name=None, child_link=None,
                 parent_link=None,
                 min_angle=-np.pi / 2.0,
                 max_angle=np.pi,
                 max_joint_velocity=None,
                 max_joint_torque=None,
                 joint_min_max_table=None,
                 joint_min_max_target=None,
                 hooks=None):
        hooks = hooks or []
        self.name = name
        self.parent_link = parent_link
        self.child_link = child_link
        self.min_angle = min_angle
        self.max_angle = max_angle
        if max_joint_velocity is None:
            max_joint_velocity = _default_max_joint_velocity
        if max_joint_torque is None:
            max_joint_torque = _default_max_joint_torque
        self.max_joint_velocity = max_joint_velocity
        self.joint_min_max_table = joint_min_max_table
        self.joint_min_max_target = joint_min_max_target
        self.default_coords = self.child_link.copy_coords()
        self._hooks = hooks

    @property
    def joint_dof(self):
        raise NotImplementedError

    def joint_angle(self, v=None, relative=None, enable_hook=True):
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.name:
            prefix = self.__class__.__name__ + \
                ' ' + hex(id(self)) + ' ' + self.name
        else:
            prefix = self.__class__.__name__ + ' ' + hex(id(self))

        return '#<%s>' % prefix

    def register_hook(self, hook):
        self._hooks.append(hook)

    def register_mimic_joint(self, joint, multiplier, offset):
        self.register_hook(
            lambda: joint.joint_angle(
                self.joint_angle() * multiplier + offset))


class RotationalJoint(Joint):

    def __init__(self, axis='z',
                 max_joint_velocity=np.deg2rad(5),
                 max_joint_torque=100,
                 *args, **kwargs):
        super(RotationalJoint, self).__init__(
            max_joint_velocity=max_joint_velocity,
            max_joint_torque=max_joint_torque,
            *args, **kwargs)
        if isinstance(axis, str):
            if axis == 'z':
                self.axis = np.array([0, 0, 1], dtype=np.float32)
            elif axis == 'y':
                self.axis = np.array([0, 1, 0], dtype=np.float32)
            elif axis == 'x':
                self.axis = np.array([1, 0, 0], dtype=np.float32)
            else:
                raise ValueError
        elif isinstance(axis, list):
            if len(axis) != 3:
                raise ValueError('Axis must be length 3(xyz)')
            self.axis = np.array(axis, dtype=np.float32)
        elif isinstance(axis, np.ndarray):
            self.axis = axis
        else:
            raise TypeError

        self._joint_angle = 0.0

        if self.min_angle is None:
            self.min_angle = - np.pi / 2.0
        if self.max_angle is None:
            self.max_angle = np.pi + self.min_angle

        self.joint_velocity = 0.0  # [rad/s]
        self.joint_acceleration = 0.0  # [rad/s^2]
        self.joint_torque = 0.0  # [Nm]

    def joint_angle(self, v=None, relative=None, enable_hook=True):
        """Joint angle method

        Return joint-angle if v is not set, if v is given, set joint
        angle.
        v is rotational value in degree.
        """
        if v is None:
            return self._joint_angle
        if self.joint_min_max_table and self.joint_min_max_target:
            self.min_angle = self.joint_min_max_table_min_angle
            self.max_angle = self.joint_min_max_table_max_angle
        if relative:
            v += self.joint_angle()
        if v > self.max_angle:
            if not relative:
                logger.warning('{} :joint-angle({}) violate max-angle({})'
                               .format(self, v, self.max_angle))
            v = self.max_angle
        elif v < self.min_angle:
            if not relative:
                logger.warning('{} :joint-angle({}) violate min-angle({})'
                               .format(self, v, self.min_angle))
            v = self.min_angle
        self._joint_angle = v
        # (send child-link :replace-coords default-coords)
        # (send child-link :rotate (deg2rad joint-angle) axis))
        self.child_link.rotation = self.default_coords.rotation.copy()
        self.child_link.translation = self.default_coords.translation.copy()
        self.child_link.rotate(self._joint_angle, self.axis)
        if enable_hook:
            for hook in self._hooks:
                hook()
        return self._joint_angle

    @property
    def joint_dof(self):
        """Returns DOF of rotational joint, 1."""
        return 1

    def calc_angle_speed_gain(self, dav, i, periodic_time):
        return calc_angle_speed_gain_scalar(self, dav, i, periodic_time)

    def calc_jacobian(self, *args, **kwargs):
        return calc_jacobian_rotational(*args, **kwargs)


class FixedJoint(Joint):

    def __init__(self,
                 *args, **kwargs):
        super(FixedJoint, self).__init__(
            max_joint_velocity=0.0,
            max_joint_torque=0.0,
            *args, **kwargs)
        self.axis = [0, 0, 1]
        self._joint_angle = 0.0
        self.min_angle = 0.0
        self.max_angle = 0.0
        self.joint_velocity = 0.0  # [rad/s]
        self.joint_acceleration = 0.0  # [rad/s^2]
        self.joint_torque = 0.0  # [Nm]

    def joint_angle(self, v=None, relative=None, enable_hook=True):
        """Joint angle method.

        Return joint_angle
        """
        return self._joint_angle

    @property
    def joint_dof(self):
        """Returns DOF of rotational joint, 0."""
        return 0

    def calc_angle_speed_gain(self, dav, i, periodic_time):
        return 1.0

    def calc_jacobian(self, *args, **kwargs):
        return calc_jacobian_rotational(*args, **kwargs)


def calc_jacobian_rotational(jacobian, row, column, joint, paxis, child_link,
                             world_default_coords, child_reverse,
                             move_target, transform_coords, rotation_axis,
                             translation_axis):
    j_rot = calc_jacobian_default_rotate_vector(
        paxis, world_default_coords, child_reverse, transform_coords)
    p_diff = np.matmul(transform_coords.worldrot().T,
                       (move_target.worldpos() - child_link.worldpos()))
    j_translation = np.cross(j_rot, p_diff)
    j_translation = calc_dif_with_axis(j_translation, translation_axis)
    jacobian[row:row + len(j_translation), column] = j_translation
    j_rotation = calc_dif_with_axis(j_rot, rotation_axis)
    jacobian[row + len(j_translation):
             row + len(j_translation) + len(j_rotation),
             column] = j_rotation
    return jacobian


def calc_jacobian_linear(jacobian, row, column,
                         joint, paxis, child_link,
                         world_default_coords, child_reverse,
                         move_target, transform_coords,
                         rotation_axis, translation_axis):
    j_trans = calc_jacobian_default_rotate_vector(
        paxis, world_default_coords, child_reverse, transform_coords)
    j_rot = np.array([0, 0, 0])
    j_trans = calc_dif_with_axis(j_trans, translation_axis)
    jacobian[row:row + len(j_trans), column] = j_trans
    j_rot = calc_dif_with_axis(j_rot, rotation_axis)
    jacobian[row + len(j_trans):
             row + len(j_trans) + len(j_rot),
             column] = j_rot
    return jacobian


def calc_jacobian_default_rotate_vector(
        paxis, world_default_coords, child_reverse,
        transform_coords):
    if child_reverse:
        sign = -1.0
    else:
        sign = 1.0
    v = sign * normalize_vector(world_default_coords.rotate_vector(paxis))
    return np.dot(transform_coords.worldrot().T, v)


class LinearJoint(Joint):

    def __init__(self, axis='z',
                 max_joint_velocity=np.pi / 4,  # [m/s]
                 max_joint_torque=100,  # [N]
                 *args, **kwargs):
        self.axis = _wrap_axis(axis)
        self._joint_angle = 0.0
        super(LinearJoint, self).__init__(
            max_joint_velocity=max_joint_velocity,
            max_joint_torque=max_joint_torque,
            *args, **kwargs)
        if self.min_angle is None:
            self.min_angle = - np.pi / 2.0
        if self.max_angle is None:
            self.max_angle = np.pi / 2.0
        self.joint_velocity = 0.0  # [m/s]
        self.joint_acceleration = 0.0  # [m/s^2]
        self.joint_torque = 0.0  # [N]

    def joint_angle(self, v=None, relative=None, enable_hook=True):
        """return joint-angle if v is not set, if v is given, set joint angle.

        v is linear value in [m].
        """
        if v is not None:
            if relative is not None:
                v = v + self._joint_angle
            if v > self.max_angle:
                if not relative:
                    logger.warning('{} :joint-angle({}) violate max-angle({})'
                                   .format(self, v, self.max_angle))
                v = self.max_angle
            elif v < self.min_angle:
                if not relative:
                    logger.warning('{} :joint-angle({}) violate min-angle({})'
                                   .format(self, v, self.min_angle))
                v = self.min_angle
            self._joint_angle = v
            self.child_link.rotation = self.default_coords.rotation.copy()
            self.child_link.translation = \
                self.default_coords.translation.copy()
            self.child_link.translate(self._joint_angle * self.axis)

            if enable_hook:
                for hook in self._hooks:
                    hook()
        return self._joint_angle

    @property
    def joint_dof(self):
        """Returns DOF of rotational joint, 1."""
        return 1

    def calc_angle_speed_gain(self, dav, i, periodic_time):
        return calc_angle_speed_gain_scalar(self, dav, i, periodic_time)

    def calc_jacobian(self, *args, **kwargs):
        return calc_jacobian_linear(*args, **kwargs)


class OmniWheelJoint(Joint):

    def __init__(self,
                 max_joint_velocity=(1.6, 1.6, np.pi / 4),
                 max_joint_torque=(100, 100, 100),
                 min_angle=np.array([-np.inf] * 3),
                 max_angle=np.array([np.inf] * 3),
                 *args, **kwargs):
        self.axis = ((1, 0, 0),
                     (0, 1, 0),
                     (0, 0, 1))
        self._joint_angle = np.zeros(3, dtype=np.float64)
        super(OmniWheelJoint, self).__init__(
            max_joint_velocity=max_joint_velocity,
            max_joint_torque=max_joint_torque,
            min_angle=min_angle,
            max_angle=max_angle,
            *args, **kwargs)
        self.joint_velocity = (0, 0, 0)
        self.joint_acceleration = (0, 0, 0)
        self.joint_torque = (0, 0, 0)

    def joint_angle(self, v=None, relative=None, enable_hook=True):
        """Return joint-angle if v is not set, if v is given, set joint angle.

        """
        if v is not None:
            v = np.array(v, dtype=np.float64)
            # translation
            if relative is not None:
                self._joint_angle = v + self._joint_angle
            else:
                self._joint_angle = v

            # min max check
            self._joint_angle = np.minimum(
                np.maximum(self._joint_angle, self.min_angle), self.max_angle)

            # update child_link
            self.child_link.rotation = self.default_coords.rotation.copy()
            self.child_link.translation = \
                self.default_coords.translation.copy()
            self.child_link.translate(
                (self._joint_angle[0], self._joint_angle[1], 0))
            self.child_link.rotate(self._joint_angle[2], 'z')
            if enable_hook:
                for hook in self._hooks:
                    hook()
        return self._joint_angle

    @property
    def joint_dof(self):
        """Returns DOF of rotational joint, 3."""
        return 3

    def calc_angle_speed_gain(self, dav, i, periodic_time):
        return calc_angle_speed_gain_vector(self, dav, i, periodic_time)

    def calc_jacobian(self,
                      jacobian, row, column,
                      joint, paxis, child_link,
                      world_default_coords, child_reverse,
                      move_target, transform_coords,
                      rotation_axis, translation_axis):
        calc_jacobian_linear(jacobian, row, column + 0,
                             joint, [1, 0, 0], child_link,
                             world_default_coords, child_reverse,
                             move_target, transform_coords,
                             rotation_axis, translation_axis)
        calc_jacobian_linear(jacobian, row, column + 1,
                             joint, [0, 1, 0], child_link,
                             world_default_coords, child_reverse,
                             move_target, transform_coords,
                             rotation_axis, translation_axis)
        calc_jacobian_rotational(jacobian, row, column + 2,
                                 joint, [0, 0, 1], child_link,
                                 world_default_coords, child_reverse,
                                 move_target, transform_coords,
                                 rotation_axis, translation_axis)
        return jacobian

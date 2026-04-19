from logging import getLogger
import math

import numpy as np

from skrobot.coordinates import convert_to_axis_vector
from skrobot.coordinates import normalize_vector
from skrobot.coordinates.math import cross_product
from skrobot.coordinates.math import select_by_mask


logger = getLogger(__name__)

_default_max_joint_velocity = 1.0
_default_max_joint_torque = 1.0


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
        if j is not None:
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


class _MimicJointHook(object):

    def __init__(self, this, other, multiplier, offset):
        self.this_joint = this
        self.other_joint = other
        self.multiplier = multiplier
        self.offset = offset

    def __call__(self):
        self.other_joint.joint_angle(
            self.this_joint.joint_angle() * self.multiplier + self.offset)


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
        self.max_joint_torque = max_joint_torque
        self.joint_min_max_table = joint_min_max_table
        self.joint_min_max_target = joint_min_max_target
        self.default_coords = self.child_link.copy_coords()
        self._hooks = hooks

    @property
    def type(self):
        raise NotImplementedError('Joint type must be defined')

    @property
    def joint_type(self):
        """Alias for type property for better clarity.

        Using joint_type makes it more explicit that this is a joint property,
        reducing confusion with other 'type' attributes in the codebase.

        Returns
        -------
        str
            Joint type ('revolute', 'continuous', 'prismatic', 'fixed', etc.)
        """
        return self.type

    @property
    def min_joint_angle(self):
        """Alias for min_angle property for better clarity.

        Returns
        -------
        float
            Minimum joint angle
        """
        return self.min_angle

    @min_joint_angle.setter
    def min_joint_angle(self, value):
        """Setter for min_joint_angle (updates min_angle)."""
        self.min_angle = value

    @property
    def max_joint_angle(self):
        """Alias for max_angle property for better clarity.

        Returns
        -------
        float
            Maximum joint angle
        """
        return self.max_angle

    @max_joint_angle.setter
    def max_joint_angle(self, value):
        """Setter for max_joint_angle (updates max_angle)."""
        self.max_angle = value

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
        # NOTE: use _MimicJointHook callable to avoid using lambda function.
        # Otherwise, lambda function will not be pickled and will not
        # be deepcopied correctly.
        mimic_joint_hook = _MimicJointHook(self, joint, multiplier, offset)
        self.register_hook(mimic_joint_hook)

    @property
    def world_axis(self):
        """Return joint axis in world coordinate system.

        For rotational and linear joints, this converts the local axis
        to world coordinates using the parent link's world transformation
        and the joint's default coordinates (initial pose).

        This follows the same calculation as the non-batch Jacobian:
        world_default_coords = parent_link.worldcoords().transform(default_coords)
        world_axis = world_default_coords.rotate_vector(axis)

        Returns
        -------
        axis : numpy.ndarray
            Joint axis in world coordinate system.
        """
        if not hasattr(self, 'axis'):
            return np.array([0, 0, 1])

        if self.parent_link is not None:
            return self.parent_link.copy_worldcoords().transform(
                self.default_coords).rotate_vector(self.axis)

        return self.default_coords.rotate_vector(self.axis)

    @property
    def world_position(self):
        """Return joint position in world coordinate system.

        The joint position is defined as the world position of the parent link.

        Returns
        -------
        position : numpy.ndarray
            Joint position in world coordinate system.
        """
        if self.parent_link is not None:
            return self.parent_link.copy_worldcoords().transform(
                self.default_coords).worldpos()
        return self.default_coords.worldpos()

    @property
    def joint_min_max_table_min_angle(self):
        """Get minimum angle from joint limit table.

        When the joint has a joint_min_max_table and joint_min_max_target,
        this returns the minimum angle based on the target joint's current angle.

        This property returns a callable float-like object that can be:
        - Used as a float value (uses target joint's current angle)
        - Called with a specific target angle: joint_min_max_table_min_angle(angle)

        Returns
        -------
        _JointLimitValue
            Callable float-like object representing the min angle.
        """
        if self.joint_min_max_table is None or self.joint_min_max_target is None:
            return self.min_angle
        return _JointLimitValue(
            self.joint_min_max_table.min_angle_function,
            self.joint_min_max_target.joint_angle()
        )

    @property
    def joint_min_max_table_max_angle(self):
        """Get maximum angle from joint limit table.

        When the joint has a joint_min_max_table and joint_min_max_target,
        this returns the maximum angle based on the target joint's current angle.

        This property returns a callable float-like object that can be:
        - Used as a float value (uses target joint's current angle)
        - Called with a specific target angle: joint_min_max_table_max_angle(angle)

        Returns
        -------
        _JointLimitValue
            Callable float-like object representing the max angle.
        """
        if self.joint_min_max_table is None or self.joint_min_max_target is None:
            return self.max_angle
        return _JointLimitValue(
            self.joint_min_max_table.max_angle_function,
            self.joint_min_max_target.joint_angle()
        )


class _JointLimitValue(float):
    """A float-like object that can also be called with a target angle.

    This class is used to provide a value that acts as a float when used
    directly, but can also be called with a specific target joint angle
    to get the limit for that angle.
    """

    def __new__(cls, func, current_target_angle):
        value = func(current_target_angle)
        instance = super(_JointLimitValue, cls).__new__(cls, value)
        instance._func = func
        return instance

    def __call__(self, target_angle):
        """Get the limit value for a specific target angle.

        Parameters
        ----------
        target_angle : float
            The target joint angle to compute the limit for.

        Returns
        -------
        float
            The limit value at the specified target angle.
        """
        return self._func(target_angle)


class RotationalJoint(Joint):

    def __init__(self, axis='z',
                 max_joint_velocity=np.deg2rad(5),
                 max_joint_torque=100,
                 *args, **kwargs):
        super(RotationalJoint, self).__init__(
            max_joint_velocity=max_joint_velocity,
            max_joint_torque=max_joint_torque,
            *args, **kwargs)
        self.axis = normalize_vector(convert_to_axis_vector(axis))
        self._joint_angle = 0.0

        if self.min_angle is None:
            self.min_angle = - np.pi / 2.0
        if self.max_angle is None:
            self.max_angle = np.pi + self.min_angle

        self.joint_velocity = 0.0  # [rad/s]
        self.joint_acceleration = 0.0  # [rad/s^2]
        self.joint_torque = 0.0  # [Nm]

        if max_joint_velocity <= 0:
            message = '[WARN] Joint "{}" '.format(self.name)
            message += "max_joint_velocity cannot be zero. "
            message += 'Setting to default value np.deg2rad(5).'
            logger.warning(message)
            self.max_joint_velocity = np.deg2rad(5)

    @property
    def type(self):
        if np.isinf(self.min_angle) or np.isinf(self.max_angle):
            return 'continuous'
        return 'revolute'

    @property
    def joint_axis(self):
        """Alias for axis property for better clarity.

        Returns
        -------
        numpy.ndarray
            Joint rotation axis (normalized 3D vector)
        """
        return self.axis

    @joint_axis.setter
    def joint_axis(self, value):
        """Setter for joint_axis (updates axis)."""
        self.axis = normalize_vector(convert_to_axis_vector(value))

    def joint_angle(self, v=None, relative=None, enable_hook=True):
        """Return joint angle.

        Return joint angle if v is not set, if v is given, set the value as
        a joint angle.

        Parameters
        ----------
        v : None or float
            Joint angle in a radian.
            If v is `None`, return this joint's joint angle.
        relative : None or bool
            If relative is `True`, an input `v` represents the relative
            translation.

        Parameters
        ----------
        self._joint_angle : float
            Current joint_angle in a radian.
        """
        if v is None:
            return self._joint_angle
        # Cache the table into a local before the ``is not None``
        # check so a user assigning ``joint.joint_min_max_table = ...``
        # post-construction still takes effect (no cached flag).  The
        # hot-path savings come from the single attribute read + the
        # other optimisations below, not from skipping this check.
        jmt = self.joint_min_max_table
        if jmt is not None and self.joint_min_max_target is not None:
            self.min_angle = self.joint_min_max_table_min_angle
            self.max_angle = self.joint_min_max_table_max_angle
        if relative:
            v += self._joint_angle
        # Cache the limits once so the three comparisons below don't
        # re-dereference ``self.max_angle`` / ``self.min_angle`` each.
        max_angle = self.max_angle
        min_angle = self.min_angle
        # Handle infeasible region where max < min (can occur at
        # extreme configurations in bidirectional joint limit tables).
        if max_angle < min_angle:
            v = 0.5 * (float(min_angle) + float(max_angle))
        elif v > max_angle:
            if not relative:
                logger.warning('%s :joint-angle(%s) violate max-angle(%s)', self, v, max_angle)
            v = max_angle
        elif v < min_angle:
            if not relative:
                logger.warning('%s :joint-angle(%s) violate min-angle(%s)', self, v, min_angle)
            v = min_angle
        diff_angle = v - self._joint_angle
        self._joint_angle = v
        if diff_angle:
            self.child_link.rotate(diff_angle, self.axis,
                                   skip_normalization=True)
        # Skip the for-loop setup when no hooks are registered.
        if enable_hook and self._hooks:
            for hook in self._hooks:
                hook()
        return self._joint_angle

    # ``joint_dof`` is a per-class constant; a class attribute is a
    # touch cheaper than the ``@property`` it used to be.  Subclasses
    # override by re-declaring it.
    joint_dof = 1

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

    @property
    def type(self):
        return 'fixed'

    def joint_angle(self, v=None, relative=None, enable_hook=True):
        """Joint angle method.

        Return joint_angle
        """
        return self._joint_angle

    joint_dof = 0

    def calc_angle_speed_gain(self, dav, i, periodic_time):
        return 1.0

    def calc_jacobian(self, *args, **kwargs):
        return calc_jacobian_rotational(*args, **kwargs)


def calc_jacobian_rotational(jacobian, row, column, joint, paxis, child_link,
                             world_default_coords,
                             move_target, transform_coords, rotation_mask,
                             position_mask):
    # ``worldrot()`` triggers a worldcoords update and returns the
    # rotation matrix; we need its transpose twice (once inside
    # calc_jacobian_default_rotate_vector, once for ``p_diff``).  Cache
    # the transpose locally and use ``np.dot`` instead of ``np.matmul``
    # (identical result for 2-D inputs, shorter numpy dispatch path for
    # small matrices -- same trick as in ``transform_coords``).
    wrot_T = transform_coords.worldrot().T
    j_rot = np.dot(wrot_T, world_default_coords.rotate_vector(paxis))
    p_diff = np.dot(wrot_T,
                    move_target.worldpos() - child_link.worldpos())
    j_translation = cross_product(j_rot, p_diff)
    j_translation = select_by_mask(j_translation, position_mask)
    n_trans = len(j_translation)
    jacobian[row:row + n_trans, column] = j_translation
    j_rotation = select_by_mask(j_rot, rotation_mask)
    jacobian[row + n_trans:row + n_trans + len(j_rotation),
             column] = j_rotation
    return jacobian


def calc_jacobian_linear(jacobian, row, column,
                         joint, paxis, child_link,
                         world_default_coords,
                         move_target, transform_coords,
                         rotation_mask, position_mask):
    j_trans = calc_jacobian_default_rotate_vector(
        paxis, world_default_coords, transform_coords)
    j_rot = np.array([0, 0, 0])
    j_trans = select_by_mask(j_trans, position_mask)
    jacobian[row:row + len(j_trans), column] = j_trans
    j_rot = select_by_mask(j_rot, rotation_mask)
    jacobian[row + len(j_trans):
             row + len(j_trans) + len(j_rot),
             column] = j_rot
    return jacobian


def calc_jacobian_default_rotate_vector(
        paxis, world_default_coords,
        transform_coords):
    v = world_default_coords.rotate_vector(paxis)
    return np.dot(transform_coords.worldrot().T, v)


class LinearJoint(Joint):

    def __init__(self, axis='z',
                 max_joint_velocity=np.pi / 4,  # [m/s]
                 max_joint_torque=100,  # [N]
                 *args, **kwargs):
        self.axis = normalize_vector(convert_to_axis_vector(axis))
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

        if max_joint_velocity <= 0:
            message = '[WARN] Joint "{}" '.format(self.name)
            message += "max_joint_velocity cannot be zero. "
            message += 'Setting to default value np.pi / 4.0.'
            logger.warning(message)
            self.max_joint_velocity = np.pi / 4.0

    @property
    def type(self):
        return 'prismatic'

    @property
    def joint_axis(self):
        """Alias for axis property for better clarity.

        Returns
        -------
        numpy.ndarray
            Joint translation axis (normalized 3D vector)
        """
        return self.axis

    @joint_axis.setter
    def joint_axis(self, value):
        """Setter for joint_axis (updates axis)."""
        self.axis = normalize_vector(convert_to_axis_vector(value))

    def joint_angle(self, v=None, relative=None, enable_hook=True):
        """Return this joint's linear translation (joint angle).

        Parameters
        ----------
        v : None or float
            Linear translation (joint angle) in a meter.
            If v is `None`, return current this joint's translation.
        relative : None or bool
            If relative is `True`, an input `v` represents the relative
            translation.

        Parameters
        ----------
        self._joint_angle : float
            current linear translation (joint_angle).
        """
        if v is not None:
            if relative is not None:
                v = v + self._joint_angle
            max_angle = self.max_angle
            min_angle = self.min_angle
            if v > max_angle:
                if not relative:
                    logger.warning('%s :joint-angle(%s) violate max-angle(%s)', self, v, max_angle)
                v = max_angle
            elif v < min_angle:
                if not relative:
                    logger.warning('%s :joint-angle(%s) violate min-angle(%s)', self, v, min_angle)
                v = min_angle
            diff_translation = v - self._joint_angle
            self._joint_angle = v
            if diff_translation:
                self.child_link.translate(diff_translation * self.axis)

            if enable_hook and self._hooks:
                for hook in self._hooks:
                    hook()
        return self._joint_angle

    joint_dof = 1

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

    joint_dof = 3

    def calc_angle_speed_gain(self, dav, i, periodic_time):
        return calc_angle_speed_gain_vector(self, dav, i, periodic_time)

    def calc_jacobian(self,
                      jacobian, row, column,
                      joint, paxis, child_link,
                      world_default_coords,
                      move_target, transform_coords,
                      rotation_axis, translation_axis):
        calc_jacobian_linear(jacobian, row, column + 0,
                             joint, [1, 0, 0], child_link,
                             world_default_coords,
                             move_target, transform_coords,
                             rotation_axis, translation_axis)
        calc_jacobian_linear(jacobian, row, column + 1,
                             joint, [0, 1, 0], child_link,
                             world_default_coords,
                             move_target, transform_coords,
                             rotation_axis, translation_axis)
        calc_jacobian_rotational(jacobian, row, column + 2,
                                 joint, [0, 0, 1], child_link,
                                 world_default_coords,
                                 move_target, transform_coords,
                                 rotation_axis, translation_axis)
        return jacobian


class PlanarJoint(Joint):
    """3-DoF virtual joint for a floating base constrained to a plane.

    Joint angle vector layout: [x, y, yaw]. Intended as a virtual joint
    inserted between the world and a robot's root_link so that fullbody
    inverse kinematics can solve for base pose on flat ground.
    """

    def __init__(self,
                 max_joint_velocity=(np.inf, np.inf, np.inf),
                 max_joint_torque=(np.inf, np.inf, np.inf),
                 min_angle=None,
                 max_angle=None,
                 *args, **kwargs):
        if min_angle is None:
            min_angle = np.array([-np.inf] * 3)
        if max_angle is None:
            max_angle = np.array([np.inf] * 3)
        self.axis = ((1, 0, 0),
                     (0, 1, 0),
                     (0, 0, 1))
        self._joint_angle = np.zeros(3, dtype=np.float64)
        super(PlanarJoint, self).__init__(
            max_joint_velocity=max_joint_velocity,
            max_joint_torque=max_joint_torque,
            min_angle=min_angle,
            max_angle=max_angle,
            *args, **kwargs)
        self.joint_velocity = (0, 0, 0)
        self.joint_acceleration = (0, 0, 0)
        self.joint_torque = (0, 0, 0)

    @property
    def type(self):
        return 'planar'

    def joint_angle(self, v=None, relative=None, enable_hook=True):
        if v is not None:
            v = np.array(v, dtype=np.float64)
            if relative is not None:
                self._joint_angle = v + self._joint_angle
            else:
                self._joint_angle = v
            self._joint_angle = np.minimum(
                np.maximum(self._joint_angle, self.min_angle), self.max_angle)
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

    joint_dof = 3

    def calc_angle_speed_gain(self, dav, i, periodic_time):
        return calc_angle_speed_gain_vector(self, dav, i, periodic_time)

    def calc_jacobian(self,
                      jacobian, row, column,
                      joint, paxis, child_link,
                      world_default_coords,
                      move_target, transform_coords,
                      rotation_axis, translation_axis):
        calc_jacobian_linear(jacobian, row, column + 0,
                             joint, [1, 0, 0], child_link,
                             world_default_coords,
                             move_target, transform_coords,
                             rotation_axis, translation_axis)
        calc_jacobian_linear(jacobian, row, column + 1,
                             joint, [0, 1, 0], child_link,
                             world_default_coords,
                             move_target, transform_coords,
                             rotation_axis, translation_axis)
        calc_jacobian_rotational(jacobian, row, column + 2,
                                 joint, [0, 0, 1], child_link,
                                 world_default_coords,
                                 move_target, transform_coords,
                                 rotation_axis, translation_axis)
        return jacobian


class FloatingJoint(Joint):
    """6-DoF virtual joint for an unconstrained floating base.

    Joint angle vector layout: [x, y, z, rx, ry, rz]. The rotational
    part is applied as successive rotations about the child link's
    local x, y, z axes (body-fixed XYZ Euler). Intended as a virtual
    joint inserted between the world and a robot's root_link so that
    fullbody inverse kinematics can solve for base pose on uneven
    terrain (e.g. slopes).
    """

    def __init__(self,
                 max_joint_velocity=(np.inf,) * 6,
                 max_joint_torque=(np.inf,) * 6,
                 min_angle=None,
                 max_angle=None,
                 *args, **kwargs):
        if min_angle is None:
            min_angle = np.array([-np.inf] * 6)
        if max_angle is None:
            max_angle = np.array([np.inf] * 6)
        self.axis = ((1, 0, 0),
                     (0, 1, 0),
                     (0, 0, 1),
                     (1, 0, 0),
                     (0, 1, 0),
                     (0, 0, 1))
        self._joint_angle = np.zeros(6, dtype=np.float64)
        super(FloatingJoint, self).__init__(
            max_joint_velocity=max_joint_velocity,
            max_joint_torque=max_joint_torque,
            min_angle=min_angle,
            max_angle=max_angle,
            *args, **kwargs)
        self.joint_velocity = (0,) * 6
        self.joint_acceleration = (0,) * 6
        self.joint_torque = (0,) * 6

    @property
    def type(self):
        return 'floating'

    def joint_angle(self, v=None, relative=None, enable_hook=True):
        if v is not None:
            v = np.array(v, dtype=np.float64)
            if relative is not None:
                self._joint_angle = v + self._joint_angle
            else:
                self._joint_angle = v
            self._joint_angle = np.minimum(
                np.maximum(self._joint_angle, self.min_angle), self.max_angle)
            self.child_link.rotation = self.default_coords.rotation.copy()
            self.child_link.translation = \
                self.default_coords.translation.copy()
            self.child_link.translate(
                (self._joint_angle[0],
                 self._joint_angle[1],
                 self._joint_angle[2]))
            self.child_link.rotate(self._joint_angle[3], 'x')
            self.child_link.rotate(self._joint_angle[4], 'y')
            self.child_link.rotate(self._joint_angle[5], 'z')
            if enable_hook:
                for hook in self._hooks:
                    hook()
        return self._joint_angle

    joint_dof = 6

    def calc_angle_speed_gain(self, dav, i, periodic_time):
        return calc_angle_speed_gain_vector(self, dav, i, periodic_time)

    def calc_jacobian(self,
                      jacobian, row, column,
                      joint, paxis, child_link,
                      world_default_coords,
                      move_target, transform_coords,
                      rotation_axis, translation_axis):
        calc_jacobian_linear(jacobian, row, column + 0,
                             joint, [1, 0, 0], child_link,
                             world_default_coords,
                             move_target, transform_coords,
                             rotation_axis, translation_axis)
        calc_jacobian_linear(jacobian, row, column + 1,
                             joint, [0, 1, 0], child_link,
                             world_default_coords,
                             move_target, transform_coords,
                             rotation_axis, translation_axis)
        calc_jacobian_linear(jacobian, row, column + 2,
                             joint, [0, 0, 1], child_link,
                             world_default_coords,
                             move_target, transform_coords,
                             rotation_axis, translation_axis)
        # Rotational DoFs are applied in body-fixed XYZ order (rx->ry->rz).
        # The effective rotation axis for each DoF must reflect the
        # intermediate rotations, otherwise ry/rz columns diverge from the
        # numerical derivative once rx/ry are non-zero.
        rx = self._joint_angle[3]
        ry = self._joint_angle[4]
        c_rx, s_rx = np.cos(rx), np.sin(rx)
        c_ry, s_ry = np.cos(ry), np.sin(ry)
        ry_axis_local = np.array([0.0, c_rx, s_rx])
        rz_axis_local = np.array([s_ry, -s_rx * c_ry, c_rx * c_ry])
        calc_jacobian_rotational(jacobian, row, column + 3,
                                 joint, [1, 0, 0], child_link,
                                 world_default_coords,
                                 move_target, transform_coords,
                                 rotation_axis, translation_axis)
        calc_jacobian_rotational(jacobian, row, column + 4,
                                 joint, ry_axis_local, child_link,
                                 world_default_coords,
                                 move_target, transform_coords,
                                 rotation_axis, translation_axis)
        calc_jacobian_rotational(jacobian, row, column + 5,
                                 joint, rz_axis_local, child_link,
                                 world_default_coords,
                                 move_target, transform_coords,
                                 rotation_axis, translation_axis)
        return jacobian


def calc_joint_angle_min_max_for_limit_calculation(j, kk, jamm=None):
    if jamm is None:
        jamm = np.zeros(3, 'f')
    if j.joint_dof > 1:
        jamm[0] = j.joint_angle()[kk]
        jamm[1] = j.max_angle[kk]
        jamm[2] = j.min_angle[kk]
    else:
        jamm[0] = j.joint_angle()
        # Use dynamic limits from joint limit table if available
        if j.joint_min_max_table is not None:
            jamm[1] = float(j.joint_min_max_table_max_angle)
            jamm[2] = float(j.joint_min_max_table_min_angle)
        else:
            jamm[1] = j.max_angle
            jamm[2] = j.min_angle
    return jamm


def _joint_limit_scalars(j, kk):
    """Return (angle, max, min) as Python floats.

    Internal helper for ``joint_angle_limit_weight`` /
    ``joint_angle_limit_nspace``; avoids the ``np.zeros(3)`` scratch
    the public ``calc_joint_angle_min_max_for_limit_calculation``
    allocates.
    """
    if j.joint_dof > 1:
        return (float(j.joint_angle()[kk]),
                float(j.max_angle[kk]),
                float(j.min_angle[kk]))
    if j.joint_min_max_table is not None:
        return (float(j.joint_angle()),
                float(j.joint_min_max_table_max_angle),
                float(j.joint_min_max_table_min_angle))
    return (float(j.joint_angle()),
            float(j.max_angle),
            float(j.min_angle))


# ``np.deg2rad(1)``; hoisted as a module-level constant so the joint-limit
# loops don't re-evaluate it per joint per IK iteration.  Keep it as a
# numpy float64 scalar -- mixing it with float32 ``jamm`` entries then
# promotes the arithmetic to float64, matching the original precision
# path that ``test_joint_angle_limit_weight`` asserts against.
_ONE_DEGREE_IN_RADIANS = np.float64(0.017453292519943295)


def joint_angle_limit_weight(joint_list):
    """Calculate joint angle limit from joint list.

    w_i = 1 + | dH/dt |      if d|dH/dt| >= 0
        = 1                  if d|dH/dt| <  0
    dH/dt = (t_max - t_min)^2 (2t - t_max - t_min) /
            (4 (t_max - t)^2 (t - t_min)^2)

    T. F. Chang and R.-V. Dubey: "A weighted least-norm solution based
    scheme for avoiding joint limits for redundant manipulators",
    in IEEE Trans. On Robotics and Automation,
    11((2):286-292, April 1995.

    Parameters
    ----------
    joint_list : list[skrobot.model.Joint]
        joint list

    Returns
    -------
    res : numpy.ndarray
        joint angle limit
    """
    # Inner loop runs ~dims times per IK iteration.  Hoist the
    # "1 degree" constant, cache its square/fourth powers, and replace
    # scalar ``np.abs`` / ``np.isinf`` / ``np.deg2rad`` with the cheaper
    # Python built-in and ``math`` equivalents.  The ``jamm`` scratch
    # buffer (float32) is kept so intermediate precision matches the
    # test expectations in ``test_joint_angle_limit_weight``.
    dims = calc_target_joint_dimension(joint_list)
    res = np.zeros(dims, 'f')
    k = 0
    kk = 0
    jamm = np.zeros(3, 'f')
    e = _ONE_DEGREE_IN_RADIANS
    e_sq = e * e
    e_quad = e_sq * e_sq
    inf = float('inf')
    for i in range(dims):
        j = joint_list[k]
        calc_joint_angle_min_max_for_limit_calculation(j, kk, jamm)
        joint_angle, joint_max, joint_min = jamm
        if j.joint_dof > 1:
            kk += 1
            if kk >= j.joint_dof:
                kk = 0
                k += 1
        else:
            k += 1

        # limitation
        d_max = abs(joint_angle - joint_max)
        d_min = abs(joint_angle - joint_min)
        if d_max < e and d_min < e:
            pass
        elif d_max < e:
            joint_angle = joint_max - e
        elif d_min < e:
            joint_angle = joint_min + e
        # calculate weight
        if abs(joint_angle - joint_max) < e and \
           abs(joint_angle - joint_min) < e:
            res[i] = inf
        else:
            if math.isinf(joint_min) or math.isinf(joint_max):
                r = 0.0
            else:
                # Check for degenerate range (can happen with dynamic limits)
                range_sq = (joint_max - joint_min) ** 2
                denom = 4.0 * ((joint_max - joint_angle) ** 2) \
                    * ((joint_angle - joint_min) ** 2)
                if range_sq < e_sq or denom < e_quad:
                    # Degenerate case: range too small or at boundary
                    r = inf
                else:
                    r = abs(range_sq
                            * (2.0 * joint_angle - joint_max - joint_min)
                            / denom)
                    # Handle NaN from numerical issues
                    if math.isnan(r) or math.isinf(r):
                        r = inf
            res[i] = r
    return res


def joint_angle_limit_nspace(
        joint_list,
        n_joint_dimension=None):
    """Calculate nspace weight for avoiding joint angle limit.

    .. math::
        \\frac{dH}{dq} = (\\frac{\\frac{t_{max} + t_{min}}{2} - t}
                          {\\frac{t_{max} - t_{min}}{2}})^2


    Parameters
    ----------
    joint_list : list[skrobot.model.Joint]
        joint list
    n_joint_dimension : int or None
        if this value is None, set n_joint_dimension by
        skrobot.model.calc_target_joint_dimension

    Returns
    -------
    nspace : numpy.ndarray
        null space shape of (n_joint_dimensoin, )
    """
    if n_joint_dimension is None:
        n_joint_dimension = calc_target_joint_dimension(joint_list)
    nspace = np.zeros(n_joint_dimension, 'f')
    k = 0
    kk = 0
    # Keep the per-joint math on Python floats; the old call to
    # ``calc_joint_angle_min_max_for_limit_calculation`` allocated a
    # 3-element ndarray per joint and handed back numpy scalars.
    for i in range(n_joint_dimension):
        joint = joint_list[k]
        joint_angle, joint_max, joint_min = _joint_limit_scalars(joint, kk)

        if joint.joint_dof > 1:
            kk += 1
            if kk >= joint.joint_dof:
                kk = 0
                k += 1
        else:
            k += 1
        # calculate weight
        span = joint_max - joint_min
        if span == 0.0 or math.isinf(joint_max) or math.isinf(joint_min):
            r = 0.0
        else:
            r = ((joint_max + joint_min) - 2.0 * joint_angle) / span
            # ``np.sign(r) * r**2`` reduces to the signed square for r != 0.
            r = r * abs(r)
        nspace[i] = r
    return nspace

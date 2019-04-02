from logging import getLogger
import itertools

import numpy as np
import numpy.linalg as LA
from ordered_set import OrderedSet

from skrobot.coordinates import CascadedCoords
from skrobot.coordinates import Coordinates
from skrobot.coordinates import make_coords
from skrobot.math import _wrap_axis
from skrobot.math import manipulability
from skrobot.math import midpoint
from skrobot.math import normalize_vector
from skrobot.math import sr_inverse
from skrobot.math import make_matrix
from skrobot.optimizer import solve_qp
from skrobot.utils.urdf import URDF
from skrobot.geo import midcoords
from skrobot.geo import orient_coords_to_axis
from skrobot.utils.listify import listify


logger = getLogger(__name__)

_default_max_joint_velocity = 1.0
_default_max_joint_torque = 1.0


def calc_angle_speed_gain_scalar(joint, dav, i, periodic_time):
    if dav[i] == 0 or periodic_time == 0:
        dav_gain = 1.0
    else:
        dav_gain = abs(joint.max_joint_velocity / (dav[i] / periodic_time))
    return min(dav_gain, 1.0)


def calc_target_joint_dimension(joint_list):
    n = 0
    for j in joint_list:
        n += j.joint_dof
    return n


def calc_dif_with_axis(dif, axis):
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
        raise ValueError('axis {} is not supported'.fomrat(axis))
    return ret


class Joint(object):

    def __init__(self, name=None, child_link=None,
                 parent_link=None,
                 min_angle=-90,
                 max_angle=90,
                 max_joint_velocity=None,
                 max_joint_torque=None,
                 joint_min_max_table=None,
                 joint_min_max_target=None):
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

    @property
    def joint_dof(self):
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


class RotationalJoint(Joint):

    def __init__(self, axis='z',
                 max_joint_velocity=5,
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
                raise ValueError("Axis must be length 3(xyz)")
            self.axis = np.array(axis, dtype=np.float32)
        elif isinstance(axis, np.ndarray):
            self.axis = axis
        else:
            raise TypeError

        self._joint_angle = 0.0

        if self.min_angle is None:
            self.min_angle = -90.0
        if self.max_angle is None:
            self.max_angle = 180.0 + self.min_angle

        self.joint_velocity = 0.0  # [rad/s]
        self.joint_acceleration = 0.0  # [rad/s^2]
        self.joint_torque = 0.0  # [Nm]

    def joint_angle(self, v=None, relative=None):
        "Return joint-angle if v is not set, \
        if v is given, set joint angle. v is rotational value in degree."
        if v is None:
            return self._joint_angle
        if self.joint_min_max_table and self.joint_min_max_target:
            self.min_angle = self.joint_min_max_table_min_angle
            self.max_angle = self.joint_min_max_table_max_angle
        if relative:
            v += self.joint_angle()
        if v > self.max_angle:
            if not relative:
                logger.warning("{} :joint-angle({}) violate max-angle({})"
                               .format(self, v, self.max_angle))
            v = self.max_angle
        elif v < self.min_angle:
            if not relative:
                logger.warning("{} :joint-angle({}) violate min-angle({})"
                               .format(self, v, self.min_angle))
            v = self.min_angle
        self._joint_angle = v
        # (send child-link :replace-coords default-coords)
        # (send child-link :rotate (deg2rad joint-angle) axis))
        self.child_link.rotation = self.default_coords.rotation.copy()
        self.child_link.pos = self.default_coords.pos.copy()
        self.child_link.rotate(np.deg2rad(self._joint_angle), self.axis)
        return self._joint_angle

    @property
    def joint_dof(self):
        "Returns DOF of rotational joint, 1."
        return 1

    def calc_angle_speed_gain(self, dav, i, periodic_time):
        return calc_angle_speed_gain_scalar(self, dav, i, periodic_time)

    def speed_to_angle(self, v):
        return np.rad2deg(v)

    def angle_to_speed(self, v):
        return np.deg2rad(v)

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

    def joint_angle(self, v=None, relative=None):
        "Return joint-angle if v is not set, \
        if v is given, set joint angle. v is rotational value in degree."
        if v is None:
            return self._joint_angle
        if self.joint_min_max_table and self.joint_min_max_target:
            self.min_angle = self.joint_min_max_table_min_angle
            self.max_angle = self.joint_min_max_table_max_angle
        if relative:
            v += self.joint_angle()
        if v > self.max_angle:
            if not relative:
                logger.warning("{} :joint-angle({}) violate max-angle({})"
                               .format(self, v, self.max_angle))
            v = self.max_angle
        elif v < self.min_angle:
            if not relative:
                logger.warning("{} :joint-angle({}) violate min-angle({})"
                               .format(self, v, self.min_angle))
            v = self.min_angle
        self._joint_angle = v
        self.child_link.rotation = self.default_coords.rotation.copy()
        self.child_link.pos = self.default_coords.pos.copy()
        self.child_link.rotate(np.deg2rad(self._joint_angle), self.axis)
        return self._joint_angle

    @property
    def joint_dof(self):
        "Returns DOF of rotational joint, 1."
        return 0

    def calc_angle_speed_gain(self, dav, i, periodic_time):
        return calc_angle_speed_gain_scalar(self, dav, i, periodic_time)

    def speed_to_angle(self, v):
        return np.rad2deg(v)

    def angle_to_speed(self, v):
        return np.deg2rad(v)

    def calc_jacobian(self, *args, **kwargs):
        return calc_jacobian_rotational(*args, **kwargs)


def calc_jacobian_rotational(fik, row, column, joint, paxis, child_link,
                             world_default_coords, child_reverse,
                             move_target, transform_coords, rotation_axis,
                             translation_axis):
    j_rot = calc_jacobian_default_rotate_vector(
        paxis, world_default_coords, child_reverse, transform_coords)
    p_diff = np.matmul(transform_coords.worldrot().T,
                       (move_target.worldpos() - child_link.worldpos()))
    j_translation = np.cross(j_rot, p_diff)
    j_translation = calc_dif_with_axis(j_translation, translation_axis)
    fik[row:row + len(j_translation), column] = j_translation
    j_rotation = calc_dif_with_axis(j_rot, rotation_axis)
    fik[row + len(j_translation):
        row + len(j_translation) + len(j_rotation),
        column] = j_rotation
    return fik


def calc_jacobian_linear(fik, row, column,
                         joint, paxis, child_link,
                         world_default_coords, child_reverse,
                         move_target, transform_coords,
                         rotation_axis, translation_axis):
    j_trans = calc_jacobian_default_rotate_vector(
        paxis, world_default_coords, child_reverse, transform_coords)
    j_rot = np.array([0, 0, 0])
    j_trans = calc_dif_with_axis(j_trans, translation_axis)
    for j in range(len(j_trans)):
        fik[j + row, column] = j_trans[j]

    j_rot = calc_dif_with_axis(j_rot, rotation_axis)
    for j in range(len(j_rot)):
        fik[j + row + len(j_trans), column] = j_rot[j]
    return fik


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
            self.min_angle = -90.0
        if self.max_angle is None:
            self.max_angle = 90.0
        self.joint_velocity = 0.0  # [m/s]
        self.joint_acceleration = 0.0  # [m/s^2]
        self.joint_torque = 0.0  # [N]

    def joint_angle(self, v=None, relative=None):
        """return joint-angle if v is not set, if v is given,
        set joint angle. v is linear value in [mm]."""
        if v is not None:
            if relative is not None:
                v = v + self._joint_angle
            if v > self.max_angle:
                if not relative:
                    logger.warning("{} :joint-angle({}) violate max-angle({})"
                                   .format(self, v, self.max_angle))
                v = self.max_angle
            elif v < self.min_angle:
                if not relative:
                    logger.warning("{} :joint-angle({}) violate min-angle({})"
                                   .format(self, v, self.min_angle))
                v = self.min_angle
            self._joint_angle = v
            self.child_link.rotation = self.default_coords.rotation.copy()
            self.child_link.pos = self.default_coords.pos.copy()
            self.child_link.translate(self._joint_angle * self.axis)
        return self._joint_angle

    @property
    def joint_dof(self):
        "Returns DOF of rotational joint, 1."
        return 1

    def calc_angle_speed_gain(self, dav, i, periodic_time):
        return calc_angle_speed_gain_scalar(self, dav, i, periodic_time)

    def speed_to_angle(self, v):
        return 1000 * v

    def angle_to_speed(self, v):
        return 0.001 * v

    def calc_jacobian(self, *args, **kwargs):
        return calc_jacobian_linear(*args, **kwargs)


class Link(CascadedCoords):

    def __init__(self, centroid=None,
                 inertia_tensor=None,
                 *args, **kwargs):
        super(Link, self).__init__(*args, **kwargs)
        self.centroid = centroid
        self.joint = None
        self._child_links = []
        self._parent_link = None
        if inertia_tensor is None:
            inertia_tensor = np.eye(3)

    @property
    def parent_link(self):
        return self._parent_link

    def add_joint(self, j):
        self.joint = j

    def delete_joint(self, j):
        self.joint = None

    def add_child_links(self, child_link):
        """Add child link"""
        if child_link is not None and child_link not in self._child_links:
            self._child_links.append(child_link)

    def del_child_link(self):
        self._child_links = []

    def add_parent_link(self, parent_link):
        self._parent_link = parent_link

    def del_parent_link(self, parent_link):
        self._parent_link = None


class CascadedLink(CascadedCoords):

    def __init__(self,
                 link_list=[],
                 joint_list=[],
                 *args, **kwargs):
        super(CascadedLink, self).__init__(*args, **kwargs)
        self.link_list = link_list
        self.joint_list = joint_list
        self.bodies = []
        self.collision_avoidance_link_list = []
        self.end_coords_list = []
        self.joint_angle_limit_weight_maps = {}

    def angle_vector(self, vec=None, av=None):
        "Returns angle-vector of this object, if vec is given, \
        it updates angles of all joint. If given angle-vector \
        violate min/max range, the value is modified."
        if av is None:
            av = np.zeros(len(self.joint_list), dtype=np.float32)

        for idx, j in enumerate(self.joint_list):
            if vec is not None:
                vec = np.array(vec)
                if not (j.joint_min_max_table is None or
                        j.joint_mix_max_target is None):
                    # currently only 1dof joint is supported
                    if j.joint_dof == 1 and j.joint_min_max_target.joint_dof == 1:
                        # find index of joint-min-max-target
                        ii = 0
                        jj = self.joint_list[ii]

                        while not jj == j.joint_min_max_target:
                            ii += 1
                            jj = self.joint_list[ii]
                        tmp_joint_angle = vec[idx]
                        tmp_target_joint_angle = vec[ii]
                        tmp_joint_min_angle = j.joint_min_max_table_min_angle(
                            tmp_target_joint_angle)
                        tmp_joint_max_angle = j.joint_min_max_table_max_angle(
                            tmp_target_joint_angle)
                        tmp_target_joint_min_angle = j.joint_min_max_table_min_angle(
                            tmp_joint_angle)
                        tmp_target_joint_max_angle = j.joint_min_max_table_max_angle(
                            tmp_joint_angle)

                        if tmp_joint_min_angle <= tmp_joint_angle and tmp_joint_min_angle <= tmp_joint_max_angle:
                            j.joint_angle = tmp_joint_angle
                            jj.joint_angle = tmp_target_joint_angle
                        else:
                            i = 0.0
                            while i > 1.0:
                                tmp_joint_min_angle = j.joint_min_max_table_min_angle(
                                    tmp_target_joint_angle)
                                tmp_joint_max_angle = j.joint_min_max_table_max_angle(
                                    tmp_target_joint_angle)
                                tmp_target_joint_min_angle = j.joint_min_max_table_min_angle(
                                    tmp_joint_angle)
                                tmp_target_joint_max_angle = j.joint_min_max_table_max_angle(
                                    tmp_joint_angle)

                                if tmp_joint_angle < tmp_joint_min_angle:
                                    tmp_joint_angle += (tmp_joint_min_angle -
                                                        tmp_joint_angle) * i
                                if tmp_joint_angle > tmp_joint_max_angle:
                                    tmp_joint_angle += (tmp_joint_max_angle -
                                                        tmp_joint_angle) * i
                                if tmp_target_joint_angle < tmp_target_joint_min_angle:
                                    tmp_target_joint_angle += (
                                        tmp_target_joint_min_angle - tmp_target_joint_angle) * i
                                if tmp_target_joint_angle > tmp_target_joint_max_angle:
                                    tmp_target_joint_angle += (
                                        tmp_target_joint_max_angle - tmp_target_joint_angle) * i
                            j.joint_angle = tmp_joint_angle
                            jj.joint_angle = tmp_target_joint_angle
                            vec[idx] = tmp_joint_angle
                            vec[ii] = tmp_target_joint_angle
                else:
                    if j.joint_dof == 1:
                        j.joint_angle(vec[idx])
                    else:
                        j.joint_angle(vec[idx:idx + j.joint_dof])
            for k in range(j.joint_dof):
                if j.joint_dof == 1:
                    av[idx] = j.joint_angle()
                else:
                    av[idx] = j.joint_angle[k]()
        return av

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = '['
        s += " ".join([j.__str__() for j in self.link_list])
        s += ']'
        return s

    def move_joints(self, union_vel,
                    union_link_list=None,
                    periodic_time=0.05,
                    joint_args=None,
                    move_joints_hook=None,
                    *args, **kwargs):
        if union_link_list is None:
            union_link_list = self.calc_union_link_list(self.link_list)

        dav = self.calc_joint_angle_speed(union_vel, *args, **kwargs)

        tmp_gain = self.calc_joint_angle_speed_gain(
            union_link_list, dav, periodic_time)
        min_gain = 1.0
        for i in range(len(tmp_gain)):
            if tmp_gain[i] < min_gain:
                min_gain = tmp_gain[i]
        dav = min_gain * dav
        i = 0
        l = 0
        while l < len(union_link_list):
            joint = union_link_list[l].joint
            if joint.joint_dof == 1:
                dtheta = joint.speed_to_angle(dav[i])
            else:
                dtheta = joint.speed_to_angle(dav[i:i+joint.joint_dof])
            union_link_list[l].joint.joint_angle(dtheta, relative=True)
            i += joint.joint_dof
            l += 1
        if move_joints_hook:
            for hook in move_joints_hook:
                hook()
        return True

    def collision_avoidance_link_pair_from_link_list(self,
                                                     link_list,
                                                     obstacles=[]):
        return []

    def move_joints_avoidance(self,
                              union_vel,
                              union_link_list=None,
                              link_list=None,
                              n_joint_dimension=None,
                              weight=None,
                              null_space=None,
                              avoid_nspace_gain=0.01,
                              avoid_weight_gain=1.0,
                              avoid_collision_distance=200,
                              avoid_collision_null_gain=1.0,
                              avoid_collision_joint_gain=1.0,
                              collision_avoidance_link_pair=None,
                              cog_gain=0.0,
                              target_centroid_pos=None,
                              centroid_offset_func=None,
                              cog_translation_axis='z',
                              cog_null_space=False,
                              additional_weight_list=None,
                              additional_nspace_list=None,
                              jacobi=None,
                              obstacles=None,
                              *args, **kwargs):
        angle_speed_collision_blending = 0.0
        if n_joint_dimension is None:
            n_joint_dimension = self.calc_target_joint_dimension(union_link_list)
        if weight is None:
            weight = np.ones(n_joint_dimension, dtype=np.float64)
        if jacobi is None:
            logger.error('jacobi is required')
            return True
        if collision_avoidance_link_pair is None:
            self.collision_avoidance_link_pair_from_link_list(
                link_list,
                obstacles=obstacles)

        weight = self.calc_inverse_kinematics_weight_from_link_list(
            link_list, weight=weight,
            n_joint_dimension=n_joint_dimension,
            avoid_weight_gain=avoid_weight_gain,
            union_link_list=union_link_list,
            additional_weight_list=additional_weight_list
        )

        # calc inverse jacobian and projection jacobian
        j_sharp = self.calc_inverse_jacobian(jacobi,
                                             weight=weight,
                                             *args, **kwargs)

        #
        # angle-speed-collision-avoidance: avoiding self collision
        #
        # qca = J#t a dx + ( I - J# J ) Jt b dx
        #
        # dx = p                     if |p| > d_yz
        #    = (dyz / |p|  - 1) p    else
        #
        # a : avoid-collision-joint-gain
        # b : avoid-collision-null-gain
        #
        # implimentation issue:
        # when link and object are collide,
        #  p = (nearestpoint_on_object_surface - center_of_link )
        # else
        #  p =  (nearestpoint_on_object_surface - neareset_point_on_link_surface)
        #
        # H. Sugiura, M. Gienger, H. Janssen, C. Goerick: "Real-Time Self
        # Collision Avoidance for Humanoids by means of Nullspace Criteria
        # and Task Intervals" In Humanoids 2006.
        #
        # H. Sugiura, M. Gienger, H. Janssen and C. Goerick : "Real-Time
        # Collision Avoidance with Whole Body Motion Control for Humanoid
        # Robots", In IROS 2007, 2053--2058
        #

        self.collision_avoidance_null_vector = []
        self.collision_avoidance_joint_vector = []

        angle_speed_collision_avoidance = None
        if collision_avoidance_link_pair is not None \
           and avoid_collision_distance > 0.0 \
           and (avoid_collision_joint_gain > 0.0 or
                avoid_collision_null_gain > 0.0):
            angle_speed_collision_avoidance = \
                self.collision_avoidance(avoid_collision_distance=avoid_collision_distance,
                                         avoid_collision_null_gain=avoid_collision_null_gain,
                                         avoid_collision_joint_gain=avoid_collision_joint_gain,
                                         weight=weight,
                                         collision_avoidance_link_pair=collision_avoidance_link_pair,
                                         *args, **kwargs)
            # (setq min-distance (car (elt (send self :get :collision-distance) 0)))
            # (setq angle-speed-collision-blending
            #  (cond ((<= avoid-collision-joint-gain 0.0) 0.0)
            #   ((< min-distance (* 0.1 avoid-collision-distance))
            #    1.0)
            #   ((< min-distance avoid-collision-distance)
            #    (/ (- avoid-collision-distance min-distance)
            #     (* 0.9 avoid-collision-distance)))
            #   (t
            #    0.0)))

        tmp_nspace = None
        # tmp_nspace = self.calc_inverse_kinematics_nspace_from_link_list(
        #     link_list, union_link_list=union_link_list,
        #     avoid_nspace_gain=avoid_nspace_gain,
        #     cog_gain=cog_gain, target_centroid_pos=target_centroid_pos,
        #     centroid_offset_func=centroid_offset_func,
        #     cog_null_space=cog_null_space,
        #     update_mass_properties=False,
        #     cog_translation_axis=cog_translation_axis,
        #     weight=weight,
        #     n_joint_dimension=n_joint_dimension,
        #     null_space=null_space,
        #     additional_nspace_list=additional_nspace_list)

        # if len(self.collision_avoidance_null_vector):
        #     tmp_nspace += self.collision_avoidance_null_vector

        #
        # q = f(d) qca + {1 - f(d)} qwbm
        #
        # f(d) = (d - da) / (db - da), if d < da
        #      = 0                   , otherwise
        # da : avoid-collision-distance
        # db : avoid-collision-distance*0.1
        #
        # H. Sugiura, IROS 2007
        #
        # qwbm = J# x + N W y
        #
        # J# = W Jt(J W Jt + kI)-1 (Weighted SR-Inverse)
        # N  = E - J#J
        #
        # SR-inverse :
        # Y. Nakamura and H. Hanafusa : "Inverse Kinematic Solutions With
        # Singularity Robustness for Robot Manipulator Control"
        # J. Dyn. Sys., Meas., Control  1986. vol 108, Issue 3, pp. 163--172.
        #
        self.move_joints(union_vel,
                         union_link_list=union_link_list,
                         null_space=tmp_nspace,
                         angle_speed=angle_speed_collision_avoidance,
                         angle_speed_blending=angle_speed_collision_blending,
                         weight=weight,
                         jacobi=jacobi,
                         j_sharp=j_sharp,
                         *args, **kwargs)

    def calc_vel_from_pos(self, dif_pos, translation_axis,
                          p_limit=100.0):
        """

        Parameters
        ----------
        dif_pos : np.ndarray
            [mm] order
        translation_axis : str
            see calc_dif_with_axis

        Returns
        -------
        vel_p : np.ndarray
        """
        if LA.norm(dif_pos) > p_limit:
            dif_pos = p_limit * normalize_vector(dif_pos)
        dif_pos = 0.001 * dif_pos  # scale [mm] -> [m]
        vel_p = calc_dif_with_axis(dif_pos, translation_axis)
        return vel_p

    def calc_vel_from_rot(self,
                          dif_rot,
                          rotation_axis,
                          r_limit=0.5):
        if LA.norm(dif_rot) > r_limit:
            dif_rot = r_limit * normalize_vector(dif_rot)
        vel_r = calc_dif_with_axis(dif_rot, rotation_axis)
        return vel_r

    def find_joint_angle_limit_weight_from_union_link_list(
            self, union_link_list):
        names = tuple(set([link.name for link in union_link_list]))
        if names in self.joint_angle_limit_weight_maps:
            return self.joint_angle_limit_weight_maps[names]
        else:
            return (names, False)

    def reset_joint_angle_limit_weight(self, union_link_list):
        names, weights = self.joint_angle_limit_weight_maps(union_link_list)
        if weights is not False:
            self.joint_angle_limit_weight_maps[names] = (names, False)

    def calc_weight_from_joint_limit(
            self,
            avoid_weight_gain,
            link_list,
            union_link_list,
            weight,
            n_joint_dimension=None):
        """Calculate weight according to joint limit

        """
        if n_joint_dimension is None:
            n_joint_dimension = self.calc_target_joint_dimension(
                union_link_list)

        link_names, previous_joint_angle_limit_weight = self.find_joint_angle_limit_weight_from_union_link_list(
            union_link_list)
        if previous_joint_angle_limit_weight is False:
            previous_joint_angle_limit_weight = np.inf * np.ones(n_joint_dimension, 'f')
            self.joint_angle_limit_weight_maps[link_names] = (
                link_names, previous_joint_angle_limit_weight)

        new_weight = np.zeros_like(weight, 'f')
        joint_list = [l.joint for l in union_link_list
                      if l.joint is not None]
        if avoid_weight_gain > 0.0:
            current_joint_angle_limit_weight = avoid_weight_gain * \
                joint_angle_limit_weight(joint_list)
            for i in range(n_joint_dimension):
                if (current_joint_angle_limit_weight[i] -
                    previous_joint_angle_limit_weight[i]) >= 0.0:
                    new_weight[i] = 1.0 / \
                        (1.0 + current_joint_angle_limit_weight[i])
                else:
                    new_weight[i] = 1.0
                previous_joint_angle_limit_weight[i] = \
                    current_joint_angle_limit_weight[i]
        elif avoid_weight_gain == 0.0:
            for i in range(n_joint_dimension):
                new_weight[i] = weight[i]

        w_cnt = 0
        for ul in union_link_list:
            dof = ul.joint.joint_dof
            n_duplicate = sum([1 for x in link_list if ul in x])
            if n_duplicate > 1:
                for i in range(dof):
                    new_weight[w_cnt + i] = new_weight[w_cnt + i] / n_duplicate
            w_cnt += dof
        return new_weight

    def calc_inverse_kinematics_weight_from_link_list(
            self,
            link_list,
            avoid_weight_gain=1.0,
            union_link_list=None,
            n_joint_dimension=None,
            weight=None,
            additional_weight_list=[]):
        """Calculate all weight from link list

        """
        if not (isinstance(link_list[0], tuple) or
                isinstance(link_list[0], list)):
            link_list = [link_list]
        if union_link_list is None:
            union_link_list = self.calc_union_link_list(link_list)
        if n_joint_dimension is None:
            n_joint_dimension = self.calc_target_joint_dimension(
                union_link_list)
        if weight is None:
            weight = np.ones(n_joint_dimension, 'f')

        # TODO support additional_weight_list

        tmp_weight = self.calc_weight_from_joint_limit(
            avoid_weight_gain,
            link_list,
            union_link_list,
            weight,
            n_joint_dimension=n_joint_dimension)
        for i in range(n_joint_dimension):
            tmp_weight[i] = weight[i] * tmp_weight[i]
        return tmp_weight

    def inverse_kinematics_loop(
            self,
            dif_pos, dif_rot,
            stop=1,
            loop=0,
            link_list=None,
            move_target=None,
            rotation_axis=True,
            translation_axis=True,
            thre=None,
            rthre=None,
            dif_pos_ratio=1.0,
            dif_rot_ratio=1.0,
            union_link_list=None,
            target_coords=None,
            jacobi=None,
            additional_check=None,
            additional_jacobi=None,
            additional_vel=None,
            centroid_thre=1.0,
            target_centroid_pos=None,
            centroid_offset_func=None,
            cog_translation_axis='z',
            cog_null_space=None,
            cog_gain=1.0,
            min_loop=None,
            inverse_kinematics_hook=[],
            **kwargs):
        """

        inverse-kinematics-loop is one loop calculation for inverse_kinematics.
        In this method, joint position difference satisfying workspace difference (dif_pos, dif_rot) are
        calculated and joint angles are updated.

        """
        if move_target is None:
            raise NotImplementedError
        move_target = listify(move_target)

        n = len(move_target)
        if rotation_axis is None:
            rotation_axis = listify(True, n)
        if translation_axis is None:
            translation_axis = listify(True, n)
        if thre is None:
            thre = listify(1, n)
        if rthre is None:
            rthre = listify(np.deg2rad(1), n)
        if min_loop is None:
            min_loop = stop // 10

        # (if target-centroid-pos (send self :update-mass-properties))
        # ;; dif-pos, dif-rot, move-target, rotation-axis, translation-axis, link-list
        # ;; -> both list and atom OK.
        tmp_dim = []
        tmp_dims = []
        vec_count = 0
        success = True
        p_limit = kwargs.pop('p_limit', None)
        r_limit = kwargs.pop('r_limit', None)
        if union_link_list is None:
            union_link_list = self.calc_union_link_list(link_list)
            # ;; tempolary reset weight for prevision joint-angle-limit-weight-old using hash-table
            # (if (/= (length union-link-list) (length link-list))
            #     (send self :reset-joint-angle-limit-weight-old union-link-list)))
        if link_list is None or not isinstance(link_list[0], list):
            link_list = [link_list]
        move_target = listify(move_target)
        target_coords = listify(target_coords)
        dif_pos = listify(dif_pos)
        dif_rot = listify(dif_rot)
        rotation_axis = listify(rotation_axis)
        translation_axis = listify(translation_axis)
        thre = listify(thre)
        rthre = listify(rthre)

        # argument check
        if not (len(translation_axis) ==
                len(rotation_axis) ==
                len(move_target) ==
                len(link_list) ==
                len(dif_pos) ==
                len(dif_rot)):
            logger.error("list length differ : translation-axis {} rotation-axis {} "
                         "move-target {} link-list {} dif-pos {} dif-rot".
                         format(len(translation_axis),
                                len(rotation_axis),
                                len(move_target),
                                len(link_list),
                                len(dif_pos),
                                len(dif_rot)))
            return "ik-continuous"

        for i in range(len(rotation_axis)):
            tmp_dims.append(np.zeros(self.calc_target_axis_dimension(
                rotation_axis[i], translation_axis[i]), 'f'))
        tmp_dim = np.zeros(self.calc_target_axis_dimension(
            rotation_axis, translation_axis), 'f')
        # (if (memq :tmp-dims ik-args)
        #     (setq tmp-dims (cadr (memq :tmp-dims ik-args)))
        #   (progn
        #     (dotimes (i (length rotation-axis))
        #       (push (instantiate float-vector (send self :calc-target-axis-dimension (elt rotation-axis i) (elt translation-axis i))) tmp-dims))
        #     (setq tmp-dims (nreverse tmp-dims))))
        # (if (memq :tmp-dim ik-args)
        #     (setq tmp-dim (cadr (memq :tmp-dim ik-args)))
        #   (setq tmp-dim (instantiate float-vector (send self :calc-target-axis-dimension rotation-axis translation-axis))))

        if callable(jacobi):
            jacobi = jacobi(
                link_list,
                move_target,
                translation_axis,
                rotation_axis)
        if jacobi is None:
            jacobi = self.calc_jacobian_from_link_list(
                move_target=move_target,
                link_list=link_list,
                translation_axis=translation_axis,
                rotation_axis=rotation_axis,
                # method_args
            )

        # convergence check
        if min_loop is not None:
            success = loop >= min_loop

        if success is True:
            success = self.ik_convergence_check(
                # loop >= min_loop if min_loop is not None else True,
                dif_pos,
                dif_rot,
                rotation_axis,
                translation_axis,
                thre,
                rthre,
                centroid_thre,
                target_centroid_pos,
                centroid_offset_func,
                cog_translation_axis,
                # update_mass_properties=False
            )
        if additional_check is not None:
            success |= additional_check()

        # calculation of move-coords velocities from vel-p and vel-r
        for i in range(len(move_target)):
            tmp_dim = tmp_dims[i]
            if p_limit is not None:
                vel_p = self.calc_vel_from_pos(
                    dif_pos[i], translation_axis[i], p_limit)
            else:
                vel_p = self.calc_vel_from_pos(
                    dif_pos[i], translation_axis[i])
            if r_limit is not None:
                vel_r = self.calc_vel_from_rot(
                    dif_rot[i], rotation_axis[i], r_limit)
            else:
                vel_r = self.calc_vel_from_rot(
                    dif_rot[i], rotation_axis[i])
            for j in range(len(vel_p)):
                tmp_dim[j] = dif_pos_ratio * vel_p[j]
            for j in range(len(vel_r)):
                tmp_dim[j + len(vel_p)] = dif_rot_ratio * vel_r[j]
            # (when (send self :get :ik-target-error)
            #   (push (list vel-p vel-r) (cdr (assoc (read-from-string (format nil ":target-~d" i)) (send self :get :ik-target-error)))))

        vec_count = 0
        for i in range(len(tmp_dims)):
            for j in range(len(tmp_dims[i])):
                tmp_dim[j + vec_count] = tmp_dims[i][j]
            vec_count += len(tmp_dims[i])

        # Use cog jacobian as first task
        # (when (and (not cog-null-space) target-centroid-pos)
        #   (setq additional-jacobi
        #         (append (list (send self :calc-cog-jacobian-from-link-list
        #                             :update-mass-properties nil
        #                             :link-list union-link-list
        #                             :translation-axis cog-translation-axis))
        #                 additional-jacobi))
        #   (let ((tmp-cog-gain (if (> cog-gain 1.0) 1.0 cog-gain)))
        #     (setq additional-vel
        #           (append (list
        #                    (send self :calc-vel-for-cog tmp-cog-gain cog-translation-axis target-centroid-pos
        #                          :centroid-offset-func centroid-offset-func
        #                          :update-mass-properties nil))
        #                   additional-vel))
        #     (when (send self :get :ik-target-error)
        #       (push (car additional-vel) (cdr (assoc :centroid (send self :get :ik-target-error)))))
        #     ))

        # append additional-jacobi and additional-vel
        if additional_jacobi is not None:
            additional_velocity_list = list(
                map(lambda x:
                    x(link_list) if callable(x) else x,
                    additional_vel))
            row0 = len(tmp_dim) - sum(map(len, additional_velocity_list))
            for i_add_jacobi in range(len(additional_jacobi)):
                add_jacobi = additional_jacobi[i_add_jacobi]
                if callable(add_jacobi):
                    add_jacobi = add_jacobi(link_list)
                add_vel = additional_velocity_list[i_add_jacobi]
                for i_row in range(add_jacobi.shape[0]):
                    # set additional-jacobi
                    for i_col in range(add_jacobi.shape[1]):
                        jacobi[row0 + i_row][i_col] = add_jacobi[i_row][i_col]
                    tmp_dim[row0 + i_row] = add_vel[i_row]
                row0 += len(add_vel)

        # check loop end
        if success:
            return 'ik-succeed'

        for hook in inverse_kinematics_hook:
            hook()
        self.collision_pair_list = None
        self.move_joints_avoidance(
            tmp_dim,
            union_link_list=union_link_list,
            link_list=link_list,
            rotation_axis=rotation_axis,
            translation_axis=translation_axis,
            jacobi=jacobi,
            **kwargs)
        return "ik-continuous"

    def inverse_kinematics_args(
            self,
            union_link_list=None,
            rotation_axis=None,
            translation_axis=None,
            additional_jacobi_dimension=None,
            **kwargs):
        c = self.calc_target_joint_dimension(
            union_link_list)
        # add dimensions of additonal-jacobi
        r = self.calc_target_axis_dimension(
            rotation_axis, translation_axis) + additional_jacobi_dimension

        fik = make_matrix(r, c)
        ret = make_matrix(c, r)

        tmp_dims = []
        for ta, ra in zip(translation_axis, rotation_axis):
            tmp_dims.append(np.zeros
                            (self.calc_target_axis_dimension(ra, ta),
                             'f'))
        return dict(dim=r,
                    fik=fik,
                    n_joint_dimension=c,
                    ret=ret,
                    **kwargs)

    def inverse_kinematics(
            self,
            target_coords,
            stop=50,
            link_list=None,
            move_target=None,
            revert_if_fail=True,
            rotation_axis=True,
            translation_axis=True,
            joint_args=None,
            thre=None,
            rthre=None,
            union_link_list=None,
            centroid_thre=1.0,
            target_centroid_pos=None,
            centroid_offset_func=None,
            cog_translation_axis='z',
            cog_null_space=False,
            periodic_time=0.5,
            check_collision=None,
            additional_jacobi=[],
            additional_vel=[],
            **kwargs):
        target_coords = listify(target_coords)
        if callable(union_link_list):
            union_link_list = union_link_list(link_list)
        else:
            union_link_list = self.calc_union_link_list(link_list)

        if thre is None:
            if isinstance(move_target, list):
                thre = [1] * len(move_target)
            else:
                thre = [1]
        if rthre is None:
            if isinstance(move_target, list):
                rthre = [np.deg2rad(1)] * len(move_target)
            else:
                rthre = [np.deg2rad(1)]

        # store current angle vector
        joint_list = list(set([l.joint for l in union_link_list] + self.joint_list))
        av0 = [j.joint_angle() for j in joint_list]
        c0 = None
        if self.parent is None:
            c0 = self.copy_worldcoords()
        target_coords0 = target_coords
        success = True
        # (old-analysis-level (send-all union-link-list :analysis-level))
        # (send-all union-link-list :analysis-level :coords)

        # argument check
        if link_list is None or move_target is None:
            logger.error("both :link-list and :move-target required")
            return True
        if (translation_axis is None or translation_axis is False) and \
           (rotation_axis is None or rotation_axis is False):
            return True
        if not isinstance(link_list[0], list):
            link_list = [link_list]
        move_target = listify(move_target)
        rotation_axis = listify(rotation_axis)
        translation_axis = listify(translation_axis)
        thre = listify(thre)
        rthre = listify(rthre)

        if not (len(translation_axis) ==
                len(rotation_axis) ==
                len(move_target) ==
                len(link_list) ==
                len(target_coords)):
            logger.error('list length differ : translation_axis {}'
                         ', rotation_axis {}, move_target {} '
                         'link_list {}, target_coords {}'.format(
                             len(translation_axis), len(rotation_axis),
                             len(move_target), len(link_list),
                             len(target_coords)))
            return False

        if not(len(additional_jacobi) ==
               len(additional_vel)):
            logger.error('list length differ : additional_jacobi {}, '
                         'additional_vel {}'.format(
                             len(additional_jacobi), len(additional_vel)))
            return False

        tmp_additional_jacobi = map(
            lambda aj: aj(link_list) if callable(aj) else aj,
            additional_jacobi)
        if cog_null_space is None and target_centroid_pos is not None:
            additional_jacobi_dimension = self.calc_target_axis_dimension(
                False, cog_translation_axis)
        else:
            additional_jacobi_dimension = 0
        additional_jacobi_dimension += sum(
            map(lambda aj: (aj(link_list) if callable(aj) else aj).shape[0], tmp_additional_jacobi))
        ik_args = self.inverse_kinematics_args(
            union_link_list=union_link_list,
            translation_axis=translation_axis,
            rotation_axis=rotation_axis,
            # evaluate additional-jacobi function and
            # calculate row dimension of additional_jacobi
            additional_jacobi_dimension=additional_jacobi_dimension,
            **kwargs)
        #    self.reset_joint_angle_limit_weight_old(union_link_list) ;; reset weight

        # inverse_kinematics loop
        loop = 0
        while loop < stop:
            loop += 1
            target_coords = list(map(
                lambda x: x() if callable(x) else x,
                target_coords))
            dif_pos = list(map(lambda mv, tc, trans_axis:
                               1000 *
                               mv.difference_position(
                                   tc, translation_axis=trans_axis),
                               move_target, target_coords, translation_axis))
            dif_rot = list(map(lambda mv, tc, rot_axis:
                               mv.difference_rotation(
                                   tc, rotation_axis=rot_axis),
                               move_target, target_coords, rotation_axis))

            success = self.inverse_kinematics_loop(
                dif_pos, dif_rot,
                target_coords=target_coords,
                periodic_time=periodic_time,
                stop=stop,
                loop=loop,
                rotation_axis=rotation_axis,
                translation_axis=translation_axis,
                move_target=move_target,
                link_list=link_list,
                union_link_list=union_link_list,
                thre=thre,
                rthre=rthre,
                additional_jacobi=additional_jacobi,
                additional_vel=additional_vel,
                ik_args=ik_args)
            if success == 'ik-succeed':
                return success

        if target_centroid_pos is not None:
            self.update_mass_properties()

        target_coords = list(map(lambda x: x() if callable(x) else x,
                                 target_coords))
        dif_pos = list(map(lambda mv, tc, trans_axis:
                           mv.difference_position(
                               tc, translation_axis=trans_axis),
                           move_target, target_coords, translation_axis))
        dif_rot = list(map(lambda mv, tc, rot_axis:
                           mv.difference_rotation(tc, rotation_axis=rot_axis),
                           move_target, target_coords, rotation_axis))

        # success
        success = self.ik_convergence_check(dif_pos, dif_rot,
                                            rotation_axis, translation_axis,
                                            thre, rthre, centroid_thre, target_centroid_pos,
                                            centroid_offset_func, cog_translation_axis,
                                            update_mass_properties=False)
        # update difference
        # (mapcar #'(lambda (l a) (send l :analysis-level a)) union-link-list old-analysis-level)

        # ;; reset weight
        #    (send self :reset-joint-angle-limit-weight-old union-link-list)
        #    ;; Get final ik success result flag
        #    (if check-collision (setq collision-status (send self :self-collision-check)))
        #    (setq success (and (or success (not revert-if-fail)) (not collision-status)))
        #    ;; check solved or not
        if success:
            return self.angle_vector()

        # reset angle vector
        for joint, angle in zip(joint_list, av0):
            joint.joint_angle(angle)
        if c0 is not None:
            self.newcoords(c0)
        return False

    def ik_convergence_check(self,
                             dif_pos,
                             dif_rot,
                             rotation_axis,
                             translation_axis,
                             thre, rthre,
                             centroid_thre=None, target_centroid_pos=None,
                             centroid_offset_func=None, cog_translation_axis=None,
                             update_mass_properties=True):
        """

        check ik convergence

        Parameters
        ----------
        dif_pos : list of np.ndarray
        dif_rot : list of np.ndarray

        translation_axis : list of axis
        rotation_axis : list of axis
            see _wrap_axis
        """
        translation_axis = list(map(_wrap_axis, translation_axis))
        rotation_axis = list(map(_wrap_axis, rotation_axis))

        for i in range(len(dif_pos)):
            if LA.norm(dif_pos[i]) > thre[i]:
                return False
        for i in range(len(dif_rot)):
            if LA.norm(dif_rot[i]) > rthre[i]:
                return False
        if target_centroid_pos is not None:
            raise NotImplementedError
        return True

    def inverse_kinematics_optimization(self,
                                        target_coords,
                                        move_target=None,
                                        link_list=None,
                                        regularization_parameter=None,
                                        init_angle_vector=None,
                                        translation_axis=True,
                                        rotation_axis=True,
                                        stop=100,
                                        dt=5e-3,
                                        inverse_kinematics_hook=[],
                                        thre=1.0,
                                        rthre=np.deg2rad(1.0),
                                        *args, **kwargs):

        if not isinstance(target_coords, list):
            target_coords = [target_coords]
        if not isinstance(move_target, list):
            move_target = [move_target]
        if not isinstance(rotation_axis, list):
            rotation_axis = [rotation_axis]
        if not isinstance(translation_axis, list):
            translation_axis = [translation_axis]
        if not (len(move_target) ==
                len(rotation_axis) ==
                len(translation_axis) ==
                len(target_coords)):
            logger.error("list length differ : target_coords {} translation_axis {} \
            rotation_axis {} move_target {}".
                         format(len(target_coords),
                                len(translation_axis),
                                len(rotation_axis),
                                len(move_target)))

        union_link_list = self.calc_union_link_list(link_list)
        for i in range(stop):
            qd = self.compute_velocity(target_coords,
                                       move_target,
                                       dt,
                                       link_list=link_list,
                                       gain=0.85,
                                       weight=1.0,
                                       translation_axis=rotation_axis,
                                       rotation_axis=translation_axis,
                                       dof_limit_gain=0.5,
                                       *args, **kwargs)
            for l, dtheta in zip(union_link_list, qd):
                l.joint.joint_angle(dtheta,
                                    relative=True)

            success = True
            for mv, tc, trans_axis, rot_axis in zip(move_target,
                                                    target_coords,
                                                    translation_axis,
                                                    rotation_axis):
                dif_pos = mv.difference_position(tc,
                                                 translation_axis=trans_axis)
                dif_rot = mv.difference_rotation(tc,
                                                 rotation_axis=rot_axis)
                dif_pos *= 1000.0

                if translation_axis is not None:
                    success = success and (LA.norm(dif_pos) < thre)
                if rotation_axis is not None:
                    success = success and (LA.norm(dif_rot) < rthre)

            if success:
                break

            for hook in inverse_kinematics_hook:
                hook()

        return success

    def calc_inverse_jacobian(self, jacobi,
                              manipulability_limit=0.1,
                              manipulability_gain=0.001,
                              weight=None,
                              *args, **kwargs):
        # m : manipulability
        m = manipulability(jacobi)
        k = 0
        if m < manipulability_limit:
            k = manipulability_gain * ((1.0 - m / manipulability_limit) ** 2)
        # calc weighted SR-inverse
        j_sharp = sr_inverse(jacobi, k, weight)
        return j_sharp

    def calc_joint_angle_speed_gain(self, union_link_list,
                                    dav,
                                    periodic_time):
        n_joint_dimension = self.calc_target_joint_dimension(union_link_list)
        av = np.zeros(n_joint_dimension)
        i = 0
        l = 0
        while l < len(union_link_list):
            j = union_link_list[l].joint
            for k in range(j.joint_dof):
                av[i + k] = j.calc_angle_speed_gain(dav, i, periodic_time)
            i += j.joint_dof
            l += 1
        return av

    def calc_joint_angle_speed(self,
                               union_vel,
                               angle_speed=None,
                               angle_speed_blending=0.5,
                               jacobi=None,
                               j_sharp=None,
                               null_space=None,
                               *args, **kwargs):
        # argument check
        if jacobi is None and j_sharp is None:
            logger.warn(
                'jacobi(j) or j_sharp(J#) is required in calc_joint_angle_speed')
            return null_space
        n_joint_dimension = jacobi.shape[1]

        # dav = J#x + (I - J#J)y
        # calculate J#x
        j_sharp_x = np.dot(j_sharp, union_vel)

        # add angle-speed to J#x using angle-speed-blending
        if angle_speed is not None:
            j_sharp_x = midpoint(angle_speed_blending,
                                 j_sharp_x,
                                 angle_speed)
        # if use null space
        if ((isinstance(null_space, list) or isinstance(null_space, np.ndarray))
                and n_joint_dimension == null_space):
            I = np.eye(n_joint_dimension)
            j_sharp_x += np.matmul(I - np.matmul(j_sharp, jacobi),
                                   null_space)
        return j_sharp_x

    def calc_target_axis_dimension(self, rotation_axis, translation_axis):
        """rotation-axis, translation-axis -> both list and atom OK."""
        if not isinstance(translation_axis, list):
            translation_axis = [translation_axis]
        if not isinstance(rotation_axis, list):
            rotation_axis = [rotation_axis]
            dim = 6 * 1
        else:
            dim = 6 * len(rotation_axis)
        if len(rotation_axis) != len(translation_axis):
            raise ValueError('list length differ: '
                             'len(translation_axis)!=len(rotation_axis) '
                             '{}!={}'.
                             format(len(rotation_axis),
                                    len(translation_axis)))

        for axis in itertools.chain(translation_axis, rotation_axis):
            if axis in ['x', 'y', 'z', 'xx', 'yy', 'zz']:
                dim -= 1
            elif axis in ['xy', 'yx', 'yz', 'zy', 'zx', 'xz']:
                dim -= 2
            elif axis is False or axis is None:
                dim -= 3
        return dim

    def compute_qp_common(self,
                          target_coords,
                          move_target,
                          dt,
                          link_list=None,
                          gain=0.85,
                          weight=1.0,
                          translation_axis=True,
                          rotation_axis=True,
                          dof_limit_gain=0.5):
        if not isinstance(target_coords, list):
            target_coords = [target_coords]
        if not isinstance(move_target, list):
            move_target = [move_target]
        if link_list is None:
            link_list = self.link_list
        if not isinstance(link_list, list):
            link_list = [link_list]
        n_target = len(target_coords)

        if not isinstance(translation_axis, list):
            translation_axis = [translation_axis] * n_target
        if not isinstance(rotation_axis, list):
            rotation_axis = [rotation_axis] * n_target
        if not isinstance(weight, list):
            weight = [weight] * n_target
        if not isinstance(gain, list):
            gain = [gain] * n_target

        union_link_list = self.calc_union_link_list(link_list)
        n = len(union_link_list)
        q = np.array(list(map(lambda l: l.joint.joint_angle(),
                              union_link_list)))
        P = np.zeros((n, n))
        v = np.zeros(n)

        J = self.calc_jacobian_from_link_list(move_target=move_target,
                                              link_list=link_list,
                                              translation_axis=translation_axis,
                                              rotation_axis=rotation_axis)
        union_vels = np.array([])
        for mv, tc, trans_axis, rot_axis, w, g in zip(move_target,
                                                      target_coords,
                                                      translation_axis,
                                                      rotation_axis,
                                                      weight,
                                                      gain):
            # TODO duplicate of jacobian based ik
            dif_pos = mv.difference_position(tc,
                                             translation_axis=trans_axis)
            dif_rot = mv.difference_rotation(tc,
                                             rotation_axis=rot_axis)
            dif_pos *= 1000.0
            vel_pos = self.calc_vel_from_pos(dif_pos, trans_axis)
            vel_rot = self.calc_vel_from_rot(dif_rot, rot_axis)
            union_vel = np.concatenate([vel_pos, vel_rot])
            union_vels = np.concatenate([union_vels, union_vel])
        r = g * union_vels
        P += w * np.dot(J.T, J)
        v += w * np.dot(-r.T, J)

        q_max = np.array(list(map(lambda l: l.joint.max_angle,
                                  union_link_list)))
        q_min = np.array(list(map(lambda l: l.joint.min_angle,
                                  union_link_list)))
        qd_max_dof_limit = dof_limit_gain * (q_max - q) / dt
        qd_min_dof_limit = dof_limit_gain * (q_min - q) / dt
        qd_max = np.minimum(np.ones(n), qd_max_dof_limit)
        qd_min = np.maximum(- np.ones(n), qd_min_dof_limit)
        return P, v, qd_max, qd_min

    def compute_velocity(self,
                         target_coords,
                         move_target,
                         dt,
                         link_list=None,
                         gain=0.85,
                         weight=1.0,
                         translation_axis=True,
                         rotation_axis=True,
                         dof_limit_gain=0.5,
                         fast=True,
                         sym_proj=False,
                         solver='cvxopt',
                         *args, **kwargs):
        if not isinstance(target_coords, list):
            target_coords = [target_coords]
        n_target = len(target_coords)
        if not isinstance(move_target, list):
            move_target = [move_target]
        if link_list is None:
            link_list = self.link_list
        if not isinstance(link_list, list):
            link_list = [link_list]
        if not isinstance(translation_axis, list):
            translation_axis = [translation_axis] * n_target
        if not isinstance(rotation_axis, list):
            rotation_axis = [rotation_axis] * n_target
        if not isinstance(weight, list):
            weight = [weight] * n_target
        if not isinstance(gain, list):
            gain = [gain] * n_target

        union_link_list = self.calc_union_link_list(link_list)

        n = len(union_link_list)

        if fast:
            P, v, qd_max, qd_min = self.compute_qp_common(
                target_coords,
                move_target,
                dt,
                link_list,
                gain=gain,
                weight=weight,
                translation_axis=translation_axis,
                rotation_axis=rotation_axis,
                dof_limit_gain=dof_limit_gain)
            G = np.vstack([+ np.eye(n), -np.eye(n)])
            h = np.hstack([qd_max, -qd_min])
        else:
            E = np.eye(n)
            Z = np.zeros((n, n))
            P0, v0, qd_max, qd_min = self.compute_qp_common(
                target_coords,
                move_target,
                dt,
                link_list,
                gain=gain,
                weight=weight,
                translation_axis=translation_axis,
                rotation_axis=rotation_axis,
                dof_limit_gain=dof_limit_gain)
            margin_reg = 1e-5
            margin_lin = 1e-3
            P = np.vstack([np.hstack([P0, Z]),
                           np.hstack([Z, margin_reg * E])])
            v = np.hstack([v0, -margin_lin * np.ones(n)])
            G = np.vstack([
                np.hstack([+E, +E / dt]),
                np.hstack([-E, +E / dt]),
                np.hstack([Z, -E])])
            h = np.hstack([qd_max, -qd_min, np.zeros(n)])

        qd = solve_qp(P,
                      v,
                      G,
                      h,
                      sym_proj=sym_proj,
                      solver=solver)
        return qd

    def find_link_route(self, to, frm=None):
        def _check_type(obj):
            if obj is not None and not isinstance(obj, Link):
                raise TypeError('Support only Link class. '
                                'get type=={}'.format(type(obj)))
        _check_type(to)
        _check_type(frm)
        pl = to.parent_link
        # if to is not included in self.link_list, just trace parent-link
        if pl and not (to in self.link_list):
            return self.find_link_route(pl, frm)
        # if self.link_list, append "to" link
        if pl and not (to == frm):
            return self.find_link_route(pl, frm) + [to]
        # if link_route, just return "frm" link
        if pl and to == frm:
            return [frm]

        # parent is None
        return []

    def link_lists(self, to, frm=None):
        """

        Find link list from to link to frm link.

        """
        ret1 = self.find_link_route(to, frm)
        if frm and not ret1[0] == frm:
            ret2 = self.find_link_route(frm, ret1[0])
            ret1 = ret2[::-1] + ret1
        return ret1

    def calc_union_link_list(self, link_list):
        if not isinstance(link_list[0], list):
            return link_list
        elif len(link_list) == 1:
            return link_list[0]
        else:
            return list(OrderedSet(list(itertools.chain(*link_list))))

    def calc_target_joint_dimension(self, link_list):
        return calc_target_joint_dimension(map(lambda l: l.joint,
                                               self.calc_union_link_list(link_list)))

    def calc_jacobian_from_link_list(self,
                                     move_target,
                                     link_list=None,
                                     transform_coords=None,
                                     rotation_axis=None,
                                     translation_axis=None,
                                     col_offset=0,
                                     dim=None,
                                     fik=None,
                                     n_joint_dimension=None,
                                     *args, **kwargs):
        if link_list is None:
            link_list = self.link_list
        if rotation_axis is None:
            if not isinstance(move_target, list):
                rotation_axis = False
            else:
                rotation_axis = [False] * len(move_target)
        if translation_axis is None:
            if not isinstance(move_target, list):
                translation_axis = True
            else:
                translation_axis = [True] * len(move_target)
        if transform_coords is None:
            transform_coords = move_target
        if dim is None:
            dim = self.calc_target_axis_dimension(
                rotation_axis, translation_axis)
        if n_joint_dimension is None:
            n_joint_dimension = self.calc_target_joint_dimension(link_list)
        if fik is None:
            fik = np.zeros((dim, n_joint_dimension), dtype=np.float32)

        union_link_list = self.calc_union_link_list(link_list)
        jdim = self.calc_target_joint_dimension(union_link_list)
        if not isinstance(link_list[0], list):
            link_lists = [link_list]
        else:
            link_lists = link_list
        if not isinstance(move_target, list):
            move_targets = [move_target]
        else:
            move_targets = move_target
        if not isinstance(transform_coords, list):
            transform_coords = [transform_coords]
        if not isinstance(rotation_axis, list):
            rotation_axes = [rotation_axis]
        else:
            rotation_axes = rotation_axis
        if not isinstance(translation_axis, list):
            translation_axes = [translation_axis]
        else:
            translation_axes = translation_axis

        col = col_offset
        i = 0
        while col < (col_offset + jdim):
            ul = union_link_list[i]
            row = 0

            for link_list, move_target, transform_coord, rotation_axis, translation_axis \
                    in zip(link_lists, move_targets, transform_coords, rotation_axes, translation_axes):
                if ul in link_list:
                    length = len(link_list)
                    l = link_list.index(ul)
                    joint = ul.joint

                    def find_parent(parent_link, link_list):
                        if parent_link is None or parent_link in link_list:
                            return parent_link
                        else:
                            return find_parent(parent_link.parent_link,
                                               link_list)

                    if not isinstance(joint.child_link, Link):
                        child_reverse = False
                    elif ((l + 1 < length) and
                          not joint.child_link == find_parent(link_list[l + 1].parent_link, link_list)) or \
                         ((l + 1 == length) and
                          (not joint.child_link == find_parent(move_target.parent, link_list))):
                        child_reverse = True
                    else:
                        child_reverse = False

                    paxis = _wrap_axis(joint.axis)
                    child_link = joint.child_link
                    parent_link = joint.parent_link
                    default_coords = joint.default_coords
                    world_default_coords = parent_link.copy_worldcoords().transform(
                        default_coords)

                    fik = joint.calc_jacobian(fik, row, col, joint, paxis,
                                              child_link, world_default_coords, child_reverse,
                                              move_target, transform_coord, rotation_axis,
                                              translation_axis)
                row += self.calc_target_axis_dimension(rotation_axis,
                                                       translation_axis)
            col += joint.joint_dof
            i += 1
        return fik

    @property
    def interlocking_joint_pairs(self):
        """Interlocking joint pairs.
        pairs are [(joint0, joint1), ...]
        If users want to use interlocking joints, please overwrite this method.
        """
        return []

    def calc_jacobian_for_interlocking_joints(
            self, link_list, interlocking_joint_pairs=None):
        if interlocking_joint_pairs is None:
            interlocking_joint_pairs = self.interlocking_joint_pairs
        union_link_list = self.calc_union_link_list(link_list)
        joint_list = list(filter(lambda j: j is not None,
                                 [l.joint for l in union_link_list]))
        pairs = list(
            filter(lambda pair:
                   not ((pair[0] not in joint_list) and
                        (pair[1] not in joint_list)),
                   interlocking_joint_pairs))
        jacobi = np.zeros((len(pairs),
                           self.calc_target_joint_dimension(union_link_list)),
                          'f')
        for i, pair in enumerate(pairs):
            index = sum(
                [j.joint_dof for j in joint_list[:joint_list.index(
                    pair[0])]])
            jacobi[i][index] = 1.0
            index = sum(
                [j.joint_dof for j in joint_list[:joint_list.index(
                    pair[1])]])
            jacobi[i][index] = -1.0
        return jacobi

    def calc_vel_for_interlocking_joints(
            self, link_list,
            interlocking_joint_pairs=None):
        """Calculate 0 velocity for keeping
        interlocking joint at the same joint angle."""
        if interlocking_joint_pairs is None:
            interlocking_joint_pairs = self.interlocking_joint_pairs
        union_link_list = self.calc_union_link_list(link_list)
        joint_list = list(filter(lambda j: j is not None,
                                 [l.joint for l in union_link_list]))
        pairs = list(
            filter(lambda pair:
                   not ((pair[0] not in joint_list) and
                        (pair[1] not in joint_list)),
                   interlocking_joint_pairs))
        vel = np.zeros(len(pairs), 'f')
        return vel



class RobotModel(CascadedLink):

    def __init__(self, link_list=[], joint_list=[],
                 root_link=None):
        super(RobotModel, self).__init__(link_list, joint_list)

        self.joint_names = []
        for joint in self.joint_list:
            self.joint_names.append(joint.name)
            joint.child_link.add_parent_link(joint.parent_link)
            joint.parent_link.add_child_links(joint.child_link)

        for link in self.link_list:
            self.__dict__[link.name] = link
        for joint in joint_list:
            self.__dict__[joint.name] = joint
        self.urdf_path = None

    def reset_pose(self):
        raise NotImplementedError()

    def reset_manip_pose(self):
        raise NotImplementedError()

    def init_pose(self):
        return self.angle_vector(np.zeros_like(self.angle_vector()))

    def load_urdf(self, urdf_path):
        self.urdf_path = urdf_path
        self.robot_urdf = URDF.from_xml_string(open(urdf_path).read())
        root_link = self.robot_urdf.link_map[self.robot_urdf.get_root()]

        links = []
        for link in self.robot_urdf.links:
            l = Link(name=link.name)
            links.append(l)

        link_maps = {l.name: l for l in links}

        joint_list = []
        joint_names = []
        for j in self.robot_urdf.joints:
            if j.limit is None:
                j.type = 'fixed'
            if j.origin is None:
                j.type = 'fixed'
            if j.axis is None:
                j.axis = 'z'
            if j.type in ['fixed']:
                joint = FixedJoint(
                    name=j.name,
                    parent_link=link_maps[j.parent],
                    child_link=link_maps[j.child])
            elif j.type == 'revolute':
                joint = RotationalJoint(
                    axis=j.axis,
                    name=j.name,
                    parent_link=link_maps[j.parent],
                    child_link=link_maps[j.child],
                    min_angle=np.rad2deg(j.limit.lower),
                    max_angle=np.rad2deg(j.limit.upper),
                    max_joint_torque=j.limit.effort,
                    max_joint_velocity=j.limit.velocity)
            elif j.type == 'continuous':
                joint = RotationalJoint(
                    axis=j.axis,
                    name=j.name,
                    parent_link=link_maps[j.parent],
                    child_link=link_maps[j.child],
                    min_angle=-np.inf,
                    max_angle=np.inf,
                    max_joint_torque=j.limit.effort,
                    max_joint_velocity=j.limit.velocity)
            elif j.type == 'prismatic':
                # http://wiki.ros.org/urdf/XML/joint
                # meters for prismatic joints
                joint = LinearJoint(
                    axis=j.axis,
                    name=j.name,
                    parent_link=link_maps[j.parent],
                    child_link=link_maps[j.child],
                    min_angle=1000 * j.limit.lower,
                    max_angle=1000 * j.limit.upper,
                    max_joint_torque=j.limit.effort,
                    max_joint_velocity=j.limit.velocity)

            if j.type not in ['fixed']:
                joint_list.append(joint)
                joint_names.append(joint.name)

            link_maps[j.parent].assoc(link_maps[j.child])
            link_maps[j.child].add_joint(joint)
            link_maps[j.child].add_parent_link(link_maps[j.parent])
            link_maps[j.parent].add_child_links(link_maps[j.child])

        for j in self.robot_urdf.joints:
            if j.origin is None:
                rpy = np.zeros(3, dtype=np.float32)
                xyz = np.zeros(3, dtype=np.float32)
            else:
                rpy = np.array(j.origin.rpy, dtype=np.float32)[::-1]
                xyz = np.array(j.origin.xyz, dtype=np.float32)
            link_maps[j.child].newcoords(rpy,
                                         xyz)
            # TODO fix automatically update default_coords
            link_maps[j.child].joint.default_coords = Coordinates(
                pos=link_maps[j.child].pos,
                rot=link_maps[j.child].rotation)

        # TODO duplicate of __init__
        self.link_list = links
        self.joint_list = joint_list

        self.joint_names = []
        for joint in self.joint_list:
            self.joint_names.append(joint.name)

        for link in self.link_list:
            self.__dict__[link.name] = link
        for joint in joint_list:
            self.__dict__[joint.name] = joint
        self.root_link = self.__dict__[root_link.name]
        self.assoc(self.root_link)

    def move_end_pos(self, pos, wrt='local', *args, **kwargs):
        pos = np.array(pos, dtype=np.float64)
        self.inverse_kinematics(
            self.end_coords.copy_worldcoords().translate(pos, wrt),
            move_target=self.end_coords,
            *args, **kwargs)
        return self.angle_vector()

    def move_end_rot(self, angle, axis, wrt='local', *args, **kwargs):
        rotation_axis = kwargs.pop('rotation_axis', True)
        self.inverse_kinematics(
            self.end_coords.copy_worldcoords().rotate(angle, axis, wrt),
            move_target=self.end_coords,
            rotation_axis=rotation_axis,
            *args, **kwargs)
        return self.angle_vector()

    def fix_leg_to_coords(self, fix_coords, leg='both', mid=0.5):
        """
        Fix robot's legs to a coords
        In the Following codes, leged robot is assumed.

        Parameters
        ----------
        fix_coords : Coordinates
            target coordinate
        leg : string
            ['both', 'rleg', 'rleg', 'left', 'right']
        mid : float
            ratio of legs coord.
        """
        if not any(self.legs):
            return None
        if leg == 'left' or leg == 'lleg':
            support_coords = self.lleg.end_coords.copy_worldcoords()
        elif leg == 'right' or leg == 'rleg':
            support_coords = self.rleg.end_coords.copy_worldcoords()
        else:
            support_coords = midcoords(
                mid,
                self.lleg.end_coords.copy_worldcoords(),
                self.rleg.end_coords.copy_worldcoords())
        tmp_coords = fix_coords.copy_worldcoords()
        move_coords = support_coords.transformation(self)
        tmp_coords.transform(move_coords, 'local')
        self.newcoords(tmp_coords)
        self.worldcoords()
        return tmp_coords

    @property
    def rarm(self):
        raise NotImplementedError

    @property
    def larm(self):
        raise NotImplementedError

    @property
    def rleg(self):
        raise NotImplementedError

    @property
    def lleg(self):
        raise NotImplementedError

    @property
    def joint_min_angles(self):
        return np.array([joint.min_angle for joint in self.joint_list],
                        dtype=np.float64)

    @property
    def joint_max_angles(self):
        return np.array([joint.max_angle for joint in self.joint_list],
                        dtype=np.float64)

    def inverse_kinematics(
            self,
            target_coords,
            move_target,
            link_list=None,
            **kwargs):
        """

        solve inverse kinematics, move move-target to target-coords
        look-at-target suppots t, nil, float-vector, coords, list of float-vector, list of coords
        link-list is set by default based on move-target -> root link link-list

        """
        if link_list is None:
            if not isinstance(move_target, list):
                link_list = self.link_lists(move_target.parent)
            else:
                link_list = list(
                    map(lambda mt: self.link_lists(mt.parent),
                        move_target))

        target_coords = listify(target_coords)
        return super(RobotModel, self).inverse_kinematics(
            target_coords, move_target=move_target,
            link_list=link_list,
            **kwargs)
        # (send self :look-at-target look-at-target :target-coords target-coords)

    def inverse_kinematics_loop(self,
                                dif_pos,
                                dif_rot,
                                move_target,
                                link_list=None,
                                target_coords=None,
                                **kwargs):
        """
        move move_target using dif_pos and dif_rot

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """
        target_coords = listify(target_coords)
        if link_list is None:
            if not isinstance(move_target, list):
                link_list = self.link_lists(move_target.parent)
            else:
                link_list = list(
                    map(lambda mt: self.link_lists(mt.parent),
                        move_target))

        return super(RobotModel, self).inverse_kinematics_loop(
            dif_pos, dif_rot,
            link_list=link_list, move_target=move_target,
            target_coords=target_coords, **kwargs)

    def inverse_kinematics_loop_for_look_at(
            self, move_target, look_at,
            link_list,
            rotation_axis='z',
            translation_axis=False,
            inverse_kinematics_hook=[],
            rthre=0.001,
            **kwargs):
        """

        Parameters
        ----------
        look_at : list or np.ndarray or Coordinates

        Returns
        -------

        """
        if isinstance(look_at, Coordinates):
            look_at_pos = look_at.worldpos()
        else:
            look_at_pos = look_at

        count = 0
        stop = kwargs.pop('stop', 100)
        p_dif_rot = None
        dif_rot = None
        joint_dimension = self.calc_target_joint_dimension(link_list)
        target_coords = orient_coords_to_axis(
            make_coords(pos=look_at_pos),
            look_at_pos - move_target.worldpos())

        return self.inverse_kinematics(target_coords,
                                       move_target=move_target,
                                       link_list=link_list,
                                       translation_axis=translation_axis,
                                       rotation_axis=rotation_axis)
        # target_coordss = []
        # while count < stop and \
        #       (p_dif_rot is None or
        #        LA.norm(p_dif_rot - dif_rot) > 1e-3):
        #     count += 1
        #     # orient target_coords to look at direction
        #     target_coords = orient_coords_to_axis(
        #         make_coords(pos=look_at_pos),
        #         look_at_pos - move_target.worldpos())
        #     target_coordss.append(target_coords.copy_worldcoords())

        #     J = self.calc_jacobian_from_link_list(move_target=move_target,
        #                                           link_list=link_list,
        #                                           translation_axis=translation_axis,
        #                                           rotation_axis=rotation_axis)
        #     print(J)

        #     p_dif_rot = dif_rot
        #     dif_pos = move_target.difference_position(
        #         target_coords, translation_axis=translation_axis)
        #     dif_rot = move_target.difference_rotation(
        #         target_coords, rotation_axis=rotation_axis)
        #     # head_weight = np.ones(joint_dimension, 'f')
        #     # head_nspace = np.zeros(joint_dimension, 'f')

        #     # target_orient = target_coords.worldrot()[:, 2]
        #     # move_orient = target_coords.worldrot()[:, 2]
        #     # for joint_id, j in enumerate(map(lambda l: l.joint, link_list)):
        #     #     axis_orient = j.parent_link.rotate_vector(_wrap_axis(j.axis))
        #     #     # direciton of non continuous area
        #     #     axis_orient2 = j.parent_link.rotate_vector([-1, 0, 0])
        #     #     if (triple_product(axis_orient, move_orient, target_orient) < 0.0 and
        #     #         triple_product(axis_orient, move_orient, axis_orient2) < 0.0 and
        #     #         triple_product(axis_orient, axis_orient2, target_orient < 0.0)) or \
        #     #         (triple_product(axis_orient, move_orient, target_orient) > 0.0 and
        #     #          triple_product(axis_orient, move_orient, axis_orient2) > 0.0 and
        #     #          triple_product(axis_orient, axis_orient2, target_orient > 0.0)):
        #     #         if triple_product(axis_orient, move_orient, target_orient) > 0.0:
        #     #             head_nspace[joint_id] = -1
        #     #         else:
        #     #             head_nspace[joint_id] = 1
        #     #         head_weight = 0.01 * np.ones(joint_dimension, 'f')
        #     #         head_weight[joint_id] = 0.0
        #     #         break
        #     print(dif_pos, dif_rot)
        #     self.inverse_kinematics_loop(
        #         dif_pos, dif_rot,
        #         move_target=move_target,
        #         rotation_axis=rotation_axis,
        #         translation_axis=translation_axis,
        #         link_list=link_list,
        #         loop=count,
        #         stop=stop,
        #         target_coords=target_coords)

        #     # vel_pos = self.calc_vel_from_pos(dif_pos, translation_axis)
        #     # vel_rot = self.calc_vel_from_rot(dif_rot, rotation_axis)
        #     # print(vel_pos, vel_rot)
        #     # union_vels = np.array([])
        #     # union_vel = np.concatenate([vel_pos, vel_rot])
        #     # union_vels = np.concatenate([union_vels, union_vel])
        #     # print('union-vel {}'.format(union_vels))

        #     # # success = True
        #     # # if rotation_axis is not None:
        #     # #     success = success and (LA.norm(dif_rot) < rthre)

        #     # if inverse_kinematics_hook:
        #     #     for hook in inverse_kinematics_hook:
        #     #         hook()

        #     # # if success:
        #     # #     return self.angle_vector()

        #     # union_link_list = self.calc_union_link_list(link_list)
        #     # self.move_joints_avoidance(
        #     #     union_vels,
        #     #     union_link_list=union_link_list,
        #     #     link_list=link_list,
        #     #     rotation_axis=rotation_axis,
        #     #     translation_axis=translation_axis,
        #     #     jacobi=J,
        #     #     # *args, **kwargs
        #     # )

        # if p_dif_rot is not None and \
        #    LA.norm(p_dif_rot - dif_rot) <= 1e-3:
        #     return target_coordss
        #     return self.angle_vector()

        # # solve failed
        # logger.warn('look at failed.')
        # logger.warn('    count : {}'.format(count))
        # if p_dif_rot is not None:
        #     logger.warn(' p_dif_rot : {}/({})'.format
        #                 (p_dif_rot, LA.norm(p_dif_rot)))
        #     logger.warn('   dif_rot : {}/({})'.format
        #                 (dif_rot, (LA.norm(dif_rot))))
        #     logger.warn('      diff : {} < {}'.format
        #                 (LA.norm(p_dif_rot - dif_rot), 1e-3))
        # return target_coordss


def calc_joint_angle_min_max_for_limit_calculation(j, kk, jamm=None):
    # TODO current support only 1-dof joint
    # fix unit system ;; [mm] -> [m], [deg] -> [rad]
    # 1-dof joint such as rotational_joint and linear_joint
    if jamm is None:
        jamm = np.zeros(3, 'f')
    jamm[0] = j.angle_to_speed(j.joint_angle())
    jamm[1] = j.angle_to_speed(j.max_angle)
    jamm[2] = j.angle_to_speed(j.min_angle)
    return jamm


def joint_angle_limit_weight(joint_list):
    """Calculate joint angle limit from joint list

    w_i = 1 + | dH/dt |      if d|dH/dt| >= 0
        = 1                  if d|dH/dt| <  0
    dH/dt = (t_max - t_min)^2 (2t - t_max - t_min) /
            (4 (t_max - t)^2 (t - t_min)^2)

    T. F. Chang and R.-V. Dubey: "A weighted least-norm solution based
    scheme for avoiding joint limits for redundant manipulators",
    in IEEE Trans. On Robotics and Automation,
    11((2):286-292, April 1995.

    """
    dims = calc_target_joint_dimension(joint_list)
    res = np.zeros(dims, 'f')
    k = 0
    kk = 0
    jamm = np.zeros(3, 'f')
    for i in range(dims):
        j = joint_list[k]
        calc_joint_angle_min_max_for_limit_calculation(j, kk, jamm)
        joint_angle, joint_max, joint_min = jamm
        e = np.deg2rad(1)
        k += 1

        # limitation
        if np.isclose(joint_angle, joint_max, e) and \
           np.isclose(joint_angle, joint_min, e):
            pass
        elif np.isclose(joint_angle, joint_max, e):
            joint_angle = joint_max - e
        elif np.isclose(joint_angle, joint_min, e):
            joint_angle = joint_min + e
        # calculate weight
        if np.isclose(joint_angle, joint_max, e) and \
           np.isclose(joint_angle, joint_min, e):
            res[i] = float('inf')
        else:
            if np.isinf(joint_min) or np.isinf(joint_max):
                r = 0.0
            else:
                r = abs(((joint_max - joint_min) ** 2) *
                        (2.0 * joint_angle - joint_max - joint_min) /
                        (4.0 * ((joint_max - joint_angle) ** 2) *
                         ((joint_angle - joint_min) ** 2)))
            res[i] = r
    return res


def joint_angle_limit_nspace(
        joint_list,
        n_joint_dimension=None):
    """Calculate nspace weight for avoiding joint angle limit

       dH/dq = (((t_max + t_min)/2 - t) / ((t_max - t_min)/2)) ^2

    """
    if n_joint_dimension is None:
        n_joint_dimension = calc_target_joint_dimension(joint_list)
    nspace = np.zeros(n_joint_dimension, 'f')
    for i in range(n_joint_dimension):
        joint = joint_list[i]
        joint_angle, joint_max, joint_min = \
            calc_joint_angle_min_max_for_limit_calculation(joint, i)

        # calculate weight
        if (joint_max - joint_min == 0.0) or \
           np.isinf(joint_max) or np.isinf(joint_min):
            r = 0.0
        else:
            r = ((joint_max + joint_min) - 2.0 * joint_angle) \
                / (joint_max - joint_min)
            r = np.sign(r) * (r ** 2)
        nspace[i] = r
    return nspace

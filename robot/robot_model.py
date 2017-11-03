from logging import getLogger
import itertools

import numpy as np
import numpy.linalg as LA
import scipy
import scipy.optimize

from robot.coordinates import CascadedCoords
from robot.coordinates import Coordinates
from robot.math import _wrap_axis
from robot.math import manipulability
from robot.math import midpoint
from robot.math import normalize_vector
from robot.math import sr_inverse
from robot.utils.urdf import URDF
from robot import worldcoords

logger = getLogger(__name__)


def calc_angle_speed_gain_scalar(joint, dav, i, periodic_time):
    dav_gain = abs(joint.max_joint_velocity / (dav[i] / periodic_time))
    return min(dav_gain, 1.0)


def calc_target_joint_dimension(joint_list):
    n = 0
    for j in joint_list:
        n += j.joint_dof
    return n


def calc_dif_with_axis(dif, axis,
                       tmp_v0=None, tmp_v1=None, tmp_v2=None):
    if axis in ['x', 'xx']:
        if tmp_v2:
            tmp_v2[0] = dif[1]
            tmp_v2[1] = dif[2]
            ret = tmp_v2
        else:
            ret = np.array([dif[1], dif[2]])
    elif axis in ['y', 'yy']:
        if tmp_v2:
            tmp_v2[0] = dif[0]
            tmp_v2[1] = dif[2]
            ret = tmp_v2
        else:
            ret = np.array([dif[0], dif[2]])
    elif axis in ['z', 'zz']:
        if tmp_v2:
            tmp_v2[0] = dif[0]
            tmp_v2[1] = dif[1]
            ret = tmp_v2
        else:
            ret = np.array([dif[0], dif[1]])
    elif axis in ['xy', 'yx']:
        if tmp_v1:
            tmp_v1[0] = dif[2]
            ret = tmp_v1
        else:
            ret = np.array([dif[2]])
    elif axis in ['yz', 'zy']:
        if tmp_v1:
            tmp_v1[0] = dif[0]
            ret = tmp_v1
        else:
            ret = np.array([dif[0]])
    elif axis in ['zx', 'xz']:
        if tmp_v1:
            tmp_v1[0] = dif[1]
            ret = tmp_v1
        else:
            ret = np.array([dif[1]])
    elif axis is None or axis is False:
        if tmp_v0:
            ret = tmp_v0
        else:
            ret = np.array([])
    elif axis in ['xm', 'ym', 'zm']:
        ret = dif
    else:
        ret = dif
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
            v += self.joint_angle
        if v > self.max_angle:
            logger.warning("{} :joint-angle({}) violate max-angle({})"
                           .format(self, v, self.max_angle))
            v = self.max_angle
        elif v < self.min_angle:
            logger.warning("{} :joint-angle({}) violate min-angle({})"
                           .format(self, v, self.min_angle))
            v = self.min_angle
        self._joint_angle = v
        # (send child-link :replace-coords default-coords)
        # (send child-link :rotate (deg2rad joint-angle) axis))
        self.child_link.rot = self.default_coords.rot.copy()
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


def calc_jacobian_rotational(fik, row, column, joint, paxis, child_link,
                             world_default_coords, child_reverse,
                             move_target, transform_coords, rotation_axis,
                             translation_axis, tmp_v0, tmp_v1, tmp_v2, tmp_v3,
                             tmp_v3a, tmp_v3b, tmp_m33):
    j_rot = calc_jacobian_default_rotate_vector(
        paxis, world_default_coords, child_reverse, transform_coords, tmp_v3, tmp_m33)
    p_diff = np.dot(transform_coords.worldrot().T,
                    (move_target.worldpos() - child_link.worldpos()))
    j_translation = np.cross(j_rot, p_diff)

    for i, j_trans in enumerate(j_translation):
        fik[i + row, column] = j_trans
    j_rotation = calc_dif_with_axis(
        j_rot, rotation_axis, tmp_v0, tmp_v1, tmp_v2)
    for i, j_rot in enumerate(j_rotation):
        fik[i + row + len(j_translation), column] = j_rot
    return fik


def calc_jacobian_default_rotate_vector(
        paxis, world_default_coords, child_reverse,
        transform_coords, tmp_v3, tmp_m33):
    if child_reverse:
        sign = -1.0
    else:
        sign = 1.0
    tmp_v3[:] = sign * \
        normalize_vector(world_default_coords.rotate_vector(paxis))
    tmp_m33[:] = transform_coords.worldrot().T
    tmp_v3[:] = np.dot(tmp_m33, tmp_v3)
    return tmp_v3


class Link(CascadedCoords):

    def __init__(self, centroid=None,
                 inertia_tensor=None,
                 *args, **kwargs):
        super(Link, self).__init__(*args, **kwargs)
        self.centroid = centroid
        self.joint = None
        if inertia_tensor is None:
            inertia_tensor = np.eye(3)

    def add_joint(self, j):
        self.joint = j

    def delete_joint(self, j):
        self.joint = None


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

    def inverse_kinematics_optimization(self,
                                        target_coords,
                                        move_target=None,
                                        link_list=None,
                                        regularization_parameter=None,
                                        init_angle_vector=None,
                                        translation_axis=True,
                                        rotation_axis=True,
                                        stop=100,
                                        options=None):
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

        joint_list = list(map(lambda l: l.joint, link_list))
        if init_angle_vector is None:
            init_angle_vector = np.array(list(map(lambda joint: joint.joint_angle(),
                                                  joint_list)))

        def objective_function(x):
            for j, theta in zip(joint_list, x):
                j.joint_angle(theta)
            cost = 0.0
            for mv, tc, trans_axis, rot_axis in zip(move_target,
                                                    target_coords,
                                                    translation_axis,
                                                    rotation_axis):
                dif_pos = mv.difference_position(tc, trans_axis) * 1000.0
                dif_rot = mv.difference_rotation(tc, rot_axis) * 1000.0
                cost += np.linalg.norm(dif_pos)
                cost += np.linalg.norm(dif_rot)
            return cost

        if options is None or not isinstance(options, dict):
            options = {}
        # Manage iterations maximum
        if stop is not None:
            options["maxiter"] = stop
        options['disp'] = True

        joint_angle_limits = list(
            map(lambda j: (j.min_angle, j.max_angle), joint_list))

        res = scipy.optimize.fmin_slsqp(
            func=objective_function,
            x0=init_angle_vector,
            bounds=joint_angle_limits)

        return res

    def calc_inverse_jacobian(self, jacobi,
                              manipulability_limit=0.1,
                              manipulability_gain=0.001,
                              weight=None):
        # m : manipulability
        m = manipulability(jacobi)
        if m < manipulability_limit:
            k = manipulability_gain * ((1.0 - m / manipulability_limit) ** 2)
        # calc weighted SR-inverse
        j_sharp = sr_inverse(jacobi, k, weight)
        return j_sharp

    def calc_joint_angle_speed_gain(self, union_link_list,
                                    dav,
                                    periodic_time):
        fik_len = self.calc_target_joint_dimension(union_link_list)
        av = np.zeros(fik_len)
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
        if jacobi is None or j_sharp is None:
            logger.warn(
                'jacobi(j) or j_sharp(J#) is required in calc_joint_angle_speed')
            return null_space
        fik_len = jacobi.shape[1]

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
                and fik_len == null_space):
            I = np.eye(fik_len)
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

        for axis in itertools.chain(translation_axis, rotation_axis):
            if axis in ['x', 'y', 'z', 'xx', 'yy', 'zz']:
                dim -= 1
            elif axis in ['xy', 'yx', 'yz', 'zy', 'zx', 'xz']:
                dim -= 2
            elif axis is False or axis is None:
                dim -= 3
        return dim

    def reset_joint_angle_limit_weight_old(self, union_link_list):
        tmp_joint_angle_limit_weight_old = self.find_joint_angle_limit_weight_old_from_union_link_list(
            union_link_list)
        if tmp_joint_angle_limit_weight_old is not None:
            tmp_joint_angle_limit_weight_old[1:][0] = None
        return tmp_joint_angle_limit_weight_old

    def calc_union_link_list(self, link_list):
        if not isinstance(link_list[0], list):
            return link_list
        if len(link_list) == 1:
            return link_list[0]
        raise NotImplementedError

    def calc_target_joint_dimension(self, link_list):
        return calc_target_joint_dimension(map(lambda l: l.joint,
                                               self.calc_union_link_list(link_list)))

    def calc_jacobian_from_link_list(self, link_list,
                                     move_target=None,
                                     transform_coords=None,
                                     rotation_axis=None,
                                     translation_axis=None,
                                     col_offset=0,
                                     dim=None,
                                     fik=None,
                                     fik_len=None,
                                     tmp_v0=np.zeros(0),
                                     tmp_v1=np.zeros(1),
                                     tmp_v2=np.zeros(2),
                                     tmp_v3=np.zeros(3),
                                     tmp_v3a=np.zeros(3),
                                     tmp_v3b=np.zeros(3),
                                     tmp_m33=np.zeros((3, 3)),
                                     *args, **kwargs):
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
        if fik_len is None:
            fik_len = self.calc_target_joint_dimension(link_list)
        if fik is None:
            fik = np.zeros((dim, fik_len), dtype=np.float32)

        union_link_list = self.calc_union_link_list(link_list)
        jdim = self.calc_target_joint_dimension(union_link_list)
        if not isinstance(link_list[0], list):
            link_lists = [link_list]
        if not isinstance(move_target, list):
            move_targets = [move_target]
        if not isinstance(transform_coords, list):
            transform_coords = [transform_coords]
        if not isinstance(rotation_axis, list):
            rotation_axes = [rotation_axis]
        if not isinstance(translation_axis, list):
            translation_axes = [translation_axis]

        col = col_offset
        i = 0
        while col < (col_offset + jdim):
            ul = union_link_list[i]
            row = 0

            for link_list, move_target, transform_coord, rotation_axis, translation_axis \
                    in zip(link_lists, move_targets, transform_coords, rotation_axes, translation_axes):
                if True:  # (member ul link-list :test #'equal)
                    length = len(link_list)
                    l = link_list.index(ul)
                    joint = ul.joint

                    def find_parent(pl, ll):
                        try:
                            is_find = ll.index[pl]
                        except:
                            is_find = False
                        if is_find or pl is False or pl is None:
                            return pl
                        else:
                            return find_parent(pl.parent, ll)

                    if not isinstance(joint.child_link, Link):
                        child_reverse = False
                    elif ((l + 1 < length) and
                          not joint.child_link != find_parent(link_list[l + 1].parent_link, link_list)) or \
                         ((l + 1 == length) and
                          (not joint.child_link == find_parent(move_target.parent, link_list))):
                        child_reverse = True
                    else:
                        child_reverse = False

                    paxis = _wrap_axis(joint.axis)
                    child_link = joint.child_link
                    parent_link = joint.parent_link
                    default_coords = joint.default_coords
                    world_default_coords = parent_link.transform(
                        default_coords)
                    joint.calc_jacobian(fik, row, col, joint, paxis,
                                        child_link, world_default_coords, child_reverse,
                                        move_target, transform_coord, rotation_axis,
                                        translation_axis,
                                        tmp_v0, tmp_v1, tmp_v2, tmp_v3, tmp_v3a, tmp_v3b, tmp_m33)
                row += self.calc_target_axis_dimension(rotation_axis,
                                                       translation_axis)
            col += joint.joint_dof
            i += 1
        return fik


class RobotModel(CascadedLink):

    def __init__(self):
        self.joint_links = []

    def reset_pose(self):
        raise NotImplementedError()

    def init_pose(self):
        return self.angle_vector(np.zeros_like(self.angle_vector()))

    def load_urdf(self, urdf_path):
        self.robot_urdf = URDF.from_xml_string(open(urdf_path).read())
        root_link = self.robot_urdf.link_map[self.robot_urdf.get_root()]

        links = []
        for link in self.robot_urdf.links:
            l = Link(name=link.name)
            links.append(l)

        link_maps = {l.name: l for l in links}

        joint_list = []
        for j in self.robot_urdf.joints:
            if j.type in ['fixed']:
                continue
            joint = RotationalJoint(
                axis=j.axis,
                name=j.name,
                parent_link=link_maps[j.parent],
                child_link=link_maps[j.child],
                min_angle=np.rad2deg(j.limit.lower),
                max_angle=np.rad2deg(j.limit.upper),
                max_joint_torque=j.limit.effort,
                max_joint_velocity=j.limit.velocity)
            joint_list.append(joint)

            link_maps[j.parent].add_child(link_maps[j.child])
            link_maps[j.child].parent_link = link_maps[j.parent]
            link_maps[j.child].add_joint(joint)

        for j in self.robot_urdf.joints:
            if j.type in ['fixed']:
                continue
            link_maps[j.child].newcoords(np.array(j.origin.rpy, dtype=np.float32),
                                         np.array(j.origin.xyz, dtype=np.float32))
            # TODO fix automatically update default_coords
            link_maps[j.child].joint.default_coords = Coordinates(pos=link_maps[j.child].pos,
                                                                  rot=link_maps[j.child].rot)
        self.link_list = links
        self.joint_list = joint_list
        for l in links:
            self.__dict__[l.name] = l
        for j in joint_list:
            self.__dict__[j.name] = j

        worldcoords.add_child(self.link_list[0])

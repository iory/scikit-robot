import io
import itertools
from logging import getLogger
import os
import sys
import tempfile
import warnings

import numpy as np
import numpy.linalg as LA
from ordered_set import OrderedSet
import six

from skrobot._lazy_imports import _lazy_trimesh
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates import convert_to_axis_vector
from skrobot.coordinates import Coordinates
from skrobot.coordinates import make_coords
from skrobot.coordinates import make_matrix
from skrobot.coordinates import midcoords
from skrobot.coordinates import midpoint
from skrobot.coordinates import normalize_vector
from skrobot.coordinates import orient_coords_to_axis
from skrobot.coordinates import rpy_angle
from skrobot.coordinates.math import jacobian_inverse
from skrobot.coordinates.math import matrix2quaternion
from skrobot.coordinates.math import matrix_log
from skrobot.coordinates.math import quaternion2matrix
from skrobot.coordinates.math import quaternion2rpy
from skrobot.coordinates.math import quaternion_inverse
from skrobot.coordinates.math import quaternion_multiply
from skrobot.coordinates.math import rodrigues
from skrobot.coordinates.math import rpy2quaternion
from skrobot.model.joint import calc_dif_with_axis
from skrobot.model.joint import calc_target_joint_dimension
from skrobot.model.joint import calc_target_joint_dimension_from_link_list
from skrobot.model.joint import FixedJoint
from skrobot.model.joint import joint_angle_limit_nspace
from skrobot.model.joint import joint_angle_limit_weight
from skrobot.model.joint import LinearJoint
from skrobot.model.joint import RotationalJoint
from skrobot.model.link import find_link_path
from skrobot.model.link import Link
from skrobot.utils import urdf
from skrobot.utils.listify import listify
from skrobot.utils.urdf import enable_mesh_cache
from skrobot.utils.urdf import URDF


try:
    from skrobot.utils.visualization import auto_ik_hook
except ImportError:
    auto_ik_hook = None

logger = getLogger(__name__)


class CascadedLink(CascadedCoords):

    def __init__(self,
                 link_list=None,
                 joint_list=None,
                 *args, **kwargs):
        super(CascadedLink, self).__init__(*args, **kwargs)
        self.link_list = link_list or []
        self.joint_list = joint_list or []
        self.bodies = []
        self.collision_avoidance_link_list = []
        self.end_coords_list = []
        self.joint_angle_limit_weight_maps = {}

        self._collision_manager = None
        self._relevance_predicate_table = None

    def _compute_relevance_predicate_table(self, joint_list=None,
                                           link_list=None):
        if joint_list is None:
            joint_list = self.joint_list
        if link_list is None:
            link_list = self.link_list
        relevance_predicate_table = {}
        for joint in joint_list:
            for link in link_list:
                key = (joint, link)
                relevance_predicate_table[key] = False

        def inner_recursion(joint, link):
            key = (joint, link)
            relevance_predicate_table[key] = True
            is_no_childlen = len(link._child_links) == 0
            if is_no_childlen:
                return
            for clink in link._child_links:
                inner_recursion(joint, clink)

        for joint in joint_list:
            link = joint.child_link
            inner_recursion(joint, link)
        return relevance_predicate_table

    def _is_relevant(self, joint, something):
        """check if `joint` affects `something`

        `something` must be at least CascadedCoords and must be
        connected to this CascadedLink. Otherwirse, this method
        raise AssertionError. If `something` is a descendant of `joint`,
        which means movement of `joint` affects `something`, thus
        this method returns `True`. Otherwise returns `False`.
        """

        assert isinstance(something, CascadedCoords), \
            "input must be at least a cascaded coords"

        def find_nearest_ancestor_link(something):
            # back-recursively find the closest ancestor link
            # if ancestor link is not found, return None
            if (something is None) or (isinstance(something, Link)):
                return something
            return find_nearest_ancestor_link(something.parent)
        link = find_nearest_ancestor_link(something)

        found_ancestor_link = (link is not None)
        assert found_ancestor_link, "input is not connected to the robot"

        key = (joint, link)
        if key in self._relevance_predicate_table:
            return self._relevance_predicate_table[key]
        return False

    def angle_vector(self, av=None, return_av=None):
        """Get or set joint angle vector.

        This is the core function for joint angle manipulation. If av is provided,
        it updates the angles of all joints. Joint limits are automatically enforced
        when setting angles.

        Parameters
        ----------
        av : numpy.ndarray or None
            Joint angle vector to set. If None, returns current joint angles.
            Values that violate joint limits are automatically clipped to valid range.
        return_av : numpy.ndarray or None
            Pre-allocated array to store the result. If None, creates a new array.
            This can improve performance when called repeatedly in optimization loops.

        Returns
        -------
        return_av : numpy.ndarray
            Current joint angle vector. Shape is (n_joints,) where n_joints is
            the total number of joint degrees of freedom.

        Notes
        -----
        This function handles:

        - Joint limit enforcement through min/max table lookup when available
        - Multi-DOF joints by processing multiple angles per joint
        - Efficient memory allocation when return_av is pre-allocated

        The function uses CascadedCoords lazy evaluation, so coordinate
        transformations are automatically updated when joint angles change.

        Examples
        --------
        >>> import numpy as np
        >>> from skrobot.models import PR2
        >>> robot = PR2()
        >>> # Get current joint angles
        >>> current_angles = robot.angle_vector()
        >>> # Set new joint angles (with automatic limit enforcement)
        >>> new_angles = np.zeros(len(robot.joint_list))
        >>> robot.angle_vector(new_angles)
        >>> # Efficient repeated calls with pre-allocated array
        >>> result_array = np.zeros(len(robot.joint_list))
        >>> for i in range(100):
        ...     angles = robot.angle_vector(return_av=result_array)
        """
        if return_av is None:
            return_av = np.zeros(len(self.joint_list), dtype=np.float32)

        for index, j in enumerate(self.joint_list):
            if av is not None:
                av = np.array(av)
                if not (j.joint_min_max_table is None
                        or j.joint_mix_max_target is None):
                    av = self.calc_joint_angle_from_min_max_table(index, j, av)
                else:
                    if j.joint_dof == 1:
                        j.joint_angle(av[index])
                    else:
                        j.joint_angle(av[index:index + j.joint_dof])
            for k in range(j.joint_dof):
                if j.joint_dof == 1:
                    return_av[index] = j.joint_angle()
                else:
                    return_av[index] = j.joint_angle[k]()
        return return_av

    def forward_kinematics(self, av=None, return_av=None):
        """Compute forward kinematics.

        This function sets joint angles and computes the forward kinematics
        to update the positions and orientations of all links in the robot.

        Parameters
        ----------
        av : numpy.ndarray or None
            Joint angle vector. If None, uses current joint angles.
        return_av : numpy.ndarray or None
            Pre-allocated array to store the result. If None, creates a new array.
            This can improve performance when called repeatedly.

        Returns
        -------
        av : numpy.ndarray
            Current angle vector after forward kinematics computation.

        Notes
        -----
        This is a wrapper function for `angle_vector` that provides a more
        intuitive name for computing forward kinematics. The function:

        1. Sets joint angles if av is provided
        2. Updates the kinematic chain (implicit through joint angle setting)
        3. Returns the current angle vector

        **Important:** After setting joint angles with `joint.joint_angle()`,
        you typically do not need to explicitly call `forward_kinematics()`
        because scikit-robot uses the CascadedCoords class which implements lazy
        evaluation. The coordinate transformations are automatically computed
        when needed (e.g., when accessing `worldpos()` or `worldrot()`).
        Explicit calls to `forward_kinematics()` are only necessary when you
        need to ensure all transformations are computed immediately.

        Examples
        --------
        >>> import numpy as np
        >>> from skrobot.models import PR2
        >>> robot = PR2()
        >>> # Set specific joint angles and compute forward kinematics
        >>> angles = np.zeros(len(robot.joint_list))
        >>> current_av = robot.forward_kinematics(angles)
        >>> # Get current configuration after forward kinematics
        >>> pos = robot.rarm.end_coords.worldpos()
        """
        return self.angle_vector(av, return_av)

    def calc_joint_angle_from_min_max_table(self, index, j, av):
        # currently only 1dof joint is supported
        if j.joint_dof == 1 and \
           j.joint_min_max_target.joint_dof == 1:
            # find index of joint-min-max-target
            ii = 0
            jj = self.joint_list[ii]

            while not jj == j.joint_min_max_target:
                ii += 1
                jj = self.joint_list[ii]
            tmp_joint_angle = av[index]
            tmp_target_joint_angle = av[ii]
            tmp_joint_min_angle = j.joint_min_max_table_min_angle(
                tmp_target_joint_angle)
            tmp_joint_max_angle = j.joint_min_max_table_max_angle(
                tmp_target_joint_angle)
            tmp_target_joint_min_angle = j.joint_min_max_table_min_angle(
                tmp_joint_angle)
            tmp_target_joint_max_angle = j.joint_min_max_table_max_angle(
                tmp_joint_angle)

            if tmp_joint_min_angle <= tmp_joint_angle and \
               tmp_joint_min_angle <= tmp_joint_max_angle:
                j.joint_angle = tmp_joint_angle
                jj.joint_angle = tmp_target_joint_angle
            else:
                i = 0.0
                while i > 1.0:
                    tmp_joint_min_angle = j.joint_min_max_table_min_angle(
                        tmp_target_joint_angle)
                    tmp_joint_max_angle = j.joint_min_max_table_max_angle(
                        tmp_target_joint_angle)
                    tmp_target_joint_min_angle = \
                        j.joint_min_max_table_min_angle(tmp_joint_angle)
                    tmp_target_joint_max_angle = \
                        j.joint_min_max_table_max_angle(tmp_joint_angle)
                    if tmp_joint_angle < tmp_joint_min_angle:
                        tmp_joint_angle += (tmp_joint_min_angle
                                            - tmp_joint_angle) * i
                    if tmp_joint_angle > tmp_joint_max_angle:
                        tmp_joint_angle += (tmp_joint_max_angle
                                            - tmp_joint_angle) * i
                    if tmp_target_joint_angle < tmp_target_joint_min_angle:
                        tmp_target_joint_angle += \
                            (tmp_target_joint_min_angle
                             - tmp_target_joint_angle) * i
                    if tmp_target_joint_angle > tmp_target_joint_max_angle:
                        tmp_target_joint_angle += \
                            (tmp_target_joint_max_angle
                             - tmp_target_joint_angle) * i
                j.joint_angle = tmp_joint_angle
                jj.joint_angle = tmp_target_joint_angle
                av[index] = tmp_joint_angle
                av[ii] = tmp_target_joint_angle
        return av

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
        link_index = 0
        while link_index < len(union_link_list):
            joint = union_link_list[link_index].joint
            if joint.joint_dof == 1:
                dtheta = dav[i]
            else:
                dtheta = dav[i:i + joint.joint_dof]
            union_link_list[link_index].joint.joint_angle(
                dtheta, relative=True)
            i += joint.joint_dof
            link_index += 1
        if move_joints_hook:
            for hook in move_joints_hook:
                hook()
        return True

    def collision_avoidance_link_pair_from_link_list(self,
                                                     link_list,
                                                     obstacles=None):
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
        additional_weight_list = additional_weight_list or []
        angle_speed_collision_blending = 0.0
        if n_joint_dimension is None:
            n_joint_dimension = self.calc_target_joint_dimension(
                union_link_list)
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
        # implementation issue:
        # when link and object are collide,
        # p = (nearestpoint_on_object_surface - center_of_link )
        # else
        # p = (nearestpoint_on_object_surface - neareset_point_on_link_surface)
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
           and (avoid_collision_joint_gain > 0.0
                or avoid_collision_null_gain > 0.0):
            angle_speed_collision_avoidance = self.collision_avoidance(
                avoid_collision_distance=avoid_collision_distance,
                avoid_collision_null_gain=avoid_collision_null_gain,
                avoid_collision_joint_gain=avoid_collision_joint_gain,
                weight=weight,
                collision_avoidance_link_pair=collision_avoidance_link_pair,
                *args,
                **kwargs)
            # (setq min-distance
            # (car (elt (send self :get :collision-distance) 0)))
            # (setq angle-speed-collision-blending
            #  (cond ((<= avoid-collision-joint-gain 0.0) 0.0)
            #   ((< min-distance (* 0.1 avoid-collision-distance))
            #    1.0)
            #   ((< min-distance avoid-collision-distance)
            #    (/ (- avoid-collision-distance min-distance)
            #     (* 0.9 avoid-collision-distance)))
            #   (t
            #    0.0)))

        tmp_nspace = self.calc_inverse_kinematics_nspace_from_link_list(
            link_list,
            union_link_list=union_link_list,
            avoid_nspace_gain=avoid_nspace_gain,
            weight=weight,
            n_joint_dimension=n_joint_dimension,
            null_space=null_space,
            additional_nspace_list=additional_nspace_list)

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
        """Calculate velocity from difference position

        Parameters
        ----------
        dif_pos : np.ndarray
            [m] order
        translation_axis : str
            see calc_dif_with_axis

        Returns
        -------
        vel_p : np.ndarray
        """
        if LA.norm(dif_pos) > p_limit:
            dif_pos = p_limit * normalize_vector(dif_pos)
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

    def calc_nspace_from_joint_limit(self,
                                     avoid_nspace_gain,
                                     union_link_list,
                                     weight):
        """Calculate null-space according to joint limit."""
        if avoid_nspace_gain > 0.0:
            joint_list = [ul.joint for ul in union_link_list]
            nspace = avoid_nspace_gain * weight * \
                joint_angle_limit_nspace(joint_list)
        else:
            raise ValueError('avoid_nspace_gain should be greater than '
                             '0.0, given {}'.format(avoid_nspace_gain))
        return nspace

    def calc_inverse_kinematics_nspace_from_link_list(
            self,
            link_list,
            avoid_nspace_gain=0.01,
            union_link_list=None,
            n_joint_dimension=None,
            null_space=None,
            additional_nspace_list=None,
            weight=None):
        additional_nspace_list = additional_nspace_list or []
        if union_link_list is None:
            union_link_list = self.calc_union_link_list(link_list)
        if n_joint_dimension is None:
            n_joint_dimension = self.calc_target_joint_dimension(
                union_link_list)
        if weight is None:
            weight = np.ones(n_joint_dimension, dtype=np.float64)
        # calc null-space from joint-limit
        nspace = self.calc_nspace_from_joint_limit(
            avoid_nspace_gain, union_link_list, weight)

        # add null-space from arguments
        # TODO(support additional_nspace_list)

        if callable(null_space):
            null_space = null_space()
        if null_space is not None:
            nspace = null_space + nspace
        return nspace

    def find_joint_angle_limit_weight_from_union_link_list(
            self, union_link_list):
        names = tuple(set([link.name for link in union_link_list]))
        if names in self.joint_angle_limit_weight_maps:
            return self.joint_angle_limit_weight_maps[names]
        else:
            return (names, False)

    def reset_joint_angle_limit_weight(self, union_link_list):
        names, weights = \
            self.find_joint_angle_limit_weight_from_union_link_list(
                union_link_list)
        if weights is not False:
            self.joint_angle_limit_weight_maps[names] = (names, False)

    def calc_weight_from_joint_limit(
            self,
            avoid_weight_gain,
            link_list,
            union_link_list,
            weight,
            n_joint_dimension=None):
        """Calculate weight according to joint limit."""
        if n_joint_dimension is None:
            n_joint_dimension = self.calc_target_joint_dimension(
                union_link_list)

        link_names, previous_joint_angle_limit_weight = \
            self.find_joint_angle_limit_weight_from_union_link_list(
                union_link_list)
        if previous_joint_angle_limit_weight is False:
            previous_joint_angle_limit_weight = np.inf * \
                np.ones(n_joint_dimension, 'f')
            self.joint_angle_limit_weight_maps[link_names] = (
                link_names, previous_joint_angle_limit_weight)

        new_weight = np.zeros_like(weight, 'f')
        joint_list = [l.joint for l in union_link_list
                      if l.joint is not None]
        if avoid_weight_gain > 0.0:
            current_joint_angle_limit_weight = avoid_weight_gain * \
                joint_angle_limit_weight(joint_list)
            for i in range(n_joint_dimension):
                if (current_joint_angle_limit_weight[i]
                        - previous_joint_angle_limit_weight[i]) >= 0.0:
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
        """Calculate all weight from link list."""
        if not (isinstance(link_list[0], tuple)
                or isinstance(link_list[0], list)):
            link_list = [link_list]
        if union_link_list is None:
            union_link_list = self.calc_union_link_list(link_list)
        if n_joint_dimension is None:
            n_joint_dimension = self.calc_target_joint_dimension(
                union_link_list)
        if weight is None:
            weight = np.ones(n_joint_dimension, 'f')

        for link, additional_weight in additional_weight_list:
            if link in union_link_list:
                link_index = union_link_list.index(link)
                index = calc_target_joint_dimension_from_link_list(
                    union_link_list[0:link_index])
                if callable(additional_weight):
                    w = additional_weight()
                else:
                    w = additional_weight
                if link.joint.joint_dof > 1:
                    for joint_dof_index in range(link.joint.joint_dof):
                        weight[index + joint_dof_index] *= w[joint_dof_index]
                else:
                    weight[index] *= w

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
            inverse_kinematics_hook=None,
            **kwargs):
        """inverse-kinematics-loop is one loop calculation.

        In this method, joint position difference satisfying workspace
        difference (dif_pos, dif_rot) are calculated and joint angles
        are updated.
        """
        inverse_kinematics_hook = inverse_kinematics_hook or []

        # Auto-inject visualization hooks from context
        if auto_ik_hook is not None:
            context_hooks = auto_ik_hook()
            if context_hooks:
                inverse_kinematics_hook = list(inverse_kinematics_hook) + context_hooks
        if move_target is None:
            raise NotImplementedError
        move_target = listify(move_target)

        n = len(move_target)
        if rotation_axis is None:
            rotation_axis = listify(True, n)
        if translation_axis is None:
            translation_axis = listify(True, n)
        if thre is None:
            thre = listify(0.001, n)
        if rthre is None:
            rthre = listify(np.deg2rad(1), n)
        if min_loop is None:
            min_loop = stop // 10

        # (if target-centroid-pos (send self :update-mass-properties))
        # ;; dif-pos, dif-rot, move-target,
        # rotation-axis, translation-axis, link-list
        # ;; -> both list and atom OK.
        union_vel = []
        union_vels = []
        vec_count = 0
        success = True
        p_limit = kwargs.pop('p_limit', None)
        r_limit = kwargs.pop('r_limit', None)
        if union_link_list is None:
            union_link_list = self.calc_union_link_list(link_list)
            # reset joint angle limit weight
            if len(union_link_list) != len(link_list):
                self.reset_joint_angle_limit_weight(union_link_list)
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
        if not (len(translation_axis)
                == len(rotation_axis)
                == len(move_target)
                == len(link_list)
                == len(dif_pos)
                == len(dif_rot)):
            logger.error(
                'list length differ : translation-axis %s rotation-axis %s '
                'move-target %s link-list %s dif-pos %s dif-rot %s',
                len(translation_axis), len(rotation_axis), len(move_target),
                len(link_list), len(dif_pos), len(dif_rot))
            return 'ik-continuous'

        for i in range(len(rotation_axis)):
            union_vels.append(np.zeros(self.calc_target_axis_dimension(
                rotation_axis[i], translation_axis[i]), 'f'))

        tmp_additional_jacobi = map(
            lambda aj: aj(link_list) if callable(aj) else aj,
            additional_jacobi)
        if cog_null_space is None and target_centroid_pos is not None:
            additional_jacobi_dimension = self.calc_target_axis_dimension(
                False, cog_translation_axis)
        else:
            additional_jacobi_dimension = 0
        additional_jacobi_dimension += sum(map(lambda aj: (
            aj(link_list) if callable(aj) else aj).shape[0],
            tmp_additional_jacobi))

        union_vel = np.zeros(self.calc_target_axis_dimension(
            rotation_axis, translation_axis)
            + additional_jacobi_dimension, 'f')

        # (if (memq :tmp-dims ik-args)
        #     (setq tmp-dims (cadr (memq :tmp-dims ik-args)))
        #   (progn
        #     (dotimes (i (length rotation-axis))
        #       (push (instantiate float-vector
        # (send self :calc-target-axis-dimension
        # (elt rotation-axis i) (elt translation-axis i))) tmp-dims))
        #     (setq tmp-dims (nreverse tmp-dims))))
        # (if (memq :tmp-dim ik-args)
        #     (setq tmp-dim (cadr (memq :tmp-dim ik-args)))
        #   (setq tmp-dim (instantiate float-vector
        # (send self
        # :calc-target-axis-dimension rotation-axis translation-axis))))

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
                additional_jacobi_dimension=additional_jacobi_dimension,
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
            success &= additional_check()

        # calculation of move-coords velocities from vel-p and vel-r
        for i in range(len(move_target)):
            tmp_union_vel = union_vels[i]
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
                tmp_union_vel[j] = dif_pos_ratio * vel_p[j]
            for j in range(len(vel_r)):
                tmp_union_vel[j + len(vel_p)] = dif_rot_ratio * vel_r[j]
            # (when (send self :get :ik-target-error)
            #   (push (list vel-p vel-r) (cdr (assoc (read-from-string (format

        vec_count = 0
        for i in range(len(union_vels)):
            for j in range(len(union_vels[i])):
                union_vel[j + vec_count] = union_vels[i][j]
            vec_count += len(union_vels[i])

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
        # (send self :calc-vel-for-cog tmp-cog-gain cog-translation-axis
        #                                                target-centroid-pos
        #                          :centroid-offset-func centroid-offset-func
        #                          :update-mass-properties nil))
        #                   additional-vel))
        #     (when (send self :get :ik-target-error)
        #       (push (car additional-vel) (cdr (assoc :centroid (send self :ge
        #     ))

        # append additional-jacobi and additional-vel
        if additional_jacobi is not None:
            additional_velocity_list = list(
                map(lambda x:
                    x(link_list) if callable(x) else x,
                    additional_vel))
            row0 = len(union_vel) - sum(map(len, additional_velocity_list))
            for i_add_jacobi in range(len(additional_jacobi)):
                add_jacobi = additional_jacobi[i_add_jacobi]
                if callable(add_jacobi):
                    add_jacobi = add_jacobi(link_list)
                add_vel = additional_velocity_list[i_add_jacobi]
                for i_row in range(add_jacobi.shape[0]):
                    # set additional-jacobi
                    for i_col in range(add_jacobi.shape[1]):
                        jacobi[row0 + i_row][i_col] = add_jacobi[i_row][i_col]
                    union_vel[row0 + i_row] = add_vel[i_row]
                row0 += len(add_vel)

        # check loop end
        if success:
            return 'ik-succeed'

        for hook in inverse_kinematics_hook:
            hook()
        self.collision_pair_list = None
        self.move_joints_avoidance(
            union_vel,
            union_link_list=union_link_list,
            link_list=link_list,
            rotation_axis=rotation_axis,
            translation_axis=translation_axis,
            jacobi=jacobi,
            **kwargs)
        return 'ik-continuous'

    def inverse_kinematics_args(
            self,
            union_link_list=None,
            rotation_axis=None,
            translation_axis=None,
            additional_jacobi_dimension=0,
            **kwargs):
        if union_link_list is None:
            union_link_list = []
        if translation_axis is None:
            translation_axis = []
        if rotation_axis is None:
            rotation_axis = []
        c = self.calc_target_joint_dimension(
            union_link_list)
        # add dimensions of additional-jacobi
        r = self.calc_target_axis_dimension(
            rotation_axis, translation_axis) + additional_jacobi_dimension

        jacobian = make_matrix(r, c)
        ret = make_matrix(c, r)

        union_vels = []
        for ta, ra in zip(translation_axis, rotation_axis):
            union_vels.append(np.zeros
                              (self.calc_target_axis_dimension(ra, ta),
                               'f'))
        return dict(dim=r,
                    jacobian=jacobian,
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
            additional_jacobi=None,
            additional_vel=None,
            **kwargs):
        additional_jacobi = additional_jacobi or []
        additional_vel = additional_vel or []
        target_coords = listify(target_coords)
        if callable(union_link_list):
            union_link_list = union_link_list(link_list)
        else:
            union_link_list = self.calc_union_link_list(link_list)

        if thre is None:
            if isinstance(move_target, list):
                thre = [0.001] * len(move_target)
            else:
                thre = [0.001]
        if rthre is None:
            if isinstance(move_target, list):
                rthre = [np.deg2rad(1)] * len(move_target)
            else:
                rthre = [np.deg2rad(1)]

        # store current angle vector
        joint_list = list(
            set([l.joint for l in union_link_list] + self.joint_list))
        if None in joint_list:
            logger.error('All links in link_list must have a parent joint')
            return True
        av0 = [j.joint_angle() for j in joint_list]
        c0 = None
        if self.parent is None:
            c0 = self.copy_worldcoords()
        success = True
        # (old-analysis-level (send-all union-link-list :analysis-level))
        # (send-all union-link-list :analysis-level :coords)

        # argument check
        if link_list is None or move_target is None:
            logger.error('both :link-list and :move-target required')
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

        if not (len(translation_axis)
                == len(rotation_axis)
                == len(move_target)
                == len(link_list)
                == len(target_coords)):
            logger.error('list length differ : translation_axis %s'
                         ', rotation_axis %s, move_target %s '
                         'link_list %s, target_coords %s',
                         len(translation_axis), len(rotation_axis),
                         len(move_target), len(link_list), len(target_coords))
            return False

        if len(additional_jacobi) != len(additional_vel):
            logger.error('list length differ : additional_jacobi %s, '
                         'additional_vel %s',
                         len(additional_jacobi), len(additional_vel))
            return False

        tmp_additional_jacobi = map(
            lambda aj: aj(link_list) if callable(aj) else aj,
            additional_jacobi)
        if cog_null_space is None and target_centroid_pos is not None:
            additional_jacobi_dimension = self.calc_target_axis_dimension(
                False, cog_translation_axis)
        else:
            additional_jacobi_dimension = 0
        additional_jacobi_dimension += sum(map(lambda aj: (
            aj(link_list) if callable(aj) else aj).shape[0],
            tmp_additional_jacobi))
        ik_args = self.inverse_kinematics_args(
            union_link_list=union_link_list,
            translation_axis=translation_axis,
            rotation_axis=rotation_axis,
            # evaluate additional-jacobi function and
            # calculate row dimension of additional_jacobi
            additional_jacobi_dimension=additional_jacobi_dimension,
            **kwargs)
        # self.reset_joint_angle_limit_weight_old(union_link_list) ;; reset
        # weight

        # inverse_kinematics loop
        loop = 0
        while loop < stop:
            loop += 1
            target_coords = list(map(
                lambda x: x() if callable(x) else x,
                target_coords))
            dif_pos = list(map(lambda mv, tc, trans_axis:
                               mv.difference_position(
                                   tc, translation_axis=trans_axis),
                               move_target, target_coords, translation_axis))
            dif_rot = list(map(lambda mv, tc, rot_axis:
                               mv.difference_rotation(
                                   tc, rotation_axis=rot_axis),
                               move_target, target_coords, rotation_axis))
            if loop == 1 and self.ik_convergence_check(
                    dif_pos, dif_rot, rotation_axis, translation_axis,
                    thre, rthre, centroid_thre, target_centroid_pos,
                    centroid_offset_func, cog_translation_axis,
            ):
                success = 'ik-succeed'
                break

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
                **ik_args)
            if success == 'ik-succeed':
                break

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
        success = self.ik_convergence_check(
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
            update_mass_properties=False)

        # reset joint angle limit weight
        self.reset_joint_angle_limit_weight(union_link_list)

        # TODO(add collision check)
        if success or not revert_if_fail:
            return self.angle_vector()

        # reset angle vector
        for joint, angle in zip(joint_list, av0):
            joint.joint_angle(angle)
        if c0 is not None:
            self.newcoords(c0)
        return False

    def ik_convergence_check(
            self,
            dif_pos,
            dif_rot,
            rotation_axis,
            translation_axis,
            thre,
            rthre,
            centroid_thre=None,
            target_centroid_pos=None,
            centroid_offset_func=None,
            cog_translation_axis=None,
            update_mass_properties=True):
        """check ik convergence.

        Parameters
        ----------
        dif_pos : list of np.ndarray
        dif_rot : list of np.ndarray

        translation_axis : list of axis
        rotation_axis : list of axis
            see convert_to_axis_vector
        """
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
                                        inverse_kinematics_hook=None,
                                        thre=0.001,
                                        rthre=np.deg2rad(1.0),
                                        *args, **kwargs):

        inverse_kinematics_hook = inverse_kinematics_hook or []

        # Auto-inject visualization hooks from context
        if auto_ik_hook is not None:
            context_hooks = auto_ik_hook()
            if context_hooks:
                inverse_kinematics_hook = list(inverse_kinematics_hook) + context_hooks

        if not isinstance(target_coords, list):
            target_coords = [target_coords]
        if not isinstance(move_target, list):
            move_target = [move_target]
        if not isinstance(rotation_axis, list):
            rotation_axis = [rotation_axis]
        if not isinstance(translation_axis, list):
            translation_axis = [translation_axis]
        if not (len(move_target)
                == len(rotation_axis)
                == len(translation_axis)
                == len(target_coords)):
            logger.error(
                'list length differ : target_coords %s translation_axis %s '
                'rotation_axis %s move_target %s',
                len(target_coords), len(translation_axis),
                len(rotation_axis), len(move_target))

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
        return jacobian_inverse(jacobi, manipulability_limit,
                                manipulability_gain, weight)

    def calc_joint_angle_speed_gain(self, union_link_list,
                                    dav,
                                    periodic_time):
        n_joint_dimension = self.calc_target_joint_dimension(union_link_list)
        av = np.zeros(n_joint_dimension)
        i = 0
        link_index = 0
        while link_index < len(union_link_list):
            j = union_link_list[link_index].joint
            for k in range(j.joint_dof):
                av[i + k] = j.calc_angle_speed_gain(dav, i, periodic_time)
            i += j.joint_dof
            link_index += 1
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
            logger.warning(
                'jacobi(j) or j_sharp(J#) is required '
                'in calc_joint_angle_speed')
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
        t = type(null_space)
        if (t is list or t is np.ndarray) and \
           n_joint_dimension == len(null_space):
            I_matrix = np.eye(n_joint_dimension)
            j_sharp_x += np.matmul(I_matrix - np.matmul(j_sharp, jacobi),
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

        J = self.calc_jacobian_from_link_list(
            move_target=move_target,
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
            # TODO(duplicate of jacobian based ik)
            dif_pos = mv.difference_position(tc,
                                             translation_axis=trans_axis)
            dif_rot = mv.difference_rotation(tc,
                                             rotation_axis=rot_axis)
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
        from skrobot.optimizer import solve_qp
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
        if pl and to not in self.link_list:
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
        """Find link list from to link to frm link."""
        ret1 = self.find_link_route(to, frm)
        if frm and not ret1[0] == frm:
            ret2 = self.find_link_route(frm, ret1[0])
            ret1 = ret2[::-1] + ret1
        return ret1

    def find_link_path(self, src_link, target_link):
        """Find paths of src_link to target_link

        Parameters
        ----------
        src_link : skrobot.model.link.Link
            source link.
        target_link : skrobot.model.link.Link
            target link.

        Returns
        -------
        ret : List[skrobot.model.link.Link]
            If the links are connected, return Link list.
            Otherwise, return an empty list.
        """
        paths, _ = find_link_path(
            src_link, target_link)
        return paths

    def calc_union_link_list(self, link_list):
        if not isinstance(link_list, list):
            raise TypeError('Input should be `list`, get type=={}'
                            .format(type(link_list)))
        if len(link_list) == 0:
            return link_list
        elif not isinstance(link_list[0], list):
            return link_list
        elif len(link_list) == 1:
            return link_list[0]
        else:
            return list(OrderedSet(list(itertools.chain(*link_list))))

    def calc_target_joint_dimension(self, link_list):
        return calc_target_joint_dimension(
            map(lambda l: l.joint, self.calc_union_link_list(link_list)))

    def calc_jacobian_from_link_list(self,
                                     move_target,
                                     link_list=None,
                                     transform_coords=None,
                                     rotation_axis=None,
                                     translation_axis=None,
                                     col_offset=0,
                                     dim=None,
                                     jacobian=None,
                                     additional_jacobi_dimension=0,
                                     n_joint_dimension=None,
                                     *args, **kwargs):
        assert self._relevance_predicate_table is not None, \
            "relevant table must be set beforehand"

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
                rotation_axis, translation_axis) + additional_jacobi_dimension
        if n_joint_dimension is None:
            n_joint_dimension = self.calc_target_joint_dimension(link_list)
        if jacobian is None:
            jacobian = np.zeros((dim, n_joint_dimension), dtype=np.float32)

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
        world_default_coords = Coordinates()
        while col < (col_offset + jdim):
            ul = union_link_list[i]
            row = 0

            for (link_list,
                 move_target,
                 transform_coord,
                 rotation_axis,
                 translation_axis) in zip(
                     link_lists,
                     move_targets,
                     transform_coords,
                     rotation_axes,
                     translation_axes):
                if ul in link_list:
                    joint = ul.joint

                    if self._is_relevant(joint, move_target):
                        if joint.joint_dof <= 1:
                            paxis = convert_to_axis_vector(joint.axis)
                        else:
                            paxis = joint.axis
                        child_link = joint.child_link
                        parent_link = joint.parent_link
                        default_coords = joint.default_coords
                        # set new coordinates to world_default_coords.
                        parent_link.worldcoords().\
                            transform(default_coords, out=world_default_coords)

                        jacobian = joint.calc_jacobian(
                            jacobian,
                            row,
                            col,
                            joint,
                            paxis,
                            child_link,
                            world_default_coords,
                            move_target,
                            transform_coord,
                            rotation_axis,
                            translation_axis)
                row += self.calc_target_axis_dimension(rotation_axis,
                                                       translation_axis)
            col += joint.joint_dof
            i += 1
        return jacobian

    @property
    def interlocking_joint_pairs(self):
        """Interlocking joint pairs.

        pairs are [(joint0, joint1), ...] If users want to use
        interlocking joints, please overwrite this method.
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
                   not ((pair[0] not in joint_list)
                        and (pair[1] not in joint_list)),
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
        """Calculate 0 velocity for keeping interlocking joint.

        at the same joint angle.
        """
        if interlocking_joint_pairs is None:
            interlocking_joint_pairs = self.interlocking_joint_pairs
        union_link_list = self.calc_union_link_list(link_list)
        joint_list = list(filter(lambda j: j is not None,
                                 [l.joint for l in union_link_list]))
        pairs = list(
            filter(lambda pair:
                   not ((pair[0] not in joint_list)
                        and (pair[1] not in joint_list)),
                   interlocking_joint_pairs))
        vel = np.zeros(len(pairs), 'f')
        return vel

    def self_collision_check(self):
        """Return collision link pair

        Note: This function uses trimesh.collision.CollisionManager internally.
        We need to install python-fcl to use this function.
        If you want to use this function, please install python-fcl by
        `pip install python-fcl`.

        Returns
        -------
        is_collision : bool
            True if a collision occurred between any pair of objects
            and False otherwise
        names : set of 2-tuple
            The set of pairwise collisions. Each tuple
            contains two names in alphabetical order indicating
            that the two corresponding objects are in collision.
        """
        if self._collision_manager is None:
            trimesh = _lazy_trimesh()
            self._collision_manager = trimesh.collision.CollisionManager()
            for link in self.link_list:
                transform = link.worldcoords().T()
                mesh = link.collision_mesh
                if mesh is not None:
                    self._collision_manager.add_object(
                        link.name, mesh, transform=transform)
        else:
            for link in self.link_list:
                mesh = link.collision_mesh
                if mesh is not None:
                    transform = link.worldcoords().T()
                    self._collision_manager.set_transform(
                        link.name, transform=transform)
        return self._collision_manager.in_collision_internal(return_names=True)


class RobotModel(CascadedLink):

    def __init__(self, link_list=None, joint_list=None,
                 root_link=None):
        link_list = link_list or []
        joint_list = joint_list or []
        super(RobotModel, self).__init__(link_list, joint_list)

        self.joint_names = []
        for joint in self.joint_list:
            self.joint_names.append(joint.name)
            joint.child_link.add_parent_link(joint.parent_link)
            joint.parent_link.add_child_link(joint.child_link)

        for link in self.link_list:
            self.__dict__[link.name] = link
        for joint in joint_list:
            self.__dict__[joint.name] = joint
        self.urdf_path = None

        self._relevance_predicate_table = \
            self._compute_relevance_predicate_table()

    def reset_pose(self):
        raise NotImplementedError()

    def reset_manip_pose(self):
        raise NotImplementedError()

    def init_pose(self):
        target_angles = np.zeros_like(self.joint_min_angles)
        target_angles = np.clip(target_angles,
                                self.joint_min_angles,
                                self.joint_max_angles)
        return self.angle_vector(target_angles)

    def _meshes_from_urdf_visuals(self, visuals):
        meshes = []
        for visual in visuals:
            meshes.extend(self._meshes_from_urdf_visual(visual))
        return meshes

    def _meshes_from_urdf_visual(self, visual):
        if not isinstance(visual, urdf.Visual):
            raise TypeError('visual must be urdf.Visual, but got: {}'
                            .format(type(visual)))

        trimesh = None
        meshes = []
        for mesh in visual.geometry.meshes:
            if trimesh is None:
                trimesh = _lazy_trimesh()
            mesh = mesh.copy()

            # rescale
            if visual.geometry.mesh is not None:
                if visual.geometry.mesh.scale is not None:
                    mesh.vertices = mesh.vertices * visual.geometry.mesh.scale

            # TextureVisuals is usually slow to render
            if not isinstance(mesh.visual, trimesh.visual.ColorVisuals):
                mesh.visual = mesh.visual.to_color()
                if mesh.visual.vertex_colors.ndim == 1:
                    mesh.visual.vertex_colors = \
                        mesh.visual.vertex_colors[None].repeat(
                            mesh.vertices.shape[0], axis=0
                        )

            # If color or texture is not specified in mesh file,
            # use information specified in URDF.
            if (
                (mesh.visual.face_colors
                 == trimesh.visual.DEFAULT_COLOR).all()
                and visual.material
            ):
                if visual.material.texture is not None:
                    warnings.warn(
                        'texture specified in URDF is not supported'
                    )
                elif visual.material.color is not None:
                    mesh.visual.face_colors = visual.material.color

            mesh.apply_transform(visual.origin)
            meshes.append(mesh)
        return meshes

    def load_urdf_from_robot_description(
            self, param_name='/robot_description',
            include_mimic_joints=True):
        """Load URDF from ROS parameter server and initialize the model.

        Waits until the specified ROS parameter is available, reads the URDF
        content, saves it to a temporary file, and then loads it using
        `load_urdf_file`. Supports both ROS1 and ROS2.

        Parameters
        ----------
        param_name : str, optional
            The name of the ROS parameter containing the URDF XML string.
            Defaults to '/robot_description'.
        include_mimic_joints : bool, optional
            If True, mimic joints are included in the `self.joint_list`.
            Passed directly to `load_urdf_file`.
        """
        ros_version = os.environ.get('ROS_VERSION', '1')

        if ros_version == '2':
            try:
                self._load_urdf_from_ros2(param_name, include_mimic_joints)
                return
            except ImportError:
                logger.warning("rclpy not available for ROS2, falling back to ROS1 method")

        try:
            self._load_urdf_from_ros1(param_name, include_mimic_joints)
        except ImportError:
            raise RuntimeError("Neither rclpy (ROS2) nor rospy (ROS1) is available. "
                               "Please ensure ROS is properly installed and sourced.")

    def _load_urdf_from_ros2(self, param_name, include_mimic_joints):
        """Load URDF from ROS2 parameter server with security and dynamic node discovery."""
        import rclpy
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()

        node = Node('urdf_loader_node_for_skrobot')

        try:
            urdf = self._discover_and_fetch_urdf_ros2(node, param_name)
            self._load_urdf_from_string_secure(urdf, include_mimic_joints)
        finally:
            # Always destroy the node to prevent resource leaks
            logger.debug("Destroying temporary ROS2 node")
            node.destroy_node()

    def _discover_and_fetch_urdf_ros2(self, node, param_name):
        """Discover nodes and fetch URDF parameter from ROS2."""

        # Get available services to find nodes with get_parameters service
        available_services = node.get_service_names_and_types()
        get_param_services = [svc[0] for svc in available_services if 'get_parameters' in svc[0]]
        logger.debug("Available get_parameters services: %s", get_param_services)

        # Extract node names from service names (dynamic discovery)
        candidate_nodes = []
        for service in get_param_services:
            if service.endswith('/get_parameters'):
                node_name = service[:-len('/get_parameters')]
                if node_name != f'/{node.get_name()}':  # Skip our own node
                    candidate_nodes.append(node_name)

        logger.info("Searching for URDF parameter in nodes: %s", candidate_nodes)

        # Clean parameter name (remove leading slash for ROS2)
        clean_param_name = param_name.lstrip('/')

        for target_node in candidate_nodes:
            try:
                urdf = self._fetch_parameter_from_node(node, target_node, clean_param_name)
                if urdf:
                    logger.info("Successfully got URDF parameter from node: %s", target_node)
                    return urdf
            except Exception as e:
                logger.debug("Failed to get parameter from %s: %s", target_node, e)
                continue

        raise RuntimeError("Could not find parameter '%s' in any ROS2 node. "
                           "Available nodes: %s" % (param_name, candidate_nodes))

    def _fetch_parameter_from_node(self, node, target_node, clean_param_name):
        """Fetch a specific parameter from a target node."""
        from rcl_interfaces.srv import GetParameters
        import rclpy

        service_name = f'{target_node}/get_parameters'
        logger.debug("Trying to get parameter from node: %s", target_node)

        # Try 1: Create client and wait for service
        try:
            client = node.create_client(GetParameters, service_name)
            if not client.wait_for_service(timeout_sec=2.0):
                logger.debug("Service not available: %s", service_name)
                return None
        except Exception as e:
            logger.debug("Failed to create client for %s: %s", service_name, e)
            return None

        # Try 2: Call service and get response
        try:
            request = GetParameters.Request()
            request.names = [clean_param_name]
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
        except Exception as e:
            logger.debug("Failed to call service %s: %s", service_name, e)
            return None

        # Try 3: Process response
        try:
            if future.result() is None:
                return None

            response = future.result()
            if not (response.values and len(response.values) > 0):
                return None

            param_value = response.values[0]
            logger.debug("Parameter type: %s, string value length: %s",
                         param_value.type,
                         len(param_value.string_value) if param_value.string_value else 0)

            # Parameter type 4 corresponds to PARAMETER_STRING in ROS2
            if (param_value.type == 4 and
                param_value.string_value):
                return param_value.string_value

        except Exception as e:
            logger.debug("Failed to process response from %s: %s", service_name, e)

        return None

    def _load_urdf_from_string_secure(self, urdf_string, include_mimic_joints):
        """Load URDF from string using secure temporary file handling."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".urdf") as f:
            f.write(urdf_string)
            tmp_path = f.name

        try:
            self.load_urdf_file(file_obj=tmp_path, include_mimic_joints=include_mimic_joints)
        finally:
            try:
                os.remove(tmp_path)
            except OSError as e:
                logger.warning("Failed to remove temporary file %s: %s", tmp_path, e)

    def _load_urdf_from_ros1(self, param_name, include_mimic_joints):
        """Load URDF from ROS1 parameter server with security and timeout handling."""
        import rospy

        # Initialize rospy if not already initialized
        if rospy.rostime._rostime_initialized is False:
            rospy.init_node('urdf_loader_node_for_skrobot', anonymous=True)
            logger.info("Initialized temporary ROS1 node")

        urdf = self._fetch_urdf_with_timeout_ros1(param_name)
        self._load_urdf_from_string_secure(urdf, include_mimic_joints)

    def _fetch_urdf_with_timeout_ros1(self, param_name, timeout_seconds=10):
        """Fetch URDF parameter from ROS1 with timeout."""
        import rospy

        rate = rospy.Rate(1)
        for attempt in range(timeout_seconds):
            if rospy.is_shutdown():
                raise RuntimeError("ROS was shut down")

            urdf = rospy.get_param(param_name, None)
            if urdf is not None:
                return urdf

            logger.warning("Waiting for ROS parameter '%s'... (attempt %d/%d)",
                           param_name, attempt + 1, timeout_seconds)
            rate.sleep()

        raise RuntimeError("Timeout: Could not find ROS parameter '%s' after %d seconds" %
                           (param_name, timeout_seconds))

    @staticmethod
    def from_robot_description(param_name='/robot_description',
                               include_mimic_joints=True):
        """Load URDF from ROS parameter server.

        Supports both ROS1 and ROS2 based on the ROS_VERSION environment variable.

        Parameters
        ----------
        param_name : str
            Parameter name in the parameter server.
        include_mimic_joints : bool, optional
            If True, mimic joints are included in the resulting
            `RobotModel`'s `joint_list`.

        Returns
        -------
        RobotModel
            Robot model loaded from URDF.
        """
        robot_model = RobotModel()
        robot_model.load_urdf_from_robot_description(
            param_name,
            include_mimic_joints=include_mimic_joints)
        return robot_model

    @staticmethod
    def from_urdf(urdf_input, include_mimic_joints=True):
        """Load URDF from a string or a file path.

        Automatically detects if the input is a URDF string or a file path.

        Parameters
        ----------
        urdf_string : str
            Either the URDF model description as a string, or the path to a
            URDF file.
        include_mimic_joints : bool, optional
            If True, mimic joints are included in the resulting
            `RobotModel`'s `joint_list`.

        Returns
        -------
        RobotModel
            Robot model loaded from the URDF.
        """
        robot_model = RobotModel()
        if os.path.isfile(urdf_input):
            try:
                with open(urdf_input, 'r') as f:
                    robot_model.load_urdf_file(
                        file_obj=f, include_mimic_joints=include_mimic_joints)
            except Exception as e:
                logger.error("Failed to load URDF from file: %s. Error: %s", urdf_input, e)
                logger.error("Attempting to load as URDF string instead.")
                robot_model.load_urdf(
                    urdf_input, include_mimic_joints=include_mimic_joints)
        else:
            robot_model.load_urdf(urdf_input,
                                  include_mimic_joints=include_mimic_joints)
        return robot_model

    def load_urdf(self, urdf, include_mimic_joints=True):
        is_python3 = sys.version_info.major > 2
        f = io.StringIO()
        if is_python3:
            f.write(str(urdf))
        else:
            f.write(urdf.decode('utf-8'))
        f.seek(0)
        f.name = "dummy"
        self.load_urdf_file(file_obj=f,
                            include_mimic_joints=include_mimic_joints)

    def load_urdf_file(self, file_obj, include_mimic_joints=True):
        """Load robot model from URDF file.

        This method parses a URDF file, creates the corresponding
        link and joint objects, and sets up the robot's kinematic tree
        structure. It also loads visual and collision meshes.

        Parameters
        ----------
        file_obj : str or file-like object
            Path to the URDF file or a file-like object containing URDF data.
        include_mimic_joints : bool, optional
            If True, mimic joints are included in the `self.joint_list`.
            If False, mimic joints are excluded from `self.joint_list`,
            although their mimic definitions are still processed and applied
            to the joints they mimic.
        """
        if isinstance(file_obj, six.string_types):
            self.urdf_path = file_obj
        else:
            self.urdf_path = getattr(file_obj, 'name', None)
        with enable_mesh_cache():
            self.urdf_robot_model = URDF.load(file_obj=file_obj)
        self.name = self.urdf_robot_model.name
        if self.urdf_robot_model.base_link is None:
            logger.error('URDF must have a base link defined.')
            root_link = None
        else:
            root_link = self.urdf_robot_model.base_link

        links = []
        for urdf_link in self.urdf_robot_model.links:
            link = Link(name=urdf_link.name)
            link.collision_mesh = urdf_link.collision_mesh
            link.visual_mesh = self._meshes_from_urdf_visuals(
                urdf_link.visuals)
            links.append(link)
        link_maps = {l.name: l for l in links}

        joint_list = []
        whole_joint_list = []
        mimic_joint_names = {j.name for j in self.urdf_robot_model.joints
                             if j.mimic is not None}

        for j in self.urdf_robot_model.joints:
            if j.parent not in link_maps or j.child not in link_maps:
                logger.warning(
                    'Joint %s has invalid parent or child link. '
                    'Skipping this joint.', j.name)
                continue
            if j.limit is None:
                j.limit = urdf.JointLimit(0, 0)
            if j.axis is None:
                j.axis = 'z'
            if j.joint_type == 'fixed':
                joint = FixedJoint(
                    name=j.name,
                    parent_link=link_maps[j.parent],
                    child_link=link_maps[j.child])
            elif j.joint_type == 'revolute':
                # For mimic joints, set a default velocity if not specified to avoid warnings
                velocity = j.limit.velocity
                if j.mimic is not None and velocity <= 0:
                    velocity = np.deg2rad(5)  # Default velocity for mimic joints
                joint = RotationalJoint(
                    axis=j.axis,
                    name=j.name,
                    parent_link=link_maps[j.parent],
                    child_link=link_maps[j.child],
                    min_angle=j.limit.lower,
                    max_angle=j.limit.upper,
                    max_joint_torque=j.limit.effort,
                    max_joint_velocity=velocity)
            elif j.joint_type == 'continuous':
                # For mimic joints, set a default velocity if not specified to avoid warnings
                velocity = j.limit.velocity
                if j.mimic is not None and velocity <= 0:
                    velocity = np.deg2rad(5)  # Default velocity for mimic joints
                joint = RotationalJoint(
                    axis=j.axis,
                    name=j.name,
                    parent_link=link_maps[j.parent],
                    child_link=link_maps[j.child],
                    min_angle=-np.inf,
                    max_angle=np.inf,
                    max_joint_torque=j.limit.effort,
                    max_joint_velocity=velocity)
            elif j.joint_type == 'prismatic':
                # http://wiki.ros.org/urdf/XML/joint
                # meters for prismatic joints
                # For mimic joints, set a default velocity if not specified to avoid warnings
                velocity = j.limit.velocity
                if j.mimic is not None and velocity <= 0:
                    velocity = np.pi / 4.0  # Default velocity for mimic joints
                joint = LinearJoint(
                    axis=j.axis,
                    name=j.name,
                    parent_link=link_maps[j.parent],
                    child_link=link_maps[j.child],
                    min_angle=j.limit.lower,
                    max_angle=j.limit.upper,
                    max_joint_torque=j.limit.effort,
                    max_joint_velocity=velocity)

            is_mimic = j.name in mimic_joint_names
            if j.joint_type != 'fixed':
                if include_mimic_joints or not is_mimic:
                    joint_list.append(joint)
            whole_joint_list.append(joint)

            # TODO(make clear the difference between assoc and add_child_link)
            link_maps[j.parent].assoc(link_maps[j.child])
            link_maps[j.child].add_joint(joint)
            link_maps[j.child].add_parent_link(link_maps[j.parent])
            link_maps[j.parent].add_child_link(link_maps[j.child])

        for j in self.urdf_robot_model.joints:
            if j.parent not in link_maps or j.child not in link_maps:
                continue
            if j.origin is None:
                rpy = np.zeros(3, dtype=np.float32)
                xyz = np.zeros(3, dtype=np.float32)
            else:
                rpy = rpy_angle(j.origin[:3, :3])[0]
                xyz = j.origin[:3, 3]
            link_maps[j.child].newcoords(rpy,
                                         xyz)
            # TODO(fix automatically update default_coords)
            link_maps[j.child].joint.default_coords = Coordinates(
                pos=link_maps[j.child].translation,
                rot=link_maps[j.child].rotation)

        # Load mass properties from URDF
        for urdf_link in self.urdf_robot_model.links:
            link = link_maps[urdf_link.name]
            if urdf_link.inertial is not None:
                # Mass in kg (URDF stores in kg)
                link.mass = urdf_link.inertial.mass

                # Center of mass in meters (URDF stores in meters)
                if urdf_link.inertial.origin is not None:
                    link.centroid = urdf_link.inertial.origin[:3, 3]
                else:
                    link.centroid = np.zeros(3)

                # Inertia tensor in kg*m^2 (URDF stores in kg*m^2)
                inertia = urdf_link.inertial.inertia
                if hasattr(inertia, 'ixx'):
                    # Individual components
                    link.inertia_tensor = np.array([
                        [inertia.ixx, inertia.ixy, inertia.ixz],
                        [inertia.ixy, inertia.iyy, inertia.iyz],
                        [inertia.ixz, inertia.iyz, inertia.izz]
                    ])
                elif isinstance(inertia, np.ndarray) and inertia.shape == (3, 3):
                    # Already a 3x3 matrix
                    link.inertia_tensor = inertia
                elif isinstance(inertia, np.ndarray) and inertia.shape == (6,):
                    # Vector format [ixx, iyy, izz, ixy, ixz, iyz]
                    link.inertia_tensor = np.array([
                        [inertia[0], inertia[3], inertia[4]],
                        [inertia[3], inertia[1], inertia[5]],
                        [inertia[4], inertia[5], inertia[2]]
                    ])
                else:
                    # Default fallback
                    link.inertia_tensor = np.eye(3) * 0.001
            else:
                # Set default minimal mass properties if not specified in URDF
                link.mass = 0.001  # 1 gram
                link.centroid = np.zeros(3)
                link.inertia_tensor = np.eye(3) * 1e-6

        # TODO(duplicate of __init__)
        self.link_list = links
        self.joint_list = joint_list

        self.joint_names = []
        for joint in self.joint_list:
            self.joint_names.append(joint.name)

        for link in self.link_list:
            self.__dict__[link.name] = link
        for joint in whole_joint_list:
            self.__dict__[joint.name] = joint
        if root_link is None:
            self.root_link = None
        else:
            self.root_link = self.__dict__[root_link.name]
            self.assoc(self.root_link)

        # Add hook of mimic joint.
        for j in self.urdf_robot_model.joints:
            if j.mimic is None:
                continue
            joint_a = self.__dict__[j.mimic.joint]
            joint_b = self.__dict__[j.name]
            multiplier = j.mimic.multiplier
            offset = j.mimic.offset
            joint_a.register_mimic_joint(joint_b, multiplier, offset)

        self._relevance_predicate_table = \
            self._compute_relevance_predicate_table()
        # Some models do not include 0 degrees within the valid joint limits,
        # so we call `init_pose` to round joint angles within the
        # limits internally.
        self.init_pose()

    def move_end_pos(self, pos, wrt='local', *args, **kwargs):
        pos = np.array(pos, dtype=np.float64)
        return self.inverse_kinematics(
            self.end_coords.copy_worldcoords().translate(pos, wrt),
            move_target=self.end_coords,
            *args, **kwargs)

    def move_end_rot(self, angle, axis, wrt='local', *args, **kwargs):
        rotation_axis = kwargs.pop('rotation_axis', True)
        return self.inverse_kinematics(
            self.end_coords.copy_worldcoords().rotate(angle, axis, wrt),
            move_target=self.end_coords,
            rotation_axis=rotation_axis,
            *args, **kwargs)

    def fix_leg_to_coords(self, fix_coords, leg='both', mid=0.5):
        """Fix robot's legs to a coords

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
            move_target=None,
            link_list=None,
            **kwargs):
        """Solve inverse kinematics.

        solve inverse kinematics, move move-target to target-coords look-at-
        target suppots t, nil, float-vector, coords, list of float-vector, list
        of coords link-list is set by default based on move-target -> root link
        link-list.
        """
        if move_target is None:
            move_target = self.end_coords
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

    def batch_inverse_kinematics(
            self,
            target_coords,
            move_target=None,
            link_list=None,
            rotation_axis=True,
            translation_axis=True,
            stop=100,
            thre=0.001,
            rthre=np.deg2rad(1.0),
            initial_angles="current",
            alpha=1.0,
            attempts_per_pose=1,
            random_initial_range=0.7,
            **kwargs):
        """Solve batch inverse kinematics for multiple target poses.

        This method efficiently processes multiple target poses in parallel,
        providing significant performance improvements over sequential IK solving.

        Parameters
        ----------
        target_coords : Union[np.ndarray, List[Coordinates]]
            Target poses as numpy array in shape (batch_size, 6) where each row is
            [x, y, z, roll, pitch, yaw] or (batch_size, 7) where quaternion is included
            [x, y, z, qx, qy, qz, qw], or list of Coordinates objects
        move_target : Optional
            Target link or end-effector. If None, uses self.end_coords
        link_list : Optional[List]
            List of links from root to target. If None, automatically computed from move_target
        rotation_axis : Union[bool, str, List]
            Rotation constraints. If True, use all axes. If False, no rotation.
            If string, specify axes (e.g., 'xy', 'z'). Can be list for multiple targets
        translation_axis : Union[bool, str, List]
            Translation constraints. If True, use all axes. If False, no translation.
            If string, specify axes (e.g., 'xy', 'z'). Can be list for multiple targets
        stop : int
            Maximum number of iterations
        thre : float
            Position error threshold in meters
        rthre : float
            Rotation error threshold in radians
        initial_angles : Optional[Union[np.ndarray, str]]
            Initial joint angles. Can be:
            - "current": Use current robot joint angles for all poses (default)
            - None or "random": Use random initial angles
            - np.ndarray of shape (batch_size, ndof): Use provided initial angles
            Note: When attempts_per_pose > 1, defaults to "random" for better exploration
        alpha : float
            Step size for gradient descent (0 < alpha <= 1)
        attempts_per_pose : int
            Number of attempts with different random initial poses per target (default: 1)
        random_initial_range : float
            Range for random initial poses as fraction of joint limits (0.0-1.0, default: 0.7)
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        Tuple[List[np.ndarray], List[bool], List[int]] or Tuple[List[np.ndarray], List[bool]]
            - List of joint angle solutions (each is ndarray matching self.angle_vector())
            - List of success flags indicating if IK was solved
            - List of attempt counts (only returned when attempts_per_pose > 1)

        Examples
        --------
        >>> import numpy as np
        >>> robot = Fetch()
        >>> robot.reset_pose()
        >>>
        >>> # Multiple target poses
        >>> target_poses = np.array([
        ...     [0.8, -0.3, 0.8, 0.0, np.deg2rad(30), np.deg2rad(-30)],
        ...     [0.7, -0.2, 0.9, 0.0, np.deg2rad(45), np.deg2rad(-15)],
        ... ])
        >>> solutions, success_flags = robot.batch_inverse_kinematics(target_poses)
        >>>
        >>> # Apply first successful solution
        >>> for i, (solution, success) in enumerate(zip(solutions, success_flags)):
        ...     if success:
        ...         robot.angle_vector(solution)
        ...         break
        """
        return self._batch_inverse_kinematics_impl(
            target_coords, move_target, link_list,
            rotation_axis, translation_axis, stop, thre, rthre,
            initial_angles, alpha, attempts_per_pose, random_initial_range, **kwargs)

    def _batch_inverse_kinematics_impl(
            self, target_coords, move_target, link_list,
            rotation_axis, translation_axis, stop, thre, rthre,
            initial_angles, alpha, attempts_per_pose, random_initial_range, **kwargs):
        """Internal implementation of batch inverse kinematics."""
        # Auto-adjust initial_angles based on attempts_per_pose
        if isinstance(initial_angles, str) and initial_angles == "current" and attempts_per_pose > 1:
            initial_angles = "random"

        if move_target is None:
            move_target = self.end_coords
        if link_list is None:
            if not isinstance(move_target, list):
                link_list = self.link_lists(move_target.parent)
            else:
                link_list = list(map(lambda mt: self.link_lists(mt.parent), move_target))

        # Handle single link list (not list of lists) for now
        if isinstance(link_list, list) and len(link_list) > 0 and isinstance(link_list[0], list):
            single_link_list = link_list[0]  # Use first link list for batch processing
        else:
            single_link_list = link_list

        # Convert input to consistent format: [x, y, z, qw, qx, qy, qz]
        if isinstance(target_coords, list) and all(isinstance(coord, Coordinates) for coord in target_coords):
            n_poses = len(target_coords)
            positions_xyz = np.array([coord.worldpos() for coord in target_coords])
            quaternions_wxyz = np.array([coord.quaternion for coord in target_coords])  # skrobot uses [w,x,y,z] order
            target_poses_xyz_qwxyz = np.concatenate([positions_xyz, quaternions_wxyz], axis=1)
        elif isinstance(target_coords, np.ndarray):
            if target_coords.ndim != 2:
                raise ValueError(f"target_coords must be 2D array, got shape {target_coords.shape}")

            n_poses = target_coords.shape[0]

            if target_coords.shape[1] == 6:
                # Convert 6D pose (x, y, z, roll, pitch, yaw) to 7D (x, y, z, qw, qx, qy, qz)
                positions_xyz = target_coords[:, :3]
                rpy_roll_pitch_yaw = target_coords[:, 3:]  # [roll, pitch, yaw] order
                quaternions_wxyz = np.array([rpy2quaternion(rpy) for rpy in rpy_roll_pitch_yaw])  # -> [w,x,y,z]
                target_poses_xyz_qwxyz = np.concatenate([positions_xyz, quaternions_wxyz], axis=1)
            elif target_coords.shape[1] == 7:
                target_poses_xyz_qwxyz = target_coords  # Assume [x,y,z,qw,qx,qy,qz] format
            else:
                raise ValueError(f"target_coords must have shape (batch, 6) or (batch, 7), got {target_coords.shape}")
        else:
            raise ValueError("target_coords must be numpy array or list of Coordinates objects")

        # Get joint information for the kinematic chain
        joint_list_without_fixed = self.joint_list_from_link_list(single_link_list, ignore_fixed_joint=True)
        ndof = calc_target_joint_dimension(joint_list_without_fixed)
        min_angles, max_angles = self.joint_limits_from_joint_list(joint_list_without_fixed)

        # Map kinematic chain joints to full robot joint indices
        robot_joint_list = self.joint_list
        joint_indices = []
        for joint in joint_list_without_fixed:
            if joint in robot_joint_list:
                joint_indices.append(robot_joint_list.index(joint))

        if initial_angles is None or (isinstance(initial_angles, str) and initial_angles == "random"):
            joint_angles_current = np.random.uniform(min_angles, max_angles, (n_poses, ndof))
        elif isinstance(initial_angles, str) and initial_angles == "current":
            current_full_angles = self.angle_vector()
            current_kinematic_angles = current_full_angles[joint_indices]
            joint_angles_current = np.tile(current_kinematic_angles, (n_poses, 1))
        elif isinstance(initial_angles, np.ndarray):
            if initial_angles.shape != (n_poses, ndof):
                raise ValueError(f"initial_angles must have shape ({n_poses}, {ndof}), got {initial_angles.shape}")
            joint_angles_current = initial_angles.copy()
        else:
            raise ValueError(
                f"initial_angles must be None, 'random', 'current', or np.ndarray, got {type(initial_angles)}")

        source_link = single_link_list[0]
        if hasattr(self, 'root_link') and source_link == self.root_link:
            base_to_source = np.eye(4, dtype=np.float64)
        else:
            base_to_source = source_link.parent.copy_worldcoords().T()

        if not isinstance(rotation_axis, list):
            rotation_axis = [rotation_axis] * n_poses
        if not isinstance(translation_axis, list):
            translation_axis = [translation_axis] * n_poses

        solutions, success_flags, attempt_counts = self._solve_batch_ik(
            single_link_list, target_poses_xyz_qwxyz, joint_angles_current, ndof, min_angles, max_angles,
            stop, thre, rthre, alpha, base_to_source,
            rotation_axis, translation_axis, joint_list_without_fixed,
            attempts_per_pose, random_initial_range
        )

        full_solutions = []
        full_av_org = self.angle_vector()
        for solution in solutions:
            full_av = full_av_org.copy()
            for i, joint_idx in enumerate(joint_indices):
                if i < len(solution):
                    full_av[joint_idx] = solution[i]
            full_solutions.append(full_av)

        return full_solutions, success_flags, attempt_counts

    def _solve_batch_ik(
            self, link_list, target_poses_xyz_qwxyz, initial_angles, ndof, min_angles, max_angles,
            stop, thre, rthre, alpha, base_to_source,
            rotation_axis, translation_axis, joint_list_without_fixed,
            attempts_per_pose, random_initial_range):
        """Batch IK solver using batch expansion for multiple attempts."""

        n_poses = target_poses_xyz_qwxyz.shape[0]

        expanded_batch_size = n_poses * attempts_per_pose
        expanded_targets = np.repeat(target_poses_xyz_qwxyz, attempts_per_pose, axis=0)
        expanded_initials = np.zeros((expanded_batch_size, ndof))

        centers = (np.array(min_angles) + np.array(max_angles)) / 2
        ranges = random_initial_range * (np.array(max_angles) - np.array(min_angles)) / 2

        for i in range(n_poses):
            start_idx = i * attempts_per_pose
            end_idx = start_idx + attempts_per_pose

            if initial_angles is not None:
                expanded_initials[start_idx] = initial_angles[i]
                if attempts_per_pose > 1:
                    random_initials = centers + np.random.uniform(
                        -ranges, ranges, size=(attempts_per_pose - 1, ndof)
                    )
                    expanded_initials[start_idx + 1:end_idx] = random_initials
            else:
                random_initials = centers + np.random.uniform(
                    -ranges, ranges, size=(attempts_per_pose, ndof)
                )
                expanded_initials[start_idx:end_idx] = random_initials

        expanded_rotation_axis = []
        expanded_translation_axis = []
        for i in range(n_poses):
            for _ in range(attempts_per_pose):
                if isinstance(rotation_axis, list):
                    expanded_rotation_axis.append(rotation_axis[i])
                else:
                    expanded_rotation_axis.append(rotation_axis)

                if isinstance(translation_axis, list):
                    expanded_translation_axis.append(translation_axis[i])
                else:
                    expanded_translation_axis.append(translation_axis)

        # Solve all attempts in parallel using batch processing
        expanded_solutions, expanded_success = self._solve_batch_ik_internal(
            link_list, expanded_targets, expanded_initials,
            stop, thre, rthre, alpha, base_to_source,
            expanded_rotation_axis, expanded_translation_axis,
            joint_list_without_fixed, min_angles, max_angles
        )

        solutions = []
        success_flags = []
        attempt_counts = []

        for i in range(n_poses):
            start_idx = i * attempts_per_pose
            end_idx = start_idx + attempts_per_pose

            target_solutions = expanded_solutions[start_idx:end_idx]
            target_success = expanded_success[start_idx:end_idx]

            best_solution = target_solutions[-1]
            is_solved = False
            attempts_used = attempts_per_pose

            for attempt_idx, (sol, success) in enumerate(zip(target_solutions, target_success)):
                if success:
                    best_solution = sol
                    is_solved = True
                    attempts_used = attempt_idx + 1
                    break

            solutions.append(best_solution)
            success_flags.append(is_solved)
            attempt_counts.append(attempts_used)

        return solutions, success_flags, attempt_counts

    def _solve_batch_ik_internal(
            self, link_list, target_poses_xyz_qwxyz, joint_angles_current,
            stop, thre, rthre, alpha, base_to_source,
            rotation_axis, translation_axis,
            joint_list_without_fixed, min_angles, max_angles):
        """Internal batch IK solver."""
        # Use integrated batch IK functions

        batch_size = target_poses_xyz_qwxyz.shape[0]

        # Track which problems are still unsolved
        unsolved_mask = np.ones(batch_size, dtype=bool)
        solutions = [None] * batch_size
        success_flags = [False] * batch_size

        # Iterative solving
        for iteration in range(stop):
            if not np.any(unsolved_mask):
                break

            # Get indices of unsolved problems
            unsolved_indices = np.where(unsolved_mask)[0]

            # Update only unsolved problems
            joint_angles_unsolved = joint_angles_current[unsolved_mask]
            target_poses_unsolved = target_poses_xyz_qwxyz[unsolved_mask]

            # Use unified constrained IK (handles both constrained and unconstrained cases)
            if isinstance(rotation_axis, list):
                rotation_axis_unsolved = [rotation_axis[i] for i in unsolved_indices]
            else:
                rotation_axis_unsolved = [rotation_axis] * len(unsolved_indices)

            if isinstance(translation_axis, list):
                translation_axis_unsolved = [translation_axis[i] for i in unsolved_indices]
            else:
                translation_axis_unsolved = [translation_axis] * len(unsolved_indices)

            joint_angles_updated, pose_errors = self._batch_ik_step_with_constraints(
                link_list, target_poses_unsolved, joint_angles_unsolved,
                alpha, base_to_source,
                rotation_axis_unsolved, translation_axis_unsolved,
                joint_list_without_fixed, min_angles, max_angles
            )

            # Check convergence with respect to constraints
            converged = np.zeros(len(unsolved_indices), dtype=bool)

            for i, global_idx in enumerate(unsolved_indices):
                # Apply the same constraints as used in IK
                constrained_pose_error = pose_errors[i].copy()

                # Handle rotation constraints
                if isinstance(rotation_axis, list):
                    rot_axis = rotation_axis[global_idx]
                else:
                    rot_axis = rotation_axis

                if rot_axis is False:
                    constrained_pose_error[:3] = 0
                elif isinstance(rot_axis, str):
                    # For mirror constraints (xm, ym, zm), don't zero out any rotation errors
                    # since we want to converge on the mirrored solution
                    if rot_axis.lower() not in ['xm', 'ym', 'zm']:
                        # Standard axis constraints
                        if 'x' not in rot_axis.lower():
                            constrained_pose_error[0] = 0
                        if 'y' not in rot_axis.lower():
                            constrained_pose_error[1] = 0
                        if 'z' not in rot_axis.lower():
                            constrained_pose_error[2] = 0

                # Handle translation constraints
                if isinstance(translation_axis, list):
                    trans_axis = translation_axis[global_idx]
                else:
                    trans_axis = translation_axis

                if trans_axis is False:
                    constrained_pose_error[3:] = 0
                elif isinstance(trans_axis, str):
                    # For mirror constraints (xm, ym, zm), don't zero out any translation errors
                    # since we want to converge on the mirrored solution
                    if trans_axis.lower() not in ['xm', 'ym', 'zm']:
                        # Standard axis constraints
                        if 'x' not in trans_axis.lower():
                            constrained_pose_error[3] = 0
                        if 'y' not in trans_axis.lower():
                            constrained_pose_error[4] = 0
                        if 'z' not in trans_axis.lower():
                            constrained_pose_error[5] = 0

                # Check convergence only for active constraints
                position_error = np.linalg.norm(constrained_pose_error[3:])
                rotation_error = np.linalg.norm(constrained_pose_error[:3])

                converged[i] = position_error < thre and rotation_error < rthre

            # Update solutions for converged problems
            converged_global_indices = unsolved_indices[converged]
            for idx in converged_global_indices:
                local_idx = np.where(unsolved_indices == idx)[0][0]
                solutions[idx] = joint_angles_updated[local_idx].copy()
                success_flags[idx] = True

            # Update current angles
            joint_angles_current[unsolved_mask] = joint_angles_updated

            # Update unsolved mask
            unsolved_mask[converged_global_indices] = False

        # Fill in remaining unsolved problems with their final angles
        for idx in range(batch_size):
            if solutions[idx] is None:
                solutions[idx] = joint_angles_current[idx].copy()

        return solutions, success_flags

    def _batch_ik_step_with_constraints(
            self, link_list, target_poses, joint_angles_current, alpha, base_to_source,
            rotation_axis_list, translation_axis_list,
            joint_list_without_fixed, min_angles, max_angles):
        """Batch IK step with per-problem axis constraints."""
        # Use integrated batch IK functions

        batch_size = joint_angles_current.shape[0]
        ndof = joint_angles_current.shape[1]

        # Get jacobians
        jacobian_matrices = self._jacobian_batch(link_list, joint_angles_current, base_to_source=base_to_source)

        # Calculate current poses
        current_poses = self._forward_kinematics_batch(
            link_list, joint_angles_current, return_quaternion=True, base_to_source=base_to_source
        )

        # Calculate pose errors
        pose_errors = np.zeros((batch_size, 6))

        # Calculate translation errors, potentially with mirroring
        translation_errors = target_poses[:, :3] - current_poses[:, :3]

        for i in range(batch_size):
            trans_axis = translation_axis_list[i]

            if isinstance(trans_axis, str) and trans_axis.lower() in ['xm', 'ym', 'zm']:
                # For translation mirror constraints, check if mirrored translation gives smaller error
                axis_char = trans_axis[0].lower()

                # Calculate error with original translation difference
                original_error = translation_errors[i].copy()
                original_error_norm = np.linalg.norm(original_error)

                # Calculate error with mirrored translation along specified axis
                mirrored_error = original_error.copy()
                if axis_char == 'x':
                    mirrored_error[0] = -mirrored_error[0]
                elif axis_char == 'y':
                    mirrored_error[1] = -mirrored_error[1]
                else:  # 'z'
                    mirrored_error[2] = -mirrored_error[2]

                mirrored_error_norm = np.linalg.norm(mirrored_error)

                # Use the translation error that gives smaller magnitude
                if mirrored_error_norm < original_error_norm:
                    translation_errors[i] = mirrored_error

        pose_errors[:, :3] = translation_errors

        # Orientation errors with mirror constraint handling
        current_pose_quat_inv = quaternion_inverse(current_poses[:, 3:])

        # Calculate rotation errors with proper single-axis constraint handling
        adjusted_target_poses = target_poses.copy()

        for i in range(batch_size):
            rot_axis = rotation_axis_list[i]

            if isinstance(rot_axis, str) and rot_axis.lower() in ['xm', 'ym', 'zm']:
                # For mirror constraints, check if mirrored target gives smaller error
                axis_char = rot_axis[0].lower()
                target_rot_matrix = quaternion2matrix(target_poses[i, 3:])
                current_rot_matrix = quaternion2matrix(current_poses[i, 3:])

                # Calculate error with original target
                original_error_matrix = np.matmul(current_rot_matrix.T, target_rot_matrix)
                original_error = matrix_log(original_error_matrix)
                original_error_norm = np.linalg.norm(original_error)

                # Calculate error with mirrored target (180° rotation around axis)
                if axis_char == 'x':
                    mirror_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                elif axis_char == 'y':
                    mirror_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                else:  # 'z'
                    mirror_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

                mirrored_target_rot = np.matmul(target_rot_matrix, mirror_matrix)
                mirrored_error_matrix = np.matmul(current_rot_matrix.T, mirrored_target_rot)
                mirrored_error = matrix_log(mirrored_error_matrix)
                mirrored_error_norm = np.linalg.norm(mirrored_error)

                # Use the target orientation that gives smaller error
                if mirrored_error_norm < original_error_norm:
                    adjusted_target_poses[i, 3:] = matrix2quaternion(mirrored_target_rot)

            elif isinstance(rot_axis, str) and rot_axis.lower() in ['x', 'y', 'z']:
                # For single axis constraints, use coordinate frame difference_rotation method
                # This ensures the same behavior as regular IK

                # Create coordinate objects
                current_coord = Coordinates()
                current_coord.newcoords(quaternion2matrix(current_poses[i, 3:]), current_poses[i, :3])

                target_coord = Coordinates()
                target_coord.newcoords(quaternion2matrix(target_poses[i, 3:]), target_poses[i, :3])

                # Calculate rotation difference using the same method as regular IK
                dif_rot = current_coord.difference_rotation(target_coord, rotation_axis=rot_axis.lower())

                # Apply the calculated rotation to current pose to get the adjusted target
                if np.linalg.norm(dif_rot) > 1e-6:
                    corrected_coord = current_coord.copy_worldcoords()
                    corrected_coord.rotate_with_matrix(rodrigues(dif_rot), wrt='local')
                    adjusted_target_poses[i, 3:] = matrix2quaternion(corrected_coord.worldrot())
                else:
                    # No rotation needed - target orientation becomes current orientation
                    adjusted_target_poses[i, 3:] = current_poses[i, 3:]

        # Calculate rotation errors with potentially adjusted targets
        rotation_error_quat = quaternion_multiply(adjusted_target_poses[:, 3:], current_pose_quat_inv)
        rotation_error_rpy = quaternion2rpy(rotation_error_quat)[0][:, ::-1]
        pose_errors[:, 3:] = rotation_error_rpy
        pose_errors = pose_errors[:, [3, 4, 5, 0, 1, 2]]

        # Apply constraints and calculate delta for each problem
        joint_angle_deltas = np.zeros((batch_size, ndof))

        for i in range(batch_size):
            # Create active constraint mask using row exclusion approach
            active_rows = []

            # Handle rotation constraints
            rot_axis = rotation_axis_list[i]
            if rot_axis is True:
                active_rows.extend([0, 1, 2])  # All rotation DOF
            elif isinstance(rot_axis, str):
                # Handle mirror constraints (xm, ym, zm)
                if rot_axis.lower() in ['xm', 'ym', 'zm']:
                    # For mirror constraints, allow all rotation DOF but we'll handle mirroring in error calculation
                    active_rows.extend([0, 1, 2])
                else:
                    # For single axis constraints, use all rotation DOF
                    # The constraint is applied in the error calculation phase
                    if rot_axis.lower() in ['x', 'y', 'z']:
                        active_rows.extend([0, 1, 2])  # All rotation DOF for single axis constraints
                    else:
                        # Multi-axis constraints like 'xy', 'xyz', etc.
                        if 'x' in rot_axis.lower():
                            active_rows.append(0)
                        if 'y' in rot_axis.lower():
                            active_rows.append(1)
                        if 'z' in rot_axis.lower():
                            active_rows.append(2)
            # If rot_axis is False, no rotation rows are added

            # Handle translation constraints
            trans_axis = translation_axis_list[i]
            if trans_axis is True:
                active_rows.extend([3, 4, 5])  # All translation DOF
            elif isinstance(trans_axis, str):
                # Handle mirror constraints (xm, ym, zm) for translation
                if trans_axis.lower() in ['xm', 'ym', 'zm']:
                    # For translation mirror constraints, allow all translation DOF
                    active_rows.extend([3, 4, 5])
                else:
                    # Standard axis constraints
                    if 'x' in trans_axis.lower():
                        active_rows.append(3)
                    if 'y' in trans_axis.lower():
                        active_rows.append(4)
                    if 'z' in trans_axis.lower():
                        active_rows.append(5)
            # If trans_axis is False, no translation rows are added

            # Extract only the active rows (exclude unwanted constraints)
            if len(active_rows) > 0:
                jacobian_active = jacobian_matrices[i][active_rows, :]
                pose_error_active = pose_errors[i][active_rows]

                # Solve the reduced problem
                jacobian_pinv_active = jacobian_inverse(jacobian_active[np.newaxis, :, :])[0]
                joint_angle_deltas[i] = np.dot(jacobian_pinv_active, pose_error_active)
            else:
                # No active constraints - don't move
                joint_angle_deltas[i] = np.zeros(ndof)

        # Update angles
        joint_angles_updated = joint_angles_current + alpha * joint_angle_deltas

        # Apply joint limits
        joint_angles_updated = np.clip(joint_angles_updated, min_angles, max_angles)

        return joint_angles_updated, pose_errors

    def inverse_kinematics_loop(self,
                                dif_pos,
                                dif_rot,
                                move_target,
                                link_list=None,
                                target_coords=None,
                                **kwargs):
        """move move_target using dif_pos and dif_rot.

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
            rthre=0.001,
            **kwargs):
        """Solve look at inverse kinematics

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

        kwargs.pop('stop', 100)
        self.calc_target_joint_dimension(link_list)
        target_coords = orient_coords_to_axis(
            make_coords(pos=look_at_pos),
            look_at_pos - move_target.worldpos())

        return self.inverse_kinematics(target_coords,
                                       move_target=move_target,
                                       link_list=link_list,
                                       translation_axis=translation_axis,
                                       rotation_axis=rotation_axis,
                                       **kwargs)

    def look_at_hand(self, coords):
        if coords == 'rarm':
            coords = self.rarm.end_coords
        elif coords == 'larm':
            coords = self.larm.end_coords
        self.inverse_kinematics_loop_for_look_at(
            self.head_end_coords,
            coords.worldpos(),
            self.head.link_list)

    def look_at(self, coords, target=None, link_list=None):
        if target is None:
            target = self.head_end_coords
        if link_list is None:
            link_list = self.head.link_list
        return self.inverse_kinematics_loop_for_look_at(
            target, coords.worldpos(), link_list)

    @staticmethod
    def sanitize_joint_limits(min_angles, max_angles, joint_limit_eps=0.001):
        """Sanitize joint limits.

        Sanitize joint limits, replacing infinities and ensuring the range
        is non-zero.

        """
        min_angles = np.array(min_angles)
        max_angles = np.array(max_angles)
        min_angles = np.where(min_angles == -np.inf, -np.pi, min_angles)
        max_angles = np.where(max_angles == np.inf, np.pi, max_angles)
        if np.any((max_angles - min_angles) < joint_limit_eps):
            raise ValueError(
                'Joint limits are too narrow, '
                'leading to zero standard deviation in samples.')
        return min_angles, max_angles

    @staticmethod
    def joint_list_from_link_list(link_list, ignore_fixed_joint=True):
        """Generate a list of joints from a list of links.

        Generate a list of joints from a list of links, optionally ignoring
        fixed joints.

        """
        if ignore_fixed_joint:
            return [l.joint for l in link_list if hasattr(l, 'joint')
                    and l.joint.__class__.__name__ != 'FixedJoint']
        else:
            return [l.joint for l in link_list if hasattr(l, 'joint')]

    @staticmethod
    def joint_limits_from_joint_list(joint_list):
        """Compute joint limits from a list of joints.

        """
        return RobotModel.sanitize_joint_limits(
            [j.min_angle for j in joint_list],
            [j.max_angle for j in joint_list],
            joint_limit_eps=0.001)

    def update_mass_properties(self):
        """Update robot mass properties by summing over all links.

        Returns
        -------
        dict
            Dictionary containing total mass, center of mass, and inertia tensor.
        """
        total_mass = 0.0
        total_moment = np.zeros(3)

        for link in self.link_list:
            if hasattr(link, 'mass') and link.mass > 0:
                total_mass += link.mass
                if link.centroid is not None:
                    com_world = link.worldpos() + link.worldrot().dot(link.centroid)
                else:
                    com_world = link.worldpos()
                total_moment += link.mass * com_world

        if total_mass > 0:
            total_centroid = total_moment / total_mass
        else:
            total_centroid = np.zeros(3)

        # Calculate inertia tensor about total centroid
        total_inertia = np.zeros((3, 3))
        for link in self.link_list:
            if hasattr(link, 'mass') and link.mass > 0:
                if link.centroid is not None:
                    com_world = link.worldpos() + link.worldrot().dot(link.centroid)
                else:
                    com_world = link.worldpos()

                # Transform link inertia to world frame
                R = link.worldrot()
                link_inertia_world = R.dot(link.inertia_tensor).dot(R.T)

                # Parallel axis theorem
                r = com_world - total_centroid
                r_cross = np.array([[0, -r[2], r[1]],
                                   [r[2], 0, -r[0]],
                                   [-r[1], r[0], 0]])
                total_inertia += link_inertia_world - link.mass * r_cross.dot(r_cross)

        return {
            'total_mass': total_mass,
            'total_centroid': total_centroid,
            'total_inertia': total_inertia
        }

    def centroid(self, update_mass_properties=True):
        """Calculate total robot centroid (Center Of Gravity, COG) in world coordinates.

        Parameters
        ----------
        update_mass_properties : bool, optional
            If True, recalculate total mass properties for all links and return
            the total robot centroid. If False, return pre-computed centroid
            without recalculation. Default is True.

        Returns
        -------
        numpy.ndarray
            3D vector of robot centroid position [m] in world coordinates.

        Notes
        -----
        This method calculates the center of gravity (centroid) of the entire robot
        by considering the mass and centroid of each link. The calculation is based
        on the weighted average of all link centroids in world coordinates.

        The centroid is calculated as:
        centroid = sum(mass_i * position_i) / sum(mass_i)

        where mass_i and position_i are the mass and world position of link i.

        Examples
        --------
        >>> robot = RobotModel()
        >>> cog = robot.centroid()  # Calculate and return centroid
        >>> cog_cached = robot.centroid(update_mass_properties=False)  # Use cached values
        """
        if update_mass_properties:
            mass_props = self.update_mass_properties()
            # Update the cache with the latest mass properties
            self._cached_mass_props = mass_props
            return mass_props['total_centroid']
        else:
            # Return cached centroid if available
            if hasattr(self, '_cached_mass_props'):
                return self._cached_mass_props['total_centroid']
            else:
                # If no cached values, calculate once
                mass_props = self.update_mass_properties()
                self._cached_mass_props = mass_props
                return mass_props['total_centroid']

    def calc_av_vel_acc_from_pos(self, dt, av_prev=None, av_curr=None, av_next=None):
        """Calculate joint velocities and accelerations from angle vectors.

        Parameters
        ----------
        dt : float
            Time step [s].
        av_prev : np.ndarray, optional
            Previous angle vector [rad]. If None, uses current angle vector.
        av_curr : np.ndarray, optional
            Current angle vector [rad]. If None, uses current angle vector.
        av_next : np.ndarray, optional
            Next angle vector [rad]. If None, assumes zero velocity.

        Returns
        -------
        joint_velocities : np.ndarray
            Joint velocities [rad/s] or [m/s] for linear joints.
        joint_accelerations : np.ndarray
            Joint accelerations [rad/s^2] or [m/s^2] for linear joints.
        """
        if av_curr is None:
            av_curr = self.angle_vector()

        if av_prev is None:
            av_prev = av_curr.copy()

        if av_next is None:
            av_next = av_curr.copy()

        # Calculate velocities using finite differences
        joint_velocities = (av_next - av_prev) / (2.0 * dt)

        # Calculate accelerations using finite differences
        joint_accelerations = (av_next - 2.0 * av_curr + av_prev) / (dt * dt)

        return joint_velocities, joint_accelerations

    def forward_all_kinematics(self, joint_velocities=None, joint_accelerations=None,
                               root_velocity=None, root_acceleration=None):
        """Propagate velocities and accelerations through kinematic chain.

        Parameters
        ----------
        joint_velocities : np.ndarray, optional
            Joint velocities [rad/s] or [m/s]. If None, uses zeros.
        joint_accelerations : np.ndarray, optional
            Joint accelerations [rad/s^2] or [m/s^2]. If None, uses zeros.
        root_velocity : np.ndarray, optional
            Root link spatial velocity [m/s] + angular velocity [rad/s].
        root_acceleration : np.ndarray, optional
            Root link spatial acceleration [m/s^2] + angular acceleration [rad/s^2].
        """
        if joint_velocities is None:
            joint_velocities = np.zeros(len(self.joint_list))
        if joint_accelerations is None:
            joint_accelerations = np.zeros(len(self.joint_list))
        if root_velocity is None:
            root_velocity = np.zeros(6)  # [linear_vel, angular_vel]
        if root_acceleration is None:
            root_acceleration = np.zeros(6)  # [linear_acc, angular_acc]

        # Set root link velocities and accelerations
        root_link = self.root_link
        root_link.spatial_velocity = root_velocity[:3]
        root_link.angular_velocity = root_velocity[3:]
        root_link.spatial_acceleration = root_acceleration[:3]
        root_link.angular_acceleration = root_acceleration[3:]

        # Propagate through kinematic chain
        joint_idx = 0
        for joint in self.joint_list:
            if joint is None:
                continue

            parent_link = joint.parent_link
            child_link = joint.child_link

            # Joint axis in world coordinates
            joint_axis = parent_link.worldrot().dot(joint.axis)

            # Propagate angular velocity
            if joint.__class__.__name__ == 'LinearJoint':
                child_link.angular_velocity = parent_link.angular_velocity.copy()
            else:  # rotational joint
                child_link.angular_velocity = (parent_link.angular_velocity +
                                                joint_velocities[joint_idx] * joint_axis)

            # Propagate spatial velocity
            joint_pos = joint.parent_link.worldpos()
            child_pos = child_link.worldpos()
            r = child_pos - joint_pos

            if joint.__class__.__name__ == 'LinearJoint':
                child_link.spatial_velocity = (parent_link.spatial_velocity +
                                                joint_velocities[joint_idx] * joint_axis +
                                                np.cross(parent_link.angular_velocity, r))
            else:  # rotational joint
                child_link.spatial_velocity = (parent_link.spatial_velocity +
                                                np.cross(parent_link.angular_velocity, r))

            # Propagate angular acceleration
            if joint.__class__.__name__ == 'LinearJoint':
                child_link.angular_acceleration = parent_link.angular_acceleration.copy()
            else:  # rotational joint
                child_link.angular_acceleration = (parent_link.angular_acceleration +
                                                   joint_accelerations[joint_idx] * joint_axis +
                                                   np.cross(parent_link.angular_velocity,
                                                            joint_velocities[joint_idx] * joint_axis))

            # Propagate spatial acceleration
            if joint.__class__.__name__ == 'LinearJoint':
                child_link.spatial_acceleration = (parent_link.spatial_acceleration +
                                                    joint_accelerations[joint_idx] * joint_axis +
                                                    np.cross(parent_link.angular_acceleration, r) +
                                                    np.cross(parent_link.angular_velocity,
                                                             np.cross(parent_link.angular_velocity, r)))
            else:  # rotational joint
                child_link.spatial_acceleration = (parent_link.spatial_acceleration +
                                                    np.cross(parent_link.angular_acceleration, r) +
                                                    np.cross(parent_link.angular_velocity,
                                                             np.cross(parent_link.angular_velocity, r)))

            joint_idx += 1

    def inverse_dynamics(self, external_forces=None, external_moments=None,
                         external_coords=None, gravity=None):
        """Compute joint torques using inverse dynamics (Newton-Euler algorithm).

        Parameters
        ----------
        external_forces : list of np.ndarray, optional
            External forces [N] applied at external_coords.
        external_moments : list of np.ndarray, optional
            External moments [Nm] applied at external_coords.
        external_coords : list of coordinates, optional
            Coordinate frames where external forces/moments are applied.
        gravity : np.ndarray, optional
            Gravity vector [m/s^2]. Defaults to [0, 0, -9.81].

        Returns
        -------
        joint_torques : np.ndarray
            Joint torques [Nm] or forces [N] for linear joints.
        """
        if gravity is None:
            gravity = np.array([0, 0, -9.81])

        # Clear previous external forces
        for link in self.link_list:
            link.clear_external_wrench()
            link._internal_force.fill(0.0)
            link._internal_moment.fill(0.0)

        # Apply external forces and moments
        if external_forces is not None and external_coords is not None:
            for force, coords in zip(external_forces, external_coords):
                if hasattr(coords, 'parent') and coords.parent in self.link_list:
                    coords.parent.apply_external_wrench(force=force,
                                                         point=coords.worldpos())

        if external_moments is not None and external_coords is not None:
            for moment, coords in zip(external_moments, external_coords):
                if hasattr(coords, 'parent') and coords.parent in self.link_list:
                    coords.parent.apply_external_wrench(moment=moment)

        # Add gravity to all links
        for link in self.link_list:
            if hasattr(link, 'mass') and link.mass > 0:
                gravity_force = link.mass * gravity
                link.apply_external_wrench(force=gravity_force)

        # Backward propagation: compute forces and moments from leaves to root
        joint_torques = np.zeros(len(self.joint_list))

        # First pass: compute forces and moments for each link
        for link in self.link_list:
            if hasattr(link, 'mass') and link.mass > 0:
                # Get center of mass in world coordinates
                if link.centroid is not None:
                    com_world = link.worldpos() + link.worldrot().dot(link.centroid)
                else:
                    com_world = link.worldpos()

                # Check if this is static case (no accelerations)
                is_static = (np.allclose(link.spatial_acceleration, np.zeros(3)) and
                             np.allclose(link.angular_acceleration, np.zeros(3)))

                if is_static:
                    # Static case: only external forces (gravity)
                    link._internal_force = link.ext_force.copy()

                    # Moment about link origin due to gravity at CoM
                    r_com = com_world - link.worldpos()
                    link._internal_moment = np.cross(r_com, link.ext_force) + link.ext_moment
                else:
                    # Dynamic case: include inertial forces
                    # Linear momentum rate
                    F_inertial = link.mass * link.spatial_acceleration

                    # Angular momentum rate about center of mass
                    R = link.worldrot()
                    I_world = R.dot(link.inertia_tensor).dot(R.T)

                    M_inertial = (I_world.dot(link.angular_acceleration) +
                                  np.cross(link.angular_velocity,
                                           I_world.dot(link.angular_velocity)))

                    # Add moment due to linear acceleration of center of mass
                    r_com = com_world - link.worldpos()
                    M_inertial += np.cross(r_com, F_inertial)

                    # Total force and moment
                    link._internal_force = F_inertial + link.ext_force
                    link._internal_moment = M_inertial + link.ext_moment
            else:
                link._internal_force = np.zeros(3)
                link._internal_moment = np.zeros(3)

        # Second pass: propagate forces from children to parents using kinematic tree
        def propagate_forces_recursive(link):
            """Recursively propagate forces from children to this link."""
            # First, propagate forces from all children
            for child_link in link.child_links:
                propagate_forces_recursive(child_link)

                # Add child's forces to this link
                link._internal_force += child_link._internal_force

                # Find the joint connecting this link to child
                connecting_joint = None
                for joint in self.joint_list:
                    if joint.parent_link == link and joint.child_link == child_link:
                        connecting_joint = joint
                        break

                # Use joint position for moment calculation if available
                if connecting_joint:
                    # The joint position is at the child link's origin in the default pose
                    joint_pos = child_link.worldpos()
                    r_child = joint_pos - link.worldpos()
                else:
                    # Fallback to link position
                    r_child = child_link.worldpos() - link.worldpos()

                link._internal_moment += (child_link._internal_moment +
                                           np.cross(r_child, child_link._internal_force))

        # Start propagation from root link
        propagate_forces_recursive(self.root_link)

        # Extract joint torques
        joint_idx = 0
        for joint in self.joint_list:
            if joint is None:
                continue

            child_link = joint.child_link
            # Get joint axis in world coordinates
            # The axis is defined in the child link's local frame
            joint_axis = child_link.worldrot().dot(joint.axis)

            # Project force/moment onto joint axis
            # Check if it's a linear joint by class type
            if joint.__class__.__name__ == 'LinearJoint':
                joint_torques[joint_idx] = np.dot(child_link._internal_force, joint_axis)
            else:  # rotational joint
                # For rotational joints, the torque is the moment about the joint axis
                # The moment is already calculated about the correct point in propagate_forces_recursive
                joint_torques[joint_idx] = np.dot(child_link._internal_moment, joint_axis)

            joint_idx += 1

        return joint_torques

    def torque_vector(self, force_list=None, moment_list=None, target_coords=None,
                      calc_statics_p=True, dt=0.005, av=None, av_prev=None, av_next=None,
                      root_coords=None, root_coords_prev=None, root_coords_next=None,
                      gravity=None):
        """Calculate joint torques using inverse dynamics.

        This method computes the joint torques required to achieve
        specified motions while balancing external forces using inverse dynamics.

        Parameters
        ----------
        force_list : list of np.ndarray, optional
            External forces [N] applied at target_coords.
        moment_list : list of np.ndarray, optional
            External moments [Nm] applied at target_coords.
        target_coords : list of coordinates, optional
            Coordinate frames where forces/moments are applied.
        calc_statics_p : bool, optional
            If True, compute statics only. If False, compute full dynamics.
            Defaults to True.
        dt : float, optional
            Time step for finite difference computation [s]. Defaults to 0.005.
        av : np.ndarray, optional
            Current joint angle vector [rad]. If None, uses current angles.
        av_prev : np.ndarray, optional
            Previous joint angle vector [rad]. For dynamics computation.
        av_next : np.ndarray, optional
            Next joint angle vector [rad]. For dynamics computation.
        root_coords : coordinates, optional
            Current root link coordinates. If None, uses current coordinates.
        root_coords_prev : coordinates, optional
            Previous root link coordinates. For dynamics computation.
        root_coords_next : coordinates, optional
            Next root link coordinates. For dynamics computation.
        gravity : np.ndarray, optional
            Gravity vector [m/s^2]. Defaults to [0, 0, -9.81].

        Returns
        -------
        joint_torques : np.ndarray
            Joint torques [Nm] or forces [N] for linear joints required to
            achieve the specified motion and balance external forces.

        Examples
        --------
        >>> # Compute static torques to balance gravity
        >>> torques = robot.torque_vector()
        >>>
        >>> # Compute torques with external force on end effector
        >>> force = np.array([10, 0, 0])  # 10N in x direction
        >>> torques = robot.torque_vector(
        ...     force_list=[force],
        ...     target_coords=[robot.rarm.end_coords]
        ... )
        >>>
        >>> # Compute dynamic torques for motion
        >>> torques = robot.torque_vector(
        ...     calc_statics_p=False,
        ...     av_prev=prev_angles,
        ...     av=curr_angles,
        ...     av_next=next_angles,
        ...     dt=0.01
        ... )
        """
        if av is None:
            av = self.angle_vector()

        # Set joint angles
        self.angle_vector(av)

        # Initialize velocities and accelerations
        joint_velocities = np.zeros(len(self.joint_list))
        joint_accelerations = np.zeros(len(self.joint_list))
        root_velocity = np.zeros(6)
        root_acceleration = np.zeros(6)

        # Compute velocities and accelerations for dynamics
        if not calc_statics_p:
            if av_prev is not None or av_next is not None:
                joint_velocities, joint_accelerations = self.calc_av_vel_acc_from_pos(
                    dt, av_prev, av, av_next)

            # TODO: Add root coordinate velocity/acceleration computation
            # if root_coords_prev is not None or root_coords_next is not None:
            #     root_velocity, root_acceleration = self.calc_root_coords_vel_acc_from_pos(
            #         dt, root_coords_prev, root_coords, root_coords_next)

        # Propagate kinematics
        self.forward_all_kinematics(joint_velocities, joint_accelerations,
                                    root_velocity, root_acceleration)

        # Handle gravity settings
        if gravity is None:
            # Default: downward gravity
            gravity = np.array([0, 0, -9.80665])  # Downward gravity in m/s^2

        # Compute inverse dynamics
        joint_torques = self.inverse_dynamics(
            external_forces=force_list,
            external_moments=moment_list,
            external_coords=target_coords,
            gravity=gravity
        )

        return joint_torques

    # ============================================================================
    # Batch Inverse Kinematics Implementation
    # ============================================================================

    @staticmethod
    def _batch_fk_iteration(joint, x_i, base_T_joint):
        """Single iteration of batch forward kinematics for a joint."""
        batch_size = x_i.shape[0]
        parent_T_child_fixed = np.repeat(
            joint.default_coords.T()[np.newaxis, :, :], batch_size, axis=0)
        base_T_joint = np.matmul(base_T_joint, parent_T_child_fixed)
        if joint.__class__.__name__ == "RotationalJoint":
            joint_rotation = rodrigues(joint.axis, x_i, skip_normalization=True)
            T = np.tile(np.eye(4, dtype=x_i.dtype), (batch_size, 1, 1))
            T[:, 0:3, 0:3] = joint_rotation
            return np.matmul(base_T_joint, T)
        if joint.__class__.__name__ == "LinearJoint":
            translations = np.outer(x_i, joint.axis)
            joint_fixed_T_joint = np.tile(
                np.eye(4, dtype=x_i.dtype), (batch_size, 1, 1))
            joint_fixed_T_joint[:, 0:3, 3] = translations
            return np.matmul(base_T_joint, joint_fixed_T_joint)
        if joint.__class__.__name__ == "FixedJoint":
            return base_T_joint
        raise RuntimeError(f"Unsupported joint type: {type(joint)}")

    @staticmethod
    def _forward_kinematics_batch(
            link_list, x, dtype=np.float64,
            return_quaternion=True, return_full_joint_fk=False,
            return_full_link_fk=False, base_to_source=None):
        """Batch forward kinematics computation."""

        batch_size = x.shape[0]
        base_T_joint = np.tile(np.eye(4, dtype=dtype), (batch_size, 1, 1))
        if base_to_source is not None:
            base_T_joint = np.matmul(base_to_source, base_T_joint)
        base_T_joints = [base_T_joint]
        base_T_links = [base_T_joint]

        i = 0
        joint_list = RobotModel.joint_list_from_link_list(link_list, ignore_fixed_joint=False)
        ndof = calc_target_joint_dimension(joint_list)
        for joint in joint_list:
            i = min(i, ndof - 1)
            base_T_joint_new = RobotModel._batch_fk_iteration(joint, x[:, i], base_T_joint)
            base_T_links.append(base_T_joint_new)
            if joint.__class__.__name__ == 'RotationalJoint':
                base_T_joints.append(base_T_joint_new)
                i += 1
            if joint.__class__.__name__ == "LinearJoint":
                base_T_joints.append(base_T_joint_new)
                i += 1
            if joint.__class__.__name__ == "FixedJoint":
                base_T_joints[-1] = base_T_joint_new
            base_T_joint = base_T_joint_new

        if return_quaternion:
            quaternions = matrix2quaternion(base_T_joint[:, 0:3, 0:3])
            translations = base_T_joint[:, 0:3, 3]
            base_T_joint = np.concatenate([translations, quaternions], axis=1)

        if return_full_joint_fk:
            ret = np.stack(base_T_joints, axis=1)
            return ret

        return base_T_joint

    @staticmethod
    def _jacobian_batch(links, x: np.ndarray,
                        base_to_source=None) -> np.ndarray:
        """Batch Jacobian computation."""
        batch_size = x.shape[0]

        joint_list = RobotModel.joint_list_from_link_list(links, ignore_fixed_joint=False)
        ndof = calc_target_joint_dimension(joint_list)

        base_T_joints = RobotModel._forward_kinematics_batch(
            links, x, return_full_joint_fk=True, dtype=np.float64,
            base_to_source=base_to_source)
        base_T_joints = base_T_joints[:, 1:, :, :]  # remove the base link

        J = np.zeros((batch_size, 6, ndof), dtype=np.float64)
        x_i = 0
        for joint in joint_list:
            if joint.__class__.__name__ == "RotationalJoint":
                axis = np.tile(joint.axis, (batch_size, 1, 1))
                J[:, :3, x_i] = np.matmul(base_T_joints[:, x_i, :3, :3], axis.reshape(batch_size, 3, 1))[:, :, 0]
                d = base_T_joints[:, -1, :3, 3] - base_T_joints[:, x_i, :3, 3]
                world_axis = np.matmul(base_T_joints[:, x_i, :3, :3], axis.reshape(batch_size, 3, 1))[:, :, 0]
                J[:, 3:6, x_i] = np.cross(world_axis, d, axis=1)
                x_i += 1
            elif joint.__class__.__name__ == "LinearJoint":
                J[:, 3, x_i], J[:, 4, x_i], J[:, 5, x_i] = joint.axis
                x_i += 1
        return J

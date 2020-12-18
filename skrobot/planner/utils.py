import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_angle
from skrobot.coordinates.math import rpy_matrix


def scipinize(fun):
    """Scipinize a function returning both f and jac

    For the detail this issue may help:
    https://github.com/scipy/scipy/issues/12692

    Parameters
    ----------
    fun: function
        function maps numpy.ndarray(n_dim,) to tuple[numpy.ndarray(m_dim,),
        numpy.ndarray(m_dim, n_dim)], where the returned tuples is
        composed of function value(vector) and the corresponding jacobian.
    Returns
    -------
    fun_scipinized : function
        function maps numpy.ndarray(n_dim,) to a value numpy.ndarray(m_dim,).
    fun_scipinized_jac : function
        function maps numpy.ndarray(n_dim,) to
        jacobian numpy.ndarray(m_dim, n_dim).
    """

    closure_member = {'jac_cache': None}

    def fun_scipinized(x):
        f, jac = fun(x)
        closure_member['jac_cache'] = jac
        return f

    def fun_scipinized_jac(x):
        return closure_member['jac_cache']
    return fun_scipinized, fun_scipinized_jac


def set_robot_config(robot_model, joint_list, av, with_base=False):
    """A utility function for setting robot state

    Parameters
    ----------
    robot_model : skrobot.model.CascadedLink
        robot model
    joint_list : list[skrobot.model.Joint]
        joint list to be set
    av : numpy.ndarray[float](n_dof,)
        angle vector which has n_dof dims.
    with_base : bool
        If `with_base=False`, `n_dof` is the number of joints `n_joint`,
        but if `with_base=True`, `n_dof = n_joint + 3`.
    """

    if with_base:
        assert len(joint_list) + 3 == len(av)
    else:
        assert len(joint_list) == len(av)

    if with_base:
        av_joint, av_base = av[:-3], av[-3:]
        x, y, theta = av_base
        co = Coordinates(pos=[x, y, 0.0], rot=rpy_matrix(theta, 0.0, 0.0))
        robot_model.newcoords(co)
    else:
        av_joint = av

    for joint, angle in zip(joint_list, av_joint):
        joint.joint_angle(angle)


def get_robot_config(robot_model, joint_list, with_base=False):
    """A utility function for getting robot state

    Parameters
    ----------
    robot_model : skrobot.model.CascadedLink
        robot model
    joint_list : list[Joint]
        joint list of which you want to know the angles
    with_base : bool
        If set to `True`, base position is also computed.

    Returns
    -------
    av_whole (or av_joint) : numpy.ndarray(n_dof,)
        angle vector. If `with_base=False`, `n_dof` is the number of
        joints `n_joint`, but if `with_base=True`, `n_dof = n_joint + 3`.
    """
    av_joint = np.array([j.joint_angle() for j in joint_list])
    if with_base:
        x, y, _ = robot_model.translation
        rpy = rpy_angle(robot_model.rotation)[0]
        theta = rpy[0]
        av_whole = np.hstack((av_joint, [x, y, theta]))
        return av_whole
    else:
        return av_joint


def forward_kinematics_multi(robot_model,
                             joint_list,
                             av,
                             move_target_list,
                             with_rot,
                             with_base,
                             with_jacobian):
    """Compute fk for multiple feature points

    Parameters
    ----------
    robot_model : skrobot.model.CascadedLink
        robot model.
    joint_list : list[skrobot.model.Joint]
        joint to be controlled
    av : numpy.ndarray(n_dof,)
        angle vector.
    move_target_list : list[skrobot.coordinates.CascadedCoords]
        the list has `n_feature` elements. Each element is the
        coordinate of the features points.
    with_rot : bool
        If set to `True`, 7(3 + 4) dim pose-fk is also computed.
        Otherwise, 3 dim point-fk is computed.
    with_base : bool
        If `with_base=False`, `n_dof` is the number of joints `n_joint`,
        but if `with_base=True`, `n_dof = n_joint + 3`.

    Returns
    -------
    pose_arr : numpy.ndarray(n_feature, dim_pose)
        array of pose of each feature points.
        `dim_pose=7' if `with_rot=True`. Otherwise, `dim_pose=3`.
    jac_arr : numpy.ndarray(n_feature, dim_pose, n_dof)
        array of jacobian of each feature points.

    """

    set_robot_config(robot_model, joint_list, av, with_base)
    n_feature = len(move_target_list)
    n_dof = len(av)
    dim_pose = 3 + (4 if with_rot else 0)

    pose_arr = np.zeros((n_feature, dim_pose))
    if with_jacobian:
        jac_arr = np.zeros((n_feature, dim_pose, n_dof))
    else:
        jac_arr = None

    for i in range(n_feature):
        pose, jac = _forward_kinematics(
            robot_model, joint_list, move_target_list[i],
            with_rot, with_base, with_jacobian)
        pose_arr[i, :] = pose
        if with_jacobian:
            jac_arr[i, :, :] = jac
    return pose_arr, jac_arr


def _forward_kinematics(robot_model,
                        joint_list,
                        move_target,
                        with_rot,
                        with_base,
                        with_jacobian):

    link_list = [joint.child_link for joint in joint_list]

    ef_pos_wrt_world = move_target.worldpos()
    ef_quat_wrt_world = move_target.worldcoords().quaternion
    world_coordinate = CascadedCoords()

    def quaternion_kinematic_matrix(q):
        # dq/dt = 0.5 * mat * omega
        q1, q2, q3, q4 = q
        mat = np.array([
            [-q2, -q3, -q4],
            [q1, q4, -q3],
            [-q4, q1, q2],
            [q3, -q2, q1]])
        return mat * 0.5

    def compute_jacobian_wrt_world():
        J_joint = robot_model.calc_jacobian_from_link_list(
            [move_target],
            link_list,
            transform_coords=world_coordinate,
            rotation_axis=with_rot)

        if with_rot:
            kine_mat = quaternion_kinematic_matrix(ef_quat_wrt_world)
            J_joint_rot_geometric = J_joint[3:, :]  # "geometric" jacobian
            J_joint_quat = kine_mat.dot(J_joint_rot_geometric)
            J_joint = np.vstack((J_joint[:3, :], J_joint_quat))

        if with_base:  # cat base jacobian if base is considered
            # please follow computation carefully
            base_pos_wrt_world = robot_model.worldpos()
            ef_pos_wrt_world = move_target.worldpos()
            ef_pos_wrt_base = ef_pos_wrt_world - base_pos_wrt_world
            x, y = ef_pos_wrt_base[0], ef_pos_wrt_base[1]
            J_base_pos = np.array([[1, 0, -y], [0, 1, x], [0, 0, 0]])

            if with_rot:
                J_base_quat_xy = np.zeros((4, 2))
                rot_axis = np.array([0, 0, 1.0])
                J_base_quat_theta = kine_mat.dot(rot_axis).reshape(4, 1)
                J_base_quat = np.hstack((J_base_quat_xy, J_base_quat_theta))
                J_base = np.vstack((J_base_pos, J_base_quat))
            else:
                J_base = J_base_pos
            J_whole = np.hstack((J_joint, J_base))
        else:
            J_whole = J_joint
        return J_whole

    pose = np.hstack((ef_pos_wrt_world, ef_quat_wrt_world)) if with_rot \
        else ef_pos_wrt_world

    if with_jacobian:
        J = compute_jacobian_wrt_world()
        return pose, J
    else:
        return pose, None

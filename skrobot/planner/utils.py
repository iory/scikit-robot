import numpy as np
from skrobot.coordinates import CascadedCoords, Coordinates
from skrobot.coordinates.math import rpy_matrix, rpy_angle

def set_robot_state(robot_model, joint_list, av, base_also=False):
    if base_also:
        av_joint, av_base = av[:-3], av[-3:] 
        x, y, theta = av_base
        co = Coordinates(pos = [x, y, 0.0], rot=rpy_matrix(theta, 0.0, 0.0))
        robot_model.newcoords(co)
    else:
        av_joint = av

    for joint, angle in zip(joint_list, av_joint):
        joint.joint_angle(angle)

def get_robot_state(robot_model, joint_list, base_also=False):
    av_joint = np.array([j.joint_angle() for j in joint_list])
    if base_also:
        x, y, _ = robot_model.translation
        rpy = rpy_angle(robot_model.rotation)[0]
        theta = rpy[0]
        av_whole = np.hstack((av_joint, [x, y, theta]))
        return av_whole
    else:
        return av_joint

def forward_kinematics(robot_model, link_list, av, move_target, rot_also, base_also, with_jacobian=True):
    joint_list = [link.joint for link in link_list]
    set_robot_state(robot_model, joint_list, av, base_also)
    ef_pos_wrt_world = move_target.worldpos()
    ef_quat_wrt_world = move_target.worldcoords().quaternion
    world_coordinate = CascadedCoords()

    def quaternion_kinematic_matrix(q):
        # dq/dt = 0.5 * mat * omega 
        q1, q2, q3, q4 = q
        mat = np.array([
            [-q2, -q3, -q4], [q1, q4, -q3], [-q4, q1, q2], [q3, -q2, q1],
            ])
        return mat * 0.5

    def compute_jacobian_wrt_world():
        J_joint = robot_model.calc_jacobian_from_link_list(
                [move_target], link_list,
                transform_coords=world_coordinate,
                rotation_axis=rot_also)
        if rot_also:
            kine_mat = quaternion_kinematic_matrix(ef_quat_wrt_world)
            J_joint_rot_geometric = J_joint[3:, :] # geometric jacobian
            J_joint_quat = kine_mat.dot(J_joint_rot_geometric)
            J_joint = np.vstack((J_joint[:3, :], J_joint_quat))

        if base_also: # cat base jacobian if base is considered
            # please follow computation carefully
            base_pos_wrt_world = robot_model.worldpos()
            ef_pos_wrt_world = move_target.worldpos()
            ef_pos_wrt_base = ef_pos_wrt_world - base_pos_wrt_world
            x, y = ef_pos_wrt_base[0], ef_pos_wrt_world[1]
            J_base_pos = np.array([[1, 0, -y], [0, 1, x], [0, 0, 0]])

            if rot_also:
                J_base_quat_xy = np.zeros((4, 2))
                rot_axis = np.array([0, 0, 1.0])
                J_base_quat_theta = kine_mat.dot(rot_axis).reshape(4, 1)
                J_base_quat = np.hstack(
                        (J_base_quat_xy, J_base_quat_theta))
                J_base = np.vstack((J_base_pos, J_base_quat))
            else:
                J_base = J_base_pos
            J_whole = np.hstack((J_joint, J_base))
        else:
            J_whole = J_joint
        return J_whole

    pose = np.hstack((ef_pos_wrt_world, ef_quat_wrt_world)) if rot_also \
            else ef_pos_wrt_world
    if with_jacobian:
        J = compute_jacobian_wrt_world()
        return pose, J
    else:
        return pose

def scipinize(fun):
    closure_member = {'jac_cache': None}

    def fun_scipinized(x):
        f, jac = fun(x)
        closure_member['jac_cache'] = jac
        return f

    def fun_scipinized_jac(x):
        return closure_member['jac_cache']
    return fun_scipinized, fun_scipinized_jac


import numpy as np
import scipy
from . import utils

def inverse_kinematics_slsqp(self, 
                             target_coords,
                             link_list,
                             move_target,
                             coll_cascaded_coords_list=[],
                             signed_distance_function=None,
                             rot_also=True,
                             base_also=False
                             ):
    joint_list = [link.joint for link in link_list]
    av_init = utils.get_robot_state(self, joint_list, base_also)
    pos_target = target_coords.worldpos()
    quat_target = target_coords.worldcoords().quaternion if rot_also else None

    joint_limits = [[j.min_angle, j.max_angle] for j in joint_list]
    if base_also:
        joint_limits += [[-np.inf, np.inf]]*3

    def endcoord_forward_kinematics(av):
        return utils.forward_kinematics(self, link_list, av, move_target, rot_also, base_also)

    n_feature = len(coll_cascaded_coords_list)
    if n_feature > 0 and signed_distance_function is not None:
        def collision_fk(av_seq):
            points, jacobs = [], [] # TODO duplicate
            for av in av_seq:
                for collision_coords in coll_cascaded_coords_list:
                    rot_also = False # rotation is nothing to do with point collision
                    p, J = utils.forward_kinematics(self, link_list, av, collision_coords, 
                            rot_also=rot_also, base_also=base_also) 
                    points.append(p)
                    jacobs.append(J)
            return np.vstack(points), np.vstack(jacobs)

        def collision_ineq_fun(av):
            av_trajectory = av.reshape(1, -1) 
            return utils.sdf_collision_inequality_function(
                    av_trajectory, collision_fk, signed_distance_function, n_feature)
    else:
        collision_ineq_fun = None

    res = inverse_kinematics_slsqp_common(
            av_init,
            endcoord_forward_kinematics, 
            joint_limits,
            pos_target, 
            quat_target,
            collision_ineq_fun)
    av_solved = res.x
    utils.set_robot_state(self, joint_list, av_solved, base_also)
    return res

def inverse_kinematics_slsqp_common(av_init, 
        endeffector_fk, 
        joint_limits, 
        pos_target, 
        quat_target=None,
        collision_ineq_fun=None
        ):

        def fun_objective(av):
            if quat_target is None:
                position, jac = endeffector_fk(av)
                diff = position - pos_target
                cost = np.linalg.norm(diff) ** 2
                cost_grad = 2 * diff.dot(jac)
            else:
                pose, jac = endeffector_fk(av)
                position, rot = pose[:3], pose[3:]
                pos_diff = position - pos_target
                cost_position = np.linalg.norm(position - pos_target) ** 2
                cost_position_grad = 2 * pos_diff.dot(jac[:3, :])

            # see below for distnace metric for quaternion (ah.. pep8)
            #https://math.stackexchange.com/questions/90081/quaternion-distance
                inpro = np.sum(rot * quat_target)
                cost_rotation = 1 - inpro ** 2
                cost_rotation_grad = - 2 * inpro * quat_target.dot(jac[3:, :])

                cost = cost_position + cost_rotation
                cost_grad = cost_position_grad + cost_rotation_grad
            return cost, cost_grad

        f, jac = utils.scipinize(fun_objective)
        if collision_ineq_fun is None:
            constraints = []
        else:
            ineq_const_scipy, ineq_const_jac_scipy = utils.scipinize(collision_ineq_fun)
            ineq_dict = {'type': 'ineq', 'fun': ineq_const_scipy,
                         'jac': ineq_const_jac_scipy}
            constraints = [ineq_dict]

        res = scipy.optimize.minimize(
                f, av_init, method='SLSQP', jac=jac, bounds=joint_limits,
                options={'ftol': 1e-4, 'disp': False}, 
                constraints=constraints)

        return res

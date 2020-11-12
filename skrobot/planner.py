import scipy
import numpy as np
import copy

def plan_trajectory(self, 
        target_coords, 
        n_wp,
        link_list, 
        end_effector_cascaded_coords,
        coll_cascaded_coords_list,
        signed_distance_function,
        weights=None,
        initial_trajectory=None):
    """ A skrobot interface to GradBasedPlannerCommon below"""

    # common stuff
    joint_list = [link.joint for link in link_list]
    joint_limits = [[j.min_angle, j.max_angle] for j in joint_list]
    set_joint_angles = lambda av: [j.joint_angle(a) for j, a in zip(joint_list, av)]
    get_joint_angles = lambda : np.array([j.joint_angle() for j in joint_list])

    # create initial solution for the optimization problem
    if initial_trajectory is None:
        av_init = get_joint_angles()
        res = self.inverse_kinematics(
            target_coords, 
            link_list=link_list, 
            move_target= end_effector_cascaded_coords, 
            rotation_axis=False)
        av_target = get_joint_angles()

        assert (res is not False), "IK to the target coords isn't solvable. \
                You can directry pass initial_trajectory instead."

        regular_interval = (av_target - av_init)/(n_wp - 1)
        initial_trajectory = np.array([av_init + i * regular_interval for i in range(n_wp)])

    # define forward kinematics functions
    def compute_jacobian_wrt_baselink(av0, move_target, rot_also=False):
        set_joint_angles(av0)
        base_link = self.link_list[0]
        J = self.calc_jacobian_from_link_list([move_target], link_list, 
                transform_coords=base_link,
                rotation_axis=rot_also)
        return J

    def collision_forward_kinematics(av_seq):
        points = []
        jacobs = []
        for av in av_seq:
            for collision_coords in coll_cascaded_coords_list:
                J = compute_jacobian_wrt_baselink(av, collision_coords)
                pos = collision_coords.worldpos()
                points.append(pos)
                jacobs.append(J)
        return np.vstack(points), np.vstack(jacobs)

    def endcoord_forward_kinematics(av_seq):
        points = []
        jacobs = []
        for av in av_seq:
            J = compute_jacobian_wrt_baselink(av, end_effector_cascaded_coords)
            pos = end_effector_cascaded_coords.worldpos()
            points.append(pos)
            jacobs.append(J)
        return np.vstack(points), np.vstack(jacobs)

    # solve!
    opt = GradBasedPlannerCommon(initial_trajectory, 
            target_coords.worldpos(), 
            collision_forward_kinematics, 
            endcoord_forward_kinematics, 
            joint_limits,
            signed_distance_function,
            weights=weights)
    optimal_trajectory = opt.solve()

    return optimal_trajectory

def construct_smoothcost_fullmat(n_dof, n_wp, weights = None): # A, b and c 

    def construct_smoothcost_mat(n_wp):
        acc_block = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) 
        vel_block = np.array([[1, -1], [-1, 1]])
        A = np.zeros((n_wp, n_wp))
        for i in [1 + i for i in range(n_wp - 2)]:
            A[i-1:i+2, i-1:i+2] += acc_block # i-1 to i+1 (3x3)
            #A[i-1:i+1, i-1:i+1] += vel_block * 2.0
        return A

    w_mat = np.eye(n_dof) if weights is None else np.diag(weights)
    Amat = construct_smoothcost_mat(n_wp)
    Afullmat = np.kron(Amat, w_mat**2)
    return Afullmat

def scipinize(fun):
    closure_member = {'jac_cache': None}
    def fun_scipinized(x):
        f, jac = fun(x)
        closure_member['jac_cache'] = jac
        return f

    fun_scipinized_jac = lambda x: closure_member['jac_cache']
    return fun_scipinized, fun_scipinized_jac

class GradBasedPlannerCommon:
    def __init__(self, av_seq_init, pos_target, collision_fk, endcoord_fk, joint_limit, sdf, sdf_margin=0.08, weights=None):
        self.av_seq_init = av_seq_init
        self.n_features = 2
        self.collision_fk = collision_fk
        self.endcoord_fk = endcoord_fk
        self.sdf = lambda X: sdf(X) - sdf_margin
        self.n_wp, self.n_dof = av_seq_init.shape
        self.pos_target = pos_target
        self.joint_limit  = joint_limit
        #w = [0.5, 0.5, 0.3, 0.1, 0.1, 0.1, 0.1]
        self.A = construct_smoothcost_fullmat(self.n_dof, self.n_wp, weights=weights) 

    def fun_objective(self, x):
        Q = x.reshape(self.n_wp, self.n_dof)
        qe = Q[-1]

        P_ef, J_ef = self.endcoord_fk([qe])
        diff = P_ef[0] - self.pos_target
        cost_terminal = np.linalg.norm(diff) ** 2
        grad_cost_terminal = 2 * diff.dot(J_ef)

        f = (0.5 * self.A.dot(x).dot(x)).item() + cost_terminal
        grad = self.A.dot(x) 
        grad[-self.n_dof:] += grad_cost_terminal # cause only final av affects the terminal cost
        return f, grad

    def fun_ineq(self, xi):
        av_seq = xi.reshape(self.n_wp, self.n_dof)
        P_link, J_link = self.collision_fk(av_seq)

        sdf_grads = np.zeros(P_link.shape)
        F_link_cost0 = self.sdf(np.array(P_link)) 
        eps = 1e-7
        for i in range(3):
            P_link_ = copy.copy(P_link);
            P_link_[:, i] += eps;
            F_link_cost1 = self.sdf(np.array(P_link_))  
            sdf_grads[:, i] = (F_link_cost1 - F_link_cost0)/eps;

        sdf_grads = sdf_grads.reshape(self.n_wp * self.n_features, 1, 3)
        J_link = J_link.reshape(self.n_wp * self.n_features, 3, self.n_dof)
        J_link_list = np.matmul(sdf_grads, J_link)
        J_link_block = J_link_list.reshape(self.n_wp, self.n_features, self.n_dof) #(n_wp, n_features, n_dof)
        J_link_full = scipy.linalg.block_diag(*list(J_link_block)) #(n_wp * n_features, n_wp * n_dof)
        F_cost_full, J_cost_full = F_link_cost0, J_link_full
        return F_cost_full, J_cost_full

    def fun_eq(self, xi):
        # terminal constraint
        Q = xi.reshape(self.n_wp, self.n_dof)
        qs = Q[0]
        q_start = self.av_seq_init[0]
        f = np.hstack((q_start - qs))
        grad_ = np.zeros((self.n_dof * 1, self.n_dof * self.n_wp))
        grad_[0:self.n_dof, 0:self.n_dof] = - np.eye(self.n_dof) 
        return f, grad_

    def solve(self):
        eq_const_scipy, eq_const_jac_scipy = scipinize(self.fun_eq)
        eq_dict = {'type': 'eq', 'fun': eq_const_scipy, 'jac': eq_const_jac_scipy}
        ineq_const_scipy, ineq_const_jac_scipy = scipinize(self.fun_ineq)
        ineq_dict = {'type': 'ineq', 'fun': ineq_const_scipy, 'jac': ineq_const_jac_scipy}
        f, jac = scipinize(self.fun_objective)

        tmp = np.array(self.joint_limit)
        lower_limit = tmp[:, 0]
        uppre_limit = tmp[:, 1]

        bounds = list(zip(lower_limit, uppre_limit)) * self.n_wp

        xi_init = self.av_seq_init.reshape((self.n_dof * self.n_wp, ))
        res = scipy.optimize.minimize(f, xi_init, method='SLSQP', jac=jac,
                bounds = bounds,
                constraints=[eq_dict, ineq_dict], options={'ftol': 1e-4, 'disp': False})
        traj_opt = res.x.reshape(self.n_wp, self.n_dof)
        return traj_opt

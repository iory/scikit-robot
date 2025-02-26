import numpy as np
import scipy

from skrobot.planner.utils import scipinize
from skrobot.pycompat import lru_cache


def sqp_plan_trajectory(collision_checker,
                        av_start,
                        av_goal,
                        joint_list,
                        n_wp,
                        safety_margin=1e-2,
                        with_base=False,
                        weights=None,
                        initial_trajectory=None,
                        slsqp_option=None
                        ):
    """Gradient based trajectory optimization using scipy's SLSQP.

    Collision constraint is considered in an inequality constraints.
    Terminal constraint (start and end) is considered as an
    equality constraint.

    Parameters
    ----------
    av_start : numpy.ndarray(n_dof,)
        joint angle vector at start point
    joint_list : list[skrobot.model.Link]
        link list to be controlled (similar to inverse_kinematics function)
    n_wp : int
        number of waypoints
    safety_margin : float
        safety margin in collision checking
    with_base: bool
        If `with_base=False`, `n_dof` is the number of joints `n_joint`,
        but if `with_base=True`, `n_dof = len(joint_list) + 3`.
    weights : numpy.ndarray(n_dof,) or  None
        cost to move of each joint. For example,
        if you set weights=numpy.ndarray([1.0, 0.1, 0.1]) for a
        3 DOF manipulator, moving the first joint is with
        high cost compared to others. If set to `None` it's automatically
        determined.
    initial_trajectory : numpy.ndarray(n_wp, n_dof) or None
        initial solution in the trajectory optimization specified by a
        angle vector sequence. If None, initial trajectory is
        automatically generated.
    slsqp_option: dict or None
        option of slsqp. Please see `options` in
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
        for the detail. If set to `None`, a default values is used.
    Returns
    ------------
    planned_trajectory : numpy.ndarray(n_wp, n_dof)
        planned trajectory.
    """

    # common stuff
    joint_limit_list = [[j.min_angle, j.max_angle] for j in joint_list]
    if with_base:
        joint_limit_list += [[-np.inf, np.inf]] * 3

    # determine default weight
    if weights is None:
        weights = [1.0] * len(joint_list)
        if with_base:
            weights += [3.0] * 3  # base should be difficult to move
    weights = tuple(weights)  # to use cache

    # create initial solution for the optimization problem
    if initial_trajectory is None:
        regular_interval = (av_goal - av_start) / (n_wp - 1)
        initial_trajectory = np.array(
            [av_start + i * regular_interval for i in range(n_wp)])

    def collision_ineq_fun(av_seq):
        with_jacobian = True
        sd_vals, sd_val_jac = collision_checker.compute_batch_sd_vals(
            joint_list, av_seq,
            with_base=with_base, with_jacobian=with_jacobian)
        sd_vals_margined = sd_vals - safety_margin
        return sd_vals_margined, sd_val_jac

    optimal_trajectory = _sqp_based_trajectory_optimization(
        initial_trajectory,
        collision_ineq_fun,
        joint_limit_list,
        weights,
        slsqp_option)
    return optimal_trajectory


def _sqp_based_trajectory_optimization(
        av_seq_init,
        collision_ineq_fun,
        joint_limit_list,
        weights,
        slsqp_option=None):

    if slsqp_option is None:
        slsqp_option = {'ftol': 1e-4, 'disp': True, 'maxiter': 100}
    n_wp, n_dof = av_seq_init.shape
    A = construct_smoothcost_fullmat(n_wp, n_dof, weights=weights)

    def fun_objective(x):
        f = (0.5 * A.dot(x).dot(x)).item() / n_wp
        grad = A.dot(x) / n_wp
        return f, grad

    def fun_ineq(xi):
        av_seq = xi.reshape(n_wp, n_dof)
        return collision_ineq_fun(av_seq)

    def fun_eq(xi):
        # terminal constraint
        Q = xi.reshape(n_wp, n_dof)
        q_start = av_seq_init[0]
        q_end = av_seq_init[-1]
        f = np.hstack((q_start - Q[0], q_end - Q[-1]))
        grad_ = np.zeros((n_dof * 2, n_dof * n_wp))
        grad_[:n_dof, :n_dof] = - np.eye(n_dof)
        grad_[-n_dof:, -n_dof:] = - np.eye(n_dof)
        return f, grad_

    eq_const_scipy, eq_const_jac_scipy = scipinize(fun_eq)
    eq_dict = {'type': 'eq', 'fun': eq_const_scipy,
               'jac': eq_const_jac_scipy}
    ineq_const_scipy, ineq_const_jac_scipy = scipinize(fun_ineq)
    ineq_dict = {'type': 'ineq', 'fun': ineq_const_scipy,
                 'jac': ineq_const_jac_scipy}
    f, jac = scipinize(fun_objective)

    tmp = np.array(joint_limit_list)
    lower_limit, uppre_limit = tmp[:, 0], tmp[:, 1]

    bounds = list(zip(lower_limit, uppre_limit)) * n_wp

    xi_init = av_seq_init.reshape((n_dof * n_wp, ))
    res = scipy.optimize.minimize(
        f, xi_init, method='SLSQP', jac=jac,
        bounds=bounds,
        constraints=[eq_dict, ineq_dict],
        options=slsqp_option)
    traj_opt = res.x.reshape(n_wp, n_dof)
    return traj_opt


@lru_cache(maxsize=1000)
def construct_smoothcost_fullmat(n_wp, n_dof, weights):
    """Compute A of eq. (17) of IJRR-version (2013) of CHOMP"""

    def construct_smoothcost_mat(n_wp):
        # In CHOMP (2013), squared sum of velocity is computed.
        # In this implementation we compute squared sum of acceralation
        # if you set acc_block * 0.0, vel_block * 1.0, then the trajectory
        # cost is same as the CHOMP one.
        acc_block = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        vel_block = np.array([[1, -1], [-1, 1]])
        A_ = np.zeros((n_wp, n_wp))
        for i in [1 + i for i in range(n_wp - 2)]:
            A_[i - 1:i + 2, i - 1:i + 2] += acc_block * 1.0
            A_[i - 1:i + 1, i - 1:i + 1] += vel_block * 0.0  # do nothing
        return A_

    w_mat = np.diag(weights)
    A_ = construct_smoothcost_mat(n_wp)
    A = np.kron(A_, w_mat**2)
    return A

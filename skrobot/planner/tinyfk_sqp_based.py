import numpy as np

from skrobot.planner.sqp_based import _sqp_based_trajectory_optimization


def tinyfk_sqp_plan_trajectory(collision_checker,
                               constraint_manager,
                               initial_trajectory,
                               joint_list,
                               n_wp,
                               safety_margin=1e-2,
                               with_base=False,
                               weights=None,
                               slsqp_option=None
                               ):
    # common stuff
    joint_limit_list = [[j.min_angle, j.max_angle] for j in joint_list]
    if with_base:
        joint_limit_list += [[-np.inf, np.inf]] * 3
    n_dof = len(joint_list) + (3 if with_base else 0)

    # determine default weight
    if weights is None:
        weights = [1.0] * len(joint_list)
        if with_base:
            weights += [3.0] * 3  # base should be difficult to move
    weights = tuple(weights)  # to use cache

    joint_name_list = [j.name for j in joint_list]
    joint_ids = collision_checker.fksolver.get_joint_ids(joint_name_list)

    def collision_ineq_fun(av_seq):
        with_jacobian = True
        sd_vals, sd_val_jac = collision_checker._compute_batch_sd_vals(
            joint_ids, av_seq,
            with_base=with_base, with_jacobian=with_jacobian)
        sd_vals_margined = sd_vals - safety_margin
        return sd_vals_margined, sd_val_jac

    optimal_trajectory = _sqp_based_trajectory_optimization(
        initial_trajectory,
        collision_ineq_fun,
        constraint_manager.gen_combined_constraint_func(),
        joint_limit_list,
        weights,
        slsqp_option)
    return optimal_trajectory

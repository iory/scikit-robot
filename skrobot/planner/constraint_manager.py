import uuid
import numpy as np

# TODO check added eq_configuration is valid by pre-solving collision checking

class EqualityConstraint(object):
    def __init__(self, n_wp, n_dof, idx_wp, name):
        assert (idx_wp in range(n_wp)), "index {0} is out fo range".format(idx_wp)
        self.n_wp = n_wp
        self.n_dof = n_dof
        self.idx_wp = idx_wp
        self.name = name

    def _check_func(self, func):
        # TODO insert jacobian test utils here
        error_report_prefix = "ill-formed function {} is detected. ".format(self.name)
        av_seq_dummy = np.zeros((self.n_wp, self.n_dof))
        try:
            f, jac = func(av_seq_dummy)
        except:
            raise Exception(
                    error_report_prefix + 
                    "check input dimension of the function")
        assert f.ndim == 1, "f must be one dim"
        dim_constraint = len(f)
        dof_all = self.n_wp * self.n_dof
        assert jac.shape == (dim_constraint, dof_all), error_report_prefix \
                + "shape of jac is strainge. Desired: {0}, Copmuted {1}".format((dim_constraint, dof_all), jac.shape)

class ConfigurationConstraint(EqualityConstraint):
    def __init__(self, n_wp, n_dof, idx_wp, av_desired, name=None):
        if name is None:
            name = 'eq_config_const_{}'.format(str(uuid.uuid1()).replace('-', '_'))
        super(ConfigurationConstraint, self).__init__(n_wp, n_dof, idx_wp, name)
        self.av_desired = av_desired
        self.rank = n_dof

    def gen_func(self):
        n_dof, n_wp = self.n_dof, self.n_wp
        n_dof_all = n_dof * n_wp
        def func(av_seq):
            f = av_seq[self.idx_wp] - self.av_desired
            grad = np.zeros((self.rank, n_dof_all)) 
            grad[:, n_dof*self.idx_wp:n_dof*(self.idx_wp+1)] = np.eye(self.rank)
            return f, grad
        self._check_func(func)
        return func

    def satisfying_angle_vector(self, av_init=None):
        return self.av_desired

class PoseConstraint(EqualityConstraint):
    def __init__(self, n_wp, n_dof, idx_wp, coords_name, pose_desired, 
            fksolver, joint_ids, with_base,
            name=None):
        # here pose order is [x, y, z, r, p, y]
        if name is None:
            name = 'eq_pose_const_{}'.format(str(uuid.uuid1()).replace('-', '_'))
        super(PoseConstraint, self).__init__(n_wp, n_dof, idx_wp, name)

        self.coords_name = coords_name
        self.pose_desired = pose_desired
        self.rank = len(pose_desired)

        self.with_rot = (self.rank == 6)

        self.joint_ids = joint_ids
        self.fksolver = fksolver
        self.with_base = with_base

    def gen_func(self):
        n_dof_all = self.n_dof * self.n_wp

        coords_ids = self.fksolver.get_link_ids([self.coords_name])
        def func(av_seq):
            J_whole = np.zeros((self.rank, n_dof_all))
            P, J = self.fksolver.solve_forward_kinematics(
                    [av_seq[self.idx_wp]], coords_ids, self.joint_ids,
                    with_rot=self.with_rot, with_base=self.with_base, with_jacobian=True) 
            J_whole[:, self.n_dof*self.idx_wp:self.n_dof*(self.idx_wp+1)] = J
            return (P - self.pose_desired).flatten(), J_whole
        self._check_func(func)
        return func

    def satisfying_angle_vector(self, av_init=None, option=None):
        if option is None:
            option = {"maxitr": 200, "ftol": 1e-4, "sr_weight":1.0}
        coords_id = self.fksolver.get_link_ids([self.coords_name])[0]
        if av_init is None:
            n_dof = len(self.joint_ids) + (3 if self.with_base else 0)
            av_init = np.zeros(n_dof)
        av_solved = self.fksolver.solve_inverse_kinematics(self.pose_desired, av_init, coords_id,
                self.joint_ids, self.with_rot, self.with_base, option=option, ignore_fail=True)
        return av_solved

# give a problem specification
class ConstraintManager(object):
    def __init__(self, n_wp, joint_names, fksolver, with_base): 
        # must be with_base=True now
        self.n_wp = n_wp
        n_dof = len(joint_names) + (3 if with_base else 0)
        self.n_dof = n_dof
        self.constraint_table = {}

        self.joint_ids = fksolver.get_joint_ids(joint_names)
        self.fksolver = fksolver
        self.with_base = with_base

    def add_eq_configuration(self, idx_wp, av_desired, force=False):
        constraint = ConfigurationConstraint(self.n_wp, self.n_dof, idx_wp, av_desired)
        self._add_constraint(idx_wp, constraint, force)

    def add_pose_constraint(self, idx_wp, coords_name, pose_desired, force=False):
        constraint = PoseConstraint(self.n_wp, self.n_dof, idx_wp,
                coords_name, pose_desired,
                self.fksolver, self.joint_ids, self.with_base)
        self._add_constraint(idx_wp, constraint, force)

    def _add_constraint(self, idx_wp, constraint, force):
        is_already_exist = idx_wp in self.constraint_table.keys()
        if is_already_exist and (not force):
            raise Exception("to overwrite the constraint, please set force=True")
        self.constraint_table[idx_wp] = constraint

    def gen_combined_constraint_func(self):
        has_initial_and_terminal_const = 0 in self.constraint_table.keys() and (self.n_wp-1) in self.constraint_table.keys(), "please set initial and terminal constraint"
        assert has_initial_and_terminal_const
        # correct all funcs
        func_list = []
        for constraint in self.constraint_table.values():
            func_list.append(constraint.gen_func())

        def func_combined(xi):
            # xi is the flattened angle vector
            av_seq = xi.reshape(self.n_wp, self.n_dof)
            f_list, jac_list = zip(*[fun(av_seq) for fun in func_list])
            return np.hstack(f_list), np.vstack(jac_list)
        return func_combined

    def gen_initial_trajectory(self, av_current=None):
        av_start = self.constraint_table[0].satisfying_angle_vector(av_init=av_current)
        av_goal = self.constraint_table[self.n_wp-1].satisfying_angle_vector(av_init=av_current)

        regular_interval = (av_goal - av_start) / (self.n_wp - 1)
        initial_trajectory = np.array(
            [av_start + i * regular_interval for i in range(self.n_wp)])
        return initial_trajectory

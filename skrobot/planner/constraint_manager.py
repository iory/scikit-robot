import uuid
import numpy as np

# TODO add class Constraint, EqualityConstraint, InequalityConstraint

# TODO add name to each function

# TODO check added eq_configuration is valid by pre-solving collision checking

class EqualityConstraint(object):
    def __init__(self, n_wp, n_dof, idx_wp, name):
        self.n_wp = n_wp
        self.n_dof = n_dof
        self.idx_wp = idx_wp
        self.name = name

    def _check_func(self, func):
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

    def gen_func(self):
        n_dof, n_wp = self.n_dof, self.n_wp
        n_dof_all = n_dof * n_wp
        rank = n_dof
        def func(av_seq):
            f = av_seq[self.idx_wp] - self.av_desired
            grad = np.zeros((rank, n_dof_all)) 
            grad[:, n_dof*self.idx_wp:n_dof*(self.idx_wp+1)] = np.eye(rank)
            return f, grad
        self._check_func(func)
        return func

"""
class PoseConstraint(EqualityConstraint):
    def __init__(self, n_wp, n_dof, idx_wp, pose_desired, name=None):
        if name is None:
            name = 'eq_pose_const_{}'.format(str(uuid.uuid1()).replace('-', '_'))
        super(ConfigurationConstraint, self).__init__(n_wp, n_dof, name)

        rank = len(pose_desired)
        n_dof_all = self.n_dof * self.n_wp

        def func(av_seq):
            f = av_seq[idx_wp] - av_desired
            grad = np.zeros((rank, n_dof_all)) 
            grad[:, n_dof*idx_wp:n_dof*(idx_wp+1)] = np.eye(rank)
            return f, grad
        self._check_func(func)

        self.func = func
        self.name = name
"""

# give a problem specification
class ConstraintManager(object):
    def __init__(self, n_wp, joint_names, fksolver, with_base): 
        # must be with_base=True now
        self.n_wp = n_wp
        n_dof = len(joint_names) + (3 if with_base else 0)
        self.n_dof = n_dof
        self.constraint_list = []

        self.joint_ids = fksolver.get_joint_ids(joint_names)
        self.fksolver = fksolver
        self.with_base = with_base

    def add_eq_configuration(self, idx_wp, av_desired):
        assert (idx_wp in range(self.n_wp)), "index {0} is out fo range".format(idx_wp)
        constraint = ConfigurationConstraint(self.n_wp, self.n_dof, idx_wp, av_desired)
        self.constraint_list.append(constraint)

    """
    def add_eq_pose(self, idx_wp, coords_name, pose_desired):
        assert len(pose_desired) == 3, "currently only position is supported"
        #assert len(pose_desired) == 7, "quaternion based pose"
        #position, rotation = pose_desired[:3], pose_desired[3:]
        rank = len(pose_desired)
        n_dof_all = self.n_dof * self.n_wp

        fksolver = self.fksolver

        ## TODO implement this in c++ side
        def quaternion_kinematic_matrix(q): # same as the one in utils
            # dq/dt = 0.5 * mat * omega
            q2, q2, q3, q4 = q
            mat = np.array([
                [-q2, -q3, -q4],
                [q1, q4, -q3],
                [-q4, q1, q2],
                [q3, -q2, q1]])
            return mat * 0.5 

        target_coords_ids = fksolver.get_link_ids([coords_name])
        def func(av_seq):
            with_rot = False
            with_jacobian = True
            J_whole = np.zeros((rank, n_dof_all))
            P, J = fksolver.solve_forward_kinematics(
                    [av_seq[idx_wp]], target_coords_ids, self.joint_ids,
                    with_rot, self.with_base, with_jacobian) 
            J_whole[:, self.n_dof*idx_wp:self.n_dof*(idx_wp+1)] = J
            return (P - pose_desired).flatten(), J_whole
        self.check_func(func)
        self.c_eq_func_list_list[idx_wp].append((func, "eq_pose"))
    """

    def gen_combined_constraint_func(self):
        # correct all funcs
        func_list = []
        for constraint in self.constraint_list:
            func_list.append(constraint.gen_func())

        def func_combined(xi):
            # xi is the flattened angle vector
            av_seq = xi.reshape(self.n_wp, self.n_dof)
            f_list, jac_list = zip(*[fun(av_seq) for fun in func_list])
            return np.hstack(f_list), np.vstack(jac_list)
        return func_combined

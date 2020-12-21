import numpy as np

# TODO add class Constraint, EqualityConstraint, InequalityConstraint
# TODO add name to each function

# give a problem specification
class ConstraintManager(object):
    def __init__(self, n_wp, n_dof): 
        # must be with_base=True now
        self.n_wp = n_wp
        self.n_dof = n_dof
        self.c_eq_rank_list = [0 for _ in range(n_wp)]
        self.c_eq_func_list_list = [[] for _ in range(n_wp)]

    def add_eq_configuration(self, idx, av_desired):
        assert (idx in range(self.n_wp)), "index {0} is out fo range".format(idx)
        n_dof_all = self.n_dof * self.n_wp
        rank = self.n_dof
        self.c_eq_rank_list[idx] += rank
        def func(av_seq):
            f = av_seq[idx] - av_desired
            grad = np.zeros((rank, n_dof_all)) 
            grad[:, self.n_dof*idx:self.n_dof*(idx+1)] = np.eye(rank)
            return f, grad
        self.check_func(func)
        self.c_eq_func_list_list[idx].append(func)

    def gen_combined_eq_constraint(self):
        return self._gen_func_combined(self.c_eq_func_list_list)

    def check_func(self, func):
        av_seq_dummy = np.zeros((self.n_wp, self.n_dof))
        try:
            f, jac = func(av_seq_dummy)
        except:
            raise Exception("check input dimension of the function")
        dim_constraint = len(f)
        dof_all = self.n_wp * self.n_dof
        assert jac.shape == (dim_constraint, dof_all), "shape of jac is strainge"

    def _gen_func_combined(self, func_list_list):
        # correct all funcs
        flattened = []
        for func_list in func_list_list:
            for func in func_list:
                flattened.append(func)

        def func_combined(xi):
            # xi is the flattened angle vector
            av_seq = xi.reshape(self.n_wp, self.n_dof)
            f_list, jac_list = zip(*[fun(av_seq) for fun in flattened])
            return np.hstack(f_list), np.vstack(jac_list)
        return func_combined

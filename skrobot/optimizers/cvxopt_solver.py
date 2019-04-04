import cvxopt
from cvxopt import matrix as cvxmat
from cvxopt.solvers import qp
import numpy as np


cvxopt.solvers.options['show_progress'] = False  # disable cvxopt output


def solve_qp(P, q, G, h,
             A=None, b=None, sym_proj=False,
             solver='cvxopt'):
    if sym_proj:
        P = .5 * (P + P.T)
    cvxmat(P)
    cvxmat(q)
    cvxmat(G)
    cvxmat(h)
    args = [cvxmat(P), cvxmat(q), cvxmat(G), cvxmat(h)]
    if A is not None:
        args.extend([cvxmat(A), cvxmat(b)])
    sol = qp(*args, solver=solver)
    if not ('optimal' in sol['status']):
        raise ValueError('QP optimum not found: %s' % sol['status'])
    return np.array(sol['x']).reshape((P.shape[1],))

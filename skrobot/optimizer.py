#!/usr/bin/env python
# -*- coding: utf-8 -*-

from skrobot.optimizers.cvxopt_solver import solve_qp as cvxopt_solve_qp
from skrobot.optimizers.quadprog_solver import solve_qp as quadprog_solve_qp


def solve_qp(P, q, G, h, A=None, b=None, solver='cvxopt', sym_proj=False):
    """n Solve a Quadratic Program defined as:

    .. math::
        \\begin{eqnarray}
        \\mathrm{minimize} & & (1/2) x^T P x + q^T x \\\\
        \\mathrm{subject\\ to} & & G x \\leq h \\\\
            & & A x = b
        \\end{eqnarray}
    Parameters
    ----------
    P : array, shape=(n, n)
        Primal quadratic cost matrix.
    q : array, shape=(n,)
        Primal quadratic cost vector.
    G : array, shape=(m, n)
        Linear inequality constraint matrix.
    h : array, shape=(m,)
        Linear inequality constraint vector.
    A : array, shape=(meq, n), optional
        Linear equality constraint matrix.
    b : array, shape=(meq,), optional
        Linear equality constraint vector.
    solver : string, optional
        Name of the QP solver to use (default is 'quadprog').
    sym_proj : bool, optional
        Set to `True` when the `P` matrix provided is not symmetric.
    Returns
    -------
    x : array, shape=(n,)
        Optimal solution to the QP, if found.
    Raises
    ------
    ValueError
        If the QP is not feasible.
    """
    if solver == 'cvxopt':
        return cvxopt_solve_qp(P, q, G, h, A, b, sym_proj=sym_proj)
    elif solver == 'quadprog':
        return quadprog_solve_qp(P, q, G, h, A, b, sym_proj=sym_proj)
    raise ValueError('QP solver {} not supported'.format(solver))

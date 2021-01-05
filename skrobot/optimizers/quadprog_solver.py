import numpy as np
from quadprog import solve_qp as _solve_qp


def solve_qp(P, q, G, h, A=None, b=None, sym_proj=False):
    """Solve a Quadratic Program defined as:

    .. math::
        \\begin{eqnarray}
        \\mathrm{minimize} & & (1/2) x^T P x + q^T x \\\\
        \\mathrm{subject\\ to} & & G x \\leq h \\\\
            & & A x = b
        \\end{eqnarray}
    using the `quadprog <https://pypi.python.org/pypi/quadprog/>`_ QP
    solver, which implements the Goldfarb-Idnani dual algorithm
    [Goldfarb83]_.
    Parameters
    ----------
    P : array, shape=(n, n)
        Symmetric quadratic-cost matrix.
    q : array, shape=(n,)
        Quadratic-cost vector.
    G : array, shape=(m, n)
        Linear inequality matrix.
    h : array, shape=(m,)
        Linear inequality vector.
    A : array, shape=(meq, n), optional
        Linear equality matrix.
    b : array, shape=(meq,), optional
        Linear equality vector.
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
    Note
    ----
    The quadprog solver assumes `P` is symmetric. If that is not the case, set
    `sym_proj=True` to project it on its symmetric part beforehand.
    """
    if sym_proj:
        qp_G = .5 * (P + P.T)
    else:
        qp_G = P
    qp_a = -q
    if A is not None:
        qp_C = - np.vstack([A, G]).T
        qp_b = - np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = - G.T
        qp_b = - h
        meq = 0
    return _solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

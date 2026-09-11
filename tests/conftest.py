"""Session-wide checks that run before any test does.

Right now there is one, and it exists because the failure it catches is
otherwise unreadable: a machine whose BLAS returns wrong numbers produces a
misshapen robot and a test failure several layers away from the cause.
"""

import sys

import numpy as np


def _homogeneous_product_is_correct():
    """Return whether this machine multiplies a homogeneous transform right.

    ``trimesh.transform_points`` -- and so every mesh a URDF scales or
    places -- stacks a column of ones onto the points and multiplies the
    (4, 4) transform by the resulting (4, n).  Some of GitHub's runners, an
    Intel Xeon 6973P-C that numpy's bundled OpenBLAS detects as Cooperlake,
    get that product wrong for n past roughly 128: most rows come back as
    ``[10*x + 10, 0, 0]``, the homogeneous row folded into the first column
    and the other two zeroed.  A scale of 10 then yields a mesh that is not
    the original scaled by anything, and the first thing to notice is a
    signed distance field built on a grid 7.4 times too wide.

    Returns
    -------
    correct : bool
        Whether the product matches multiplying the points by the scale
        directly, which does not go through BLAS.
    """
    n = 2503
    points = np.random.default_rng(0).random((n, 3))
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag([10.0, 10.0, 10.0])
    stacked = np.column_stack((points, np.ones(n)))
    product = np.dot(matrix, stacked.T).T[:, :3]
    return np.abs(product - points * 10.0).max() <= 1e-9


def pytest_sessionstart(session):
    if _homogeneous_product_is_correct():
        return
    sys.stderr.write(
        '\nnumpy on this machine returns wrong results for the matrix '
        'product behind every mesh transform, so the models these tests '
        'build are not the shapes they are meant to be and the failures '
        'would not point here.\n\n'
        'Set a working kernel before starting python -- OpenBLAS reads this '
        'when it loads, so it cannot be done from inside the session:\n\n'
        '    OPENBLAS_CORETYPE=HASWELL pytest ...\n\n')
    raise SystemExit(1)

import numpy as np


def vector3p(vec):
    """Checks that the translation vector is valid.

    Parameters
    ----------
    vec : `np.ndarray` or `list`
        vector of 3x1 or 1x3
    """
    if not isinstance(vec, np.ndarray) or not np.issubdtype(
            vec.dtype, np.number):
        raise ValueError(
            'vector must be specified as numeric numpy array')

    t = vec.squeeze()
    if len(t.shape) != 1 or t.shape[0] != 3:
        raise ValueError(
            'vector must be specified as a 3-vector, 3x1 ndarray, '
            'or 1x3 ndarray')


def vectorp(vec):
    """Checks that the translation vector is valid.

    Parameters
    ----------
    vec : `np.ndarray` or `list`
        vector
    """
    if not isinstance(vec, np.ndarray) or not np.issubdtype(
            vec.dtype, np.number):
        raise ValueError(
            'vector must be specified as numeric numpy array')

    t = vec.squeeze()
    if not (len(t.shape) == 1 or t.shape[0] == 1):
        raise ValueError(
            'vector must be specified as a n-vector, nx1 ndarray, '
            'or 1xn ndarray')

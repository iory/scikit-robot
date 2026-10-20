"""Backend-agnostic math utilities for differentiable computation.

This module provides common mathematical operations that work with
any backend (NumPy, JAX) for use in kinematics and dynamics computations.
"""

import numpy as np


# _CROSS[i, j, k] = -epsilon_ijk, so _CROSS @ v is the cross-product
# matrix of v. One matmul builds it for a single vector or a batch; nested
# stacks would cost several times as much per call on the NumPy FK path.
_CROSS = np.zeros((3, 3, 3))
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _CROSS[_i, _j, _k] = -1.0
    _CROSS[_i, _k, _j] = 1.0


def rodrigues_rotation(backend, axis, angle):
    """Compute rotation matrix from axis-angle using Rodrigues' formula.

    R = I + sin(θ)K + (1-cos(θ))K²

    where K is the skew-symmetric matrix of the normalized axis.

    Parameters
    ----------
    backend : DifferentiableBackend or module
        Backend to use for computation (numpy or jax). The ``numpy`` and
        ``jax.numpy`` modules work as well.
    axis : array, shape (3,) or (..., 3)
        Rotation axis (will be normalized), or a batch of them.
    angle : float or array, shape () or (...)
        Rotation angle in radians, one per axis.

    Returns
    -------
    rotation : array, shape (3, 3) or (..., 3, 3)
        Rotation matrix, or one per axis.

    Examples
    --------
    >>> from skrobot.backend import get_backend
    >>> from skrobot.backend.math_utils import rodrigues_rotation
    >>> backend = get_backend('numpy')
    >>> axis = backend.array([0.0, 0.0, 1.0])
    >>> R = rodrigues_rotation(backend, axis, 0.5)
    """
    # Normalize axis
    axis = axis / (backend.sqrt((axis ** 2).sum(axis=-1, keepdims=True))
                   + 1e-10)
    K = skew_symmetric(backend, axis)

    # Rodrigues formula
    sin = backend.sin(angle)[..., None, None]
    cos = backend.cos(angle)[..., None, None]
    return backend.eye(3) + sin * K + (1.0 - cos) * (K @ K)


def skew_symmetric(backend, v):
    """Create skew-symmetric matrix from a 3D vector.

    The skew-symmetric matrix [v]_× satisfies: [v]_× @ u = v × u

    Parameters
    ----------
    backend : DifferentiableBackend or module
        Backend to use for computation. The ``numpy`` and ``jax.numpy``
        modules work as well.
    v : array, shape (3,) or (..., 3)
        Input vector, or a batch of them.

    Returns
    -------
    K : array, shape (3, 3) or (..., 3, 3)
        Skew-symmetric matrix, or one per vector.
    """
    return (_CROSS @ v[..., None, :, None])[..., 0]

"""Backend-agnostic math utilities for differentiable computation.

This module provides common mathematical operations that work with
any backend (NumPy, JAX) for use in kinematics and dynamics computations.
"""


def rodrigues_rotation(backend, axis, angle):
    """Compute rotation matrix from axis-angle using Rodrigues' formula.

    R = I + sin(θ)K + (1-cos(θ))K²

    where K is the skew-symmetric matrix of the normalized axis.

    Parameters
    ----------
    backend : DifferentiableBackend
        Backend to use for computation (numpy or jax).
    axis : array, shape (3,)
        Rotation axis (will be normalized).
    angle : float or array
        Rotation angle in radians.

    Returns
    -------
    rotation : array, shape (3, 3)
        Rotation matrix.

    Examples
    --------
    >>> from skrobot.backend import get_backend
    >>> from skrobot.backend.math_utils import rodrigues_rotation
    >>> backend = get_backend('numpy')
    >>> axis = backend.array([0.0, 0.0, 1.0])
    >>> R = rodrigues_rotation(backend, axis, 0.5)
    """
    # Normalize axis
    axis = axis / (backend.norm(axis) + 1e-10)

    # Skew-symmetric matrix
    K = backend.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])

    # Rodrigues formula
    return (backend.eye(3)
            + backend.sin(angle) * K
            + (1.0 - backend.cos(angle)) * (K @ K))


def skew_symmetric(backend, v):
    """Create skew-symmetric matrix from a 3D vector.

    The skew-symmetric matrix [v]_× satisfies: [v]_× @ u = v × u

    Parameters
    ----------
    backend : DifferentiableBackend
        Backend to use for computation.
    v : array, shape (3,)
        Input vector.

    Returns
    -------
    K : array, shape (3, 3)
        Skew-symmetric matrix.
    """
    return backend.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

"""NumPy backend implementation.

This module provides a NumPy-based backend that serves as the baseline
implementation. It uses numerical differentiation for gradient computation.
"""

from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np


class NumpyBackend:
    """NumPy backend for array operations.

    This backend uses NumPy for all array operations and numerical
    differentiation for gradient computation. It serves as the baseline
    implementation and fallback when JAX is not available.

    Parameters
    ----------
    dtype : numpy.dtype, optional
        Default data type for arrays. Default is float64.

    Examples
    --------
    >>> from skrobot.backend.numpy_backend import NumpyBackend
    >>> backend = NumpyBackend()
    >>> x = backend.array([1.0, 2.0, 3.0])
    >>> backend.sum(x)
    6.0
    """

    def __init__(self, dtype=np.float64):
        self._dtype = dtype

    # === Backend Info ===

    @property
    def name(self) -> str:
        """Backend name."""
        return 'numpy'

    @property
    def supports_autodiff(self) -> bool:
        """NumPy uses numerical differentiation."""
        return False

    @property
    def supports_jit(self) -> bool:
        """NumPy does not support JIT."""
        return False

    # === Array Creation ===

    def array(self, data) -> np.ndarray:
        """Convert data to numpy array."""
        return np.asarray(data, dtype=self._dtype)

    def zeros(self, shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """Create array of zeros."""
        return np.zeros(shape, dtype=self._dtype)

    def ones(self, shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """Create array of ones."""
        return np.ones(shape, dtype=self._dtype)

    def eye(self, n: int) -> np.ndarray:
        """Create identity matrix."""
        return np.eye(n, dtype=self._dtype)

    def arange(
        self,
        start: float,
        stop: Optional[float] = None,
        step: float = 1,
    ) -> np.ndarray:
        """Create array with evenly spaced values."""
        if stop is None:
            return np.arange(start, dtype=self._dtype)
        return np.arange(start, stop, step, dtype=self._dtype)

    def linspace(
        self,
        start: float,
        stop: float,
        num: int,
    ) -> np.ndarray:
        """Create array with evenly spaced values over interval."""
        return np.linspace(start, stop, num, dtype=self._dtype)

    def to_numpy(self, arr: np.ndarray) -> np.ndarray:
        """Convert to numpy array (identity for numpy backend)."""
        return np.asarray(arr)

    # === Array Manipulation ===

    def reshape(
        self,
        arr: np.ndarray,
        shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reshape array."""
        return arr.reshape(shape)

    def transpose(
        self,
        arr: np.ndarray,
        axes: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        """Transpose array."""
        return np.transpose(arr, axes)

    def concatenate(
        self,
        arrays: List[np.ndarray],
        axis: int = 0,
    ) -> np.ndarray:
        """Concatenate arrays along axis."""
        return np.concatenate(arrays, axis=axis)

    def stack(
        self,
        arrays: List[np.ndarray],
        axis: int = 0,
    ) -> np.ndarray:
        """Stack arrays along new axis."""
        return np.stack(arrays, axis=axis)

    def squeeze(self, arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Remove single-dimensional entries."""
        return np.squeeze(arr, axis=axis)

    def expand_dims(self, arr: np.ndarray, axis: int) -> np.ndarray:
        """Expand dimensions."""
        return np.expand_dims(arr, axis=axis)

    # === Math Operations ===

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication."""
        return np.matmul(a, b)

    def dot(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Dot product."""
        return np.dot(a, b)

    def sum(
        self,
        arr: np.ndarray,
        axis: Optional[int] = None,
    ) -> np.ndarray:
        """Sum of array elements."""
        return np.sum(arr, axis=axis)

    def mean(
        self,
        arr: np.ndarray,
        axis: Optional[int] = None,
    ) -> np.ndarray:
        """Mean of array elements."""
        return np.mean(arr, axis=axis)

    def sqrt(self, arr: np.ndarray) -> np.ndarray:
        """Square root."""
        return np.sqrt(arr)

    def sin(self, arr: np.ndarray) -> np.ndarray:
        """Sine."""
        return np.sin(arr)

    def cos(self, arr: np.ndarray) -> np.ndarray:
        """Cosine."""
        return np.cos(arr)

    def arctan2(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Arc tangent of y/x."""
        return np.arctan2(y, x)

    def clip(
        self,
        arr: np.ndarray,
        min_val: float,
        max_val: float,
    ) -> np.ndarray:
        """Clip values to range."""
        return np.clip(arr, min_val, max_val)

    def maximum(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise maximum."""
        return np.maximum(a, b)

    def minimum(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise minimum."""
        return np.minimum(a, b)

    def abs(self, arr: np.ndarray) -> np.ndarray:
        """Absolute value."""
        return np.abs(arr)

    def exp(self, arr: np.ndarray) -> np.ndarray:
        """Exponential."""
        return np.exp(arr)

    def log(self, arr: np.ndarray) -> np.ndarray:
        """Natural logarithm."""
        return np.log(arr)

    def power(self, arr: np.ndarray, p: float) -> np.ndarray:
        """Element-wise power."""
        return np.power(arr, p)

    # === Linear Algebra ===

    def norm(
        self,
        arr: np.ndarray,
        axis: Optional[int] = None,
    ) -> np.ndarray:
        """Vector/matrix norm."""
        return np.linalg.norm(arr, axis=axis)

    def inv(self, arr: np.ndarray) -> np.ndarray:
        """Matrix inverse."""
        return np.linalg.inv(arr)

    def pinv(self, arr: np.ndarray) -> np.ndarray:
        """Moore-Penrose pseudo-inverse."""
        return np.linalg.pinv(arr)

    def det(self, arr: np.ndarray) -> np.ndarray:
        """Matrix determinant."""
        return np.linalg.det(arr)

    def svd(self, arr: np.ndarray, full_matrices: bool = True):
        """Singular value decomposition."""
        return np.linalg.svd(arr, full_matrices=full_matrices)

    def eigh(self, arr: np.ndarray):
        """Eigenvalue decomposition for symmetric/Hermitian matrices."""
        return np.linalg.eigh(arr)

    def cross(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cross product of two 3D vectors."""
        return np.cross(a, b)

    def outer(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Outer product."""
        return np.outer(a, b)

    def trace(self, arr: np.ndarray) -> np.ndarray:
        """Matrix trace."""
        return np.trace(arr)

    # === Automatic Differentiation (Numerical) ===

    def gradient(self, fn: Callable, eps: float = 1e-7) -> Callable:
        """Return function that computes numerical gradient.

        Parameters
        ----------
        fn : callable
            Scalar-valued function.
        eps : float
            Finite difference step size.

        Returns
        -------
        callable
            Function that computes numerical gradient.
        """
        def grad_fn(x):
            x = np.asarray(x, dtype=self._dtype)
            grad = np.zeros_like(x)
            for i in range(x.size):
                x_plus = x.copy()
                x_plus.flat[i] += eps
                x_minus = x.copy()
                x_minus.flat[i] -= eps
                grad.flat[i] = (fn(x_plus) - fn(x_minus)) / (2 * eps)
            return grad
        return grad_fn

    def jacobian(self, fn: Callable, eps: float = 1e-7) -> Callable:
        """Return function that computes numerical Jacobian.

        Parameters
        ----------
        fn : callable
            Vector-valued function.
        eps : float
            Finite difference step size.

        Returns
        -------
        callable
            Function that computes Jacobian matrix.
        """
        def jac_fn(x):
            x = np.asarray(x, dtype=self._dtype)
            f0 = fn(x)
            f0 = np.atleast_1d(f0).flatten()
            m = f0.size
            n = x.size
            jac = np.zeros((m, n), dtype=self._dtype)
            for i in range(n):
                x_plus = x.copy()
                x_plus.flat[i] += eps
                f_plus = np.atleast_1d(fn(x_plus)).flatten()
                jac[:, i] = (f_plus - f0) / eps
            return jac
        return jac_fn

    def value_and_grad(self, fn: Callable, eps: float = 1e-7) -> Callable:
        """Return function that computes value and numerical gradient.

        Parameters
        ----------
        fn : callable
            Scalar-valued function.
        eps : float
            Finite difference step size.

        Returns
        -------
        callable
            Function that returns (value, gradient) tuple.
        """
        grad_fn = self.gradient(fn, eps)

        def val_grad_fn(x):
            return fn(x), grad_fn(x)
        return val_grad_fn

    def hessian(self, fn: Callable, eps: float = 1e-5) -> Callable:
        """Return function that computes numerical Hessian.

        Parameters
        ----------
        fn : callable
            Scalar-valued function.
        eps : float
            Finite difference step size.

        Returns
        -------
        callable
            Function that computes Hessian matrix.
        """
        def hess_fn(x):
            x = np.asarray(x, dtype=self._dtype)
            n = x.size
            hess = np.zeros((n, n), dtype=self._dtype)

            for i in range(n):
                for j in range(i, n):
                    x_pp = x.copy()
                    x_pp.flat[i] += eps
                    x_pp.flat[j] += eps

                    x_pm = x.copy()
                    x_pm.flat[i] += eps
                    x_pm.flat[j] -= eps

                    x_mp = x.copy()
                    x_mp.flat[i] -= eps
                    x_mp.flat[j] += eps

                    x_mm = x.copy()
                    x_mm.flat[i] -= eps
                    x_mm.flat[j] -= eps

                    hess[i, j] = (
                        fn(x_pp) - fn(x_pm) - fn(x_mp) + fn(x_mm)
                    ) / (4 * eps * eps)
                    hess[j, i] = hess[i, j]

            return hess
        return hess_fn

    # === Compilation and Vectorization ===

    def compile(self, fn: Callable) -> Callable:
        """No-op for NumPy (no JIT)."""
        return fn

    def vmap(self, fn: Callable, in_axes: int = 0) -> Callable:
        """Vectorize function over batch dimension.

        Parameters
        ----------
        fn : callable
            Function to vectorize.
        in_axes : int
            Input axis to vectorize over.

        Returns
        -------
        callable
            Vectorized function.
        """
        def batched_fn(x):
            if in_axes != 0:
                x = np.moveaxis(x, in_axes, 0)
            results = [fn(xi) for xi in x]
            return np.stack(results, axis=0)
        return batched_fn

    # === Indexing and Slicing ===

    def take(
        self,
        arr: np.ndarray,
        indices,
        axis: Optional[int] = None,
    ) -> np.ndarray:
        """Take elements from array along axis."""
        return np.take(arr, indices, axis=axis)

    def where(self, condition, x, y) -> np.ndarray:
        """Return elements from x or y depending on condition."""
        return np.where(condition, x, y)

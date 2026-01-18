"""JAX backend implementation.

This module provides a JAX-based backend with automatic differentiation,
JIT compilation, and vectorization support.
"""

import os
import platform
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


# Ensure CPU backend on Mac before JAX imports
if platform.system() == 'Darwin':
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np


# Lazy import JAX to avoid import errors when JAX is not installed
_jax = None
_jnp = None
_lax = None


def _ensure_jax():
    """Ensure JAX is imported."""
    global _jax, _jnp, _lax
    if _jax is None:
        import jax
        from jax import lax
        import jax.numpy as jnp

        _jax = jax
        _jnp = jnp
        _lax = lax

        # Enable float64 by default
        jax.config.update("jax_enable_x64", True)


class JaxBackend:
    """JAX backend for differentiable array operations.

    This backend uses JAX for array operations with automatic differentiation,
    JIT compilation, and vectorization support.

    Parameters
    ----------
    enable_x64 : bool
        Enable 64-bit floating point precision. Default is True.

    Examples
    --------
    >>> from skrobot.backend.jax_backend import JaxBackend
    >>> backend = JaxBackend()
    >>> x = backend.array([1.0, 2.0, 3.0])
    >>> def f(x):
    ...     return backend.sum(x ** 2)
    >>> grad_f = backend.gradient(f)
    >>> grad_f(x)
    Array([2., 4., 6.], dtype=float64)
    """

    def __init__(self, enable_x64: bool = True):
        _ensure_jax()
        if enable_x64:
            _jax.config.update("jax_enable_x64", True)
        self._jit_cache = {}

    # === Backend Info ===

    @property
    def name(self) -> str:
        """Backend name."""
        return 'jax'

    @property
    def supports_autodiff(self) -> bool:
        """JAX supports automatic differentiation."""
        return True

    @property
    def supports_jit(self) -> bool:
        """JAX supports JIT compilation."""
        return True

    # === Array Creation ===

    def array(self, data):
        """Convert data to JAX array."""
        _ensure_jax()
        return _jnp.asarray(data)

    def zeros(self, shape: Union[int, Tuple[int, ...]]):
        """Create array of zeros."""
        _ensure_jax()
        return _jnp.zeros(shape)

    def ones(self, shape: Union[int, Tuple[int, ...]]):
        """Create array of ones."""
        _ensure_jax()
        return _jnp.ones(shape)

    def eye(self, n: int):
        """Create identity matrix."""
        _ensure_jax()
        return _jnp.eye(n)

    def arange(
        self,
        start: float,
        stop: Optional[float] = None,
        step: float = 1,
    ):
        """Create array with evenly spaced values."""
        _ensure_jax()
        if stop is None:
            return _jnp.arange(start)
        return _jnp.arange(start, stop, step)

    def linspace(
        self,
        start: float,
        stop: float,
        num: int,
    ):
        """Create array with evenly spaced values over interval."""
        _ensure_jax()
        return _jnp.linspace(start, stop, num)

    def to_numpy(self, arr) -> np.ndarray:
        """Convert JAX array to numpy array."""
        return np.asarray(arr)

    # === Array Manipulation ===

    def reshape(self, arr, shape: Tuple[int, ...]):
        """Reshape array."""
        return arr.reshape(shape)

    def transpose(self, arr, axes: Optional[Tuple[int, ...]] = None):
        """Transpose array."""
        _ensure_jax()
        return _jnp.transpose(arr, axes)

    def concatenate(self, arrays: List, axis: int = 0):
        """Concatenate arrays along axis."""
        _ensure_jax()
        return _jnp.concatenate(arrays, axis=axis)

    def stack(self, arrays: List, axis: int = 0):
        """Stack arrays along new axis."""
        _ensure_jax()
        return _jnp.stack(arrays, axis=axis)

    def squeeze(self, arr, axis: Optional[int] = None):
        """Remove single-dimensional entries."""
        _ensure_jax()
        return _jnp.squeeze(arr, axis=axis)

    def expand_dims(self, arr, axis: int):
        """Expand dimensions."""
        _ensure_jax()
        return _jnp.expand_dims(arr, axis=axis)

    # === Math Operations ===

    def matmul(self, a, b):
        """Matrix multiplication."""
        _ensure_jax()
        return _jnp.matmul(a, b)

    def dot(self, a, b):
        """Dot product."""
        _ensure_jax()
        return _jnp.dot(a, b)

    def sum(self, arr, axis: Optional[int] = None):
        """Sum of array elements."""
        _ensure_jax()
        return _jnp.sum(arr, axis=axis)

    def mean(self, arr, axis: Optional[int] = None):
        """Mean of array elements."""
        _ensure_jax()
        return _jnp.mean(arr, axis=axis)

    def sqrt(self, arr):
        """Square root."""
        _ensure_jax()
        return _jnp.sqrt(arr)

    def sin(self, arr):
        """Sine."""
        _ensure_jax()
        return _jnp.sin(arr)

    def cos(self, arr):
        """Cosine."""
        _ensure_jax()
        return _jnp.cos(arr)

    def arctan2(self, y, x):
        """Arc tangent of y/x."""
        _ensure_jax()
        return _jnp.arctan2(y, x)

    def clip(self, arr, min_val: float, max_val: float):
        """Clip values to range."""
        _ensure_jax()
        return _jnp.clip(arr, min_val, max_val)

    def maximum(self, a, b):
        """Element-wise maximum."""
        _ensure_jax()
        return _jnp.maximum(a, b)

    def minimum(self, a, b):
        """Element-wise minimum."""
        _ensure_jax()
        return _jnp.minimum(a, b)

    def abs(self, arr):
        """Absolute value."""
        _ensure_jax()
        return _jnp.abs(arr)

    def exp(self, arr):
        """Exponential."""
        _ensure_jax()
        return _jnp.exp(arr)

    def log(self, arr):
        """Natural logarithm."""
        _ensure_jax()
        return _jnp.log(arr)

    def power(self, arr, p: float):
        """Element-wise power."""
        _ensure_jax()
        return _jnp.power(arr, p)

    # === Linear Algebra ===

    def norm(self, arr, axis: Optional[int] = None):
        """Vector/matrix norm."""
        _ensure_jax()
        return _jnp.linalg.norm(arr, axis=axis)

    def inv(self, arr):
        """Matrix inverse."""
        _ensure_jax()
        return _jnp.linalg.inv(arr)

    def pinv(self, arr):
        """Moore-Penrose pseudo-inverse."""
        _ensure_jax()
        return _jnp.linalg.pinv(arr)

    def det(self, arr):
        """Matrix determinant."""
        _ensure_jax()
        return _jnp.linalg.det(arr)

    def svd(self, arr, full_matrices: bool = True):
        """Singular value decomposition."""
        _ensure_jax()
        return _jnp.linalg.svd(arr, full_matrices=full_matrices)

    def eigh(self, arr):
        """Eigenvalue decomposition for symmetric/Hermitian matrices."""
        _ensure_jax()
        return _jnp.linalg.eigh(arr)

    def cross(self, a, b):
        """Cross product of two 3D vectors."""
        _ensure_jax()
        return _jnp.cross(a, b)

    def outer(self, a, b):
        """Outer product."""
        _ensure_jax()
        return _jnp.outer(a, b)

    def trace(self, arr):
        """Matrix trace."""
        _ensure_jax()
        return _jnp.trace(arr)

    # === Automatic Differentiation ===

    def gradient(self, fn: Callable) -> Callable:
        """Return function that computes gradient.

        Parameters
        ----------
        fn : callable
            Scalar-valued function.

        Returns
        -------
        callable
            Function that computes gradient using JAX autodiff.
        """
        _ensure_jax()
        return _jax.grad(fn)

    def jacobian(self, fn: Callable) -> Callable:
        """Return function that computes Jacobian.

        Parameters
        ----------
        fn : callable
            Vector-valued function.

        Returns
        -------
        callable
            Function that computes Jacobian matrix.
        """
        _ensure_jax()
        return _jax.jacobian(fn)

    def value_and_grad(self, fn: Callable) -> Callable:
        """Return function that computes value and gradient.

        Parameters
        ----------
        fn : callable
            Scalar-valued function.

        Returns
        -------
        callable
            Function that returns (value, gradient) tuple.
        """
        _ensure_jax()
        return _jax.value_and_grad(fn)

    def hessian(self, fn: Callable) -> Callable:
        """Return function that computes Hessian.

        Parameters
        ----------
        fn : callable
            Scalar-valued function.

        Returns
        -------
        callable
            Function that computes Hessian matrix.
        """
        _ensure_jax()
        return _jax.hessian(fn)

    # === Compilation and Vectorization ===

    def compile(self, fn: Callable) -> Callable:
        """JIT compile function for faster execution.

        Parameters
        ----------
        fn : callable
            Function to compile.

        Returns
        -------
        callable
            JIT-compiled function.
        """
        _ensure_jax()
        fn_id = id(fn)
        if fn_id not in self._jit_cache:
            self._jit_cache[fn_id] = _jax.jit(fn)
        return self._jit_cache[fn_id]

    def vmap(self, fn: Callable, in_axes: int = 0) -> Callable:
        """Vectorize function over batch dimension.

        Parameters
        ----------
        fn : callable
            Function to vectorize.
        in_axes : int or tuple
            Input axes to vectorize over.

        Returns
        -------
        callable
            Vectorized function.
        """
        _ensure_jax()
        return _jax.vmap(fn, in_axes=in_axes)

    # === JAX-specific Operations ===

    def scan(self, fn: Callable, init, xs, length: Optional[int] = None):
        """Efficient sequential computation.

        Parameters
        ----------
        fn : callable
            Function (carry, x) -> (carry, y).
        init
            Initial carry value.
        xs
            Input sequence. Can be None if length is specified.
        length : int, optional
            Number of iterations. Required if xs is None.

        Returns
        -------
        tuple
            (final_carry, stacked_outputs)
        """
        _ensure_jax()
        return _lax.scan(fn, init, xs, length=length)

    def cond(self, pred, true_fn: Callable, false_fn: Callable, *operands):
        """Conditional execution.

        Parameters
        ----------
        pred : bool
            Condition.
        true_fn : callable
            Function to call if pred is True.
        false_fn : callable
            Function to call if pred is False.
        *operands
            Arguments to pass to the selected function.

        Returns
        -------
        Result of the selected function.
        """
        _ensure_jax()
        return _lax.cond(pred, true_fn, false_fn, *operands)

    def while_loop(self, cond_fn: Callable, body_fn: Callable, init):
        """While loop.

        Parameters
        ----------
        cond_fn : callable
            Condition function.
        body_fn : callable
            Loop body function.
        init
            Initial value.

        Returns
        -------
        Final value after loop terminates.
        """
        _ensure_jax()
        return _lax.while_loop(cond_fn, body_fn, init)

    def fori_loop(self, lower: int, upper: int, body_fn: Callable, init):
        """For loop with integer indices.

        Parameters
        ----------
        lower : int
            Lower bound (inclusive).
        upper : int
            Upper bound (exclusive).
        body_fn : callable
            Loop body (i, val) -> val.
        init
            Initial value.

        Returns
        -------
        Final value after loop completes.
        """
        _ensure_jax()
        return _lax.fori_loop(lower, upper, body_fn, init)

    # === Indexing and Slicing ===

    def take(self, arr, indices, axis: Optional[int] = None):
        """Take elements from array along axis."""
        _ensure_jax()
        return _jnp.take(arr, indices, axis=axis)

    def where(self, condition, x, y):
        """Return elements from x or y depending on condition."""
        _ensure_jax()
        return _jnp.where(condition, x, y)

    def at_set(self, arr, indices, values):
        """Functional array update (JAX-style).

        Parameters
        ----------
        arr : array
            Input array.
        indices : index
            Indices to update.
        values : array
            Values to set.

        Returns
        -------
        array
            New array with updated values.
        """
        return arr.at[indices].set(values)

    def at_add(self, arr, indices, values):
        """Functional array addition (JAX-style).

        Parameters
        ----------
        arr : array
            Input array.
        indices : index
            Indices to update.
        values : array
            Values to add.

        Returns
        -------
        array
            New array with added values.
        """
        return arr.at[indices].add(values)

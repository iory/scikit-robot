"""Backend abstraction for differentiable array operations.

This module provides a unified interface for different differentiable
backends (NumPy, JAX) with autodiff support.

Example
-------
>>> from skrobot.backend import get_backend
>>> backend = get_backend('jax')
>>> x = backend.array([1.0, 2.0, 3.0])
>>> def f(x): return backend.sum(x ** 2)
>>> grad_f = backend.gradient(f)
>>> grad_f(x)  # Returns [2.0, 4.0, 6.0]
"""

# Differentiable backend system
from skrobot.backend.numpy_backend import NumpyBackend
from skrobot.backend.registry import BackendRegistry
from skrobot.backend.registry import get_backend
from skrobot.backend.registry import list_backends
from skrobot.backend.registry import set_default_backend
from skrobot.backend.registry import use_backend


BackendRegistry.register('numpy', NumpyBackend)


# Lazy registration of optional backends
def _register_optional_backends():
    """Register optional backends if their dependencies are available."""
    # JAX backend
    try:
        from skrobot.backend.jax_backend import JaxBackend
        BackendRegistry.register('jax', JaxBackend)
    except ImportError:
        pass


_register_optional_backends()


__all__ = [
    'BackendRegistry',
    'get_backend',
    'set_default_backend',
    'list_backends',
    'use_backend',
]

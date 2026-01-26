"""Backend registry for managing and switching between backends.

This module provides a registry for differentiable backends, allowing
users to easily switch between NumPy, JAX, and other backends.
"""

from contextlib import contextmanager
from typing import Dict
from typing import List
from typing import Optional
from typing import Type


class BackendRegistry:
    """Registry for managing differentiable backends.

    This class provides a central registry for backend implementations,
    allowing users to register, retrieve, and switch between backends.

    Examples
    --------
    >>> from skrobot.backend import get_backend, set_default_backend
    >>> backend = get_backend('jax')  # Get JAX backend
    >>> set_default_backend('jax')  # Set JAX as default
    >>> backend = get_backend()  # Now returns JAX backend
    """

    _backends: Dict[str, Type] = {}
    _default: Optional[str] = None
    _instance_cache: Dict[str, object] = {}

    @classmethod
    def register(cls, name: str, backend_class: Type) -> None:
        """Register a backend implementation.

        Parameters
        ----------
        name : str
            Name of the backend (e.g., 'numpy', 'jax').
        backend_class : type
            Backend class implementing DifferentiableBackend protocol.
        """
        cls._backends[name] = backend_class

    @classmethod
    def get(cls, name: Optional[str] = None, **kwargs) -> object:
        """Get a backend instance.

        Parameters
        ----------
        name : str, optional
            Name of the backend. If None, returns the default backend.
        **kwargs
            Additional arguments passed to the backend constructor.

        Returns
        -------
        DifferentiableBackend
            Backend instance.

        Raises
        ------
        ValueError
            If the backend is not found.
        """
        if name is None:
            name = cls._default or cls._auto_select()

        if name not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(
                f"Unknown backend: '{name}'. "
                f"Available backends: {available}"
            )

        # Use cached instance if no kwargs and already cached
        cache_key = name
        if not kwargs and cache_key in cls._instance_cache:
            return cls._instance_cache[cache_key]

        # Create new instance
        try:
            instance = cls._backends[name](**kwargs)
            if not kwargs:
                cls._instance_cache[cache_key] = instance
            return instance
        except (ImportError, AttributeError, TypeError) as e:
            raise ImportError(
                f"Backend '{name}' is registered but its dependencies "
                f"are not available: {e}"
            )

    @classmethod
    def set_default(cls, name: str) -> None:
        """Set the default backend.

        Parameters
        ----------
        name : str
            Name of the backend to use as default.

        Raises
        ------
        ValueError
            If the backend is not registered.
        """
        if name not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(
                f"Unknown backend: '{name}'. "
                f"Available backends: {available}"
            )
        cls._default = name

    @classmethod
    def available(cls) -> List[str]:
        """Get list of available backends.

        A backend is considered available if it is registered and
        its dependencies can be imported.

        Returns
        -------
        list of str
            Names of available backends.
        """
        available = []
        for name in cls._backends:
            try:
                cls.get(name)
                available.append(name)
            except (ImportError, AttributeError, TypeError):
                pass
        return available

    @classmethod
    def _auto_select(cls) -> str:
        """Automatically select the best available backend.

        Priority order:
        1. JAX (if available and compatible with NumPy version)
        2. NumPy (always available)

        Returns
        -------
        str
            Name of the selected backend.
        """
        # Check if JAX is truly usable (not just importable)
        from skrobot.pycompat import HAS_JAX

        if HAS_JAX and 'jax' in cls._backends:
            try:
                cls.get('jax')
                return 'jax'
            except (ImportError, AttributeError, TypeError):
                pass

        # Fallback to numpy
        return 'numpy'

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the instance cache."""
        cls._instance_cache.clear()


# Convenience functions

def get_backend(name: Optional[str] = None, **kwargs):
    """Get a backend instance.

    Parameters
    ----------
    name : str, optional
        Name of the backend ('numpy', 'jax').
        If None, returns the default backend.
    **kwargs
        Additional arguments passed to the backend constructor.

    Returns
    -------
    DifferentiableBackend
        Backend instance.

    Examples
    --------
    >>> from skrobot.backend import get_backend
    >>> backend = get_backend('jax')
    >>> x = backend.array([1.0, 2.0, 3.0])
    >>> backend.sum(x)
    Array(6., dtype=float64)

    >>> backend = get_backend('numpy')
    >>> x = backend.array([1.0, 2.0, 3.0])
    >>> backend.sum(x)
    6.0
    """
    return BackendRegistry.get(name, **kwargs)


def set_default_backend(name: str) -> None:
    """Set the default backend.

    Parameters
    ----------
    name : str
        Name of the backend to use as default.

    Examples
    --------
    >>> from skrobot.backend import set_default_backend, get_backend
    >>> set_default_backend('jax')
    >>> backend = get_backend()  # Returns JAX backend
    """
    BackendRegistry.set_default(name)


def list_backends() -> List[str]:
    """Get list of available backends.

    Returns
    -------
    list of str
        Names of available backends.

    Examples
    --------
    >>> from skrobot.backend import list_backends
    >>> list_backends()
    ['numpy', 'jax']
    """
    return BackendRegistry.available()


@contextmanager
def use_backend(name: str, **kwargs):
    """Context manager for temporarily using a different backend.

    Parameters
    ----------
    name : str
        Name of the backend to use.
    **kwargs
        Additional arguments passed to the backend constructor.

    Yields
    ------
    DifferentiableBackend
        Backend instance.

    Examples
    --------
    >>> from skrobot.backend import use_backend, get_backend
    >>> with use_backend('jax') as backend:
    ...     x = backend.array([1.0, 2.0, 3.0])
    ...     print(backend.sum(x))
    6.0
    """
    old_default = BackendRegistry._default
    try:
        BackendRegistry.set_default(name)
        yield BackendRegistry.get(name, **kwargs)
    finally:
        BackendRegistry._default = old_default

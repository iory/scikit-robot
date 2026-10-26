from functools import lru_cache  # noqa: F401  re-exported for backward compatibility
import os
import platform


# Ensure CPU backend on Mac before JAX imports
if platform.system() == 'Darwin':
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = 'cpu'


def has_jax():
    """Whether JAX is installed and works with the installed NumPy.

    Checked on first call, not on import: importing JAX (and running a
    small ``lax.scan`` to check it) takes a noticeable fraction of a
    second and can print a GPU backend warning, so modules that only need
    ``lru_cache`` or ``is_wsl`` from here should not pay for it.
    """
    global _HAS_JAX
    if _HAS_JAX is None:
        try:
            import jax  # NOQA
            from jax import lax
            import jax.numpy as jnp

            # Verify JAX actually works with current NumPy version.
            # JAX may import successfully but fail at runtime with older
            # NumPy; lax.scan triggers np.asarray(..., copy=...) internally.
            def _test_fn(carry, _):
                return carry + 1, None
            lax.scan(_test_fn, jnp.array(0.0), None, length=1)
            _HAS_JAX = True
        except (ImportError, AttributeError, TypeError):
            # JAX not installed or incompatible with current NumPy/SciPy
            _HAS_JAX = False
    return _HAS_JAX


_HAS_JAX = None


def __getattr__(name):
    # ``from skrobot.pycompat import HAS_JAX`` keeps working, checked lazily.
    if name == 'HAS_JAX':
        return has_jax()
    raise AttributeError(
        'module {!r} has no attribute {!r}'.format(__name__, name))


def is_wsl():
    """Check if running in WSL environment."""
    if platform.system() != 'Linux':
        return False
    if os.path.exists('/proc/version'):
        try:
            with open('/proc/version', 'r', encoding='utf-8') as f:
                version_info = f.read().lower()
            if 'microsoft' in version_info or 'wsl' in version_info:
                return True
        except (OSError, IOError, PermissionError):
            pass
    return False

import functools
import os
import platform


if hasattr(functools, 'lru_cache'):
    lru_cache = functools.lru_cache
else:
    import repoze.lru
    lru_cache = repoze.lru.lru_cache


# Ensure CPU backend on Mac before JAX imports
if platform.system() == 'Darwin':
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = 'cpu'

# Check if JAX is available and compatible with current NumPy version
try:
    import jax  # noqa: F401
    HAS_JAX = True
except (ImportError, AttributeError):
    # JAX not installed or incompatible with current NumPy version
    HAS_JAX = False


def is_wsl():
    """Check if running in WSL environment."""
    if platform.system() != 'Linux':
        return False
    if os.path.exists('/proc/version'):
        try:
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
            if 'microsoft' in version_info or 'wsl' in version_info:
                return True
        except (OSError, IOError, PermissionError):
            pass
    return False

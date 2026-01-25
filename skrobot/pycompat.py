import functools
import os
import platform


if hasattr(functools, 'lru_cache'):
    lru_cache = functools.lru_cache
else:
    import repoze.lru
    lru_cache = repoze.lru.lru_cache


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

import functools


if hasattr(functools, 'lru_cache'):
    lru_cache = functools.lru_cache
else:
    import repoze.lru
    lru_cache = repoze.lru.lru_cache

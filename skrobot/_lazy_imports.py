_trimesh = None


def _lazy_trimesh():
    global _trimesh
    if _trimesh is None:
        import trimesh
        _trimesh = trimesh
    return _trimesh


_scipy = None


def _lazy_scipy():
    global _scipy
    if _scipy is None:
        import scipy
        _scipy = scipy
    return _scipy

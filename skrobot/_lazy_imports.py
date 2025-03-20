_trimesh = None


def _lazy_trimesh():
    global _trimesh
    if _trimesh is None:
        import trimesh
        _trimesh = trimesh
    return _trimesh

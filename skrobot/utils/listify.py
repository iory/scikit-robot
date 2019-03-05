def listify(x, n=1):
    ret = None
    if isinstance(x, list):
        ret = x
    else:
        ret = [x] * n
    return ret

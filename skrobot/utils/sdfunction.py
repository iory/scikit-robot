import numpy as np

def sdf_box(X, b, c):
    """ signed distnace function for a box.
    Parameters
    -------
    X : 2d numpy.ndarray (n_point x 3)
        input points
    Returns
    ------
    singed distances : 1d numpy.ndarray (n_point)
    """
    n_pts, dim =  X.shape
    assert dim==3, "dim must be 3"
    center = np.array(c).reshape(1, dim)
    center_copied = np.repeat(center, n_pts, axis=0)
    P = X - center_copied
    Q = np.abs(P) - np.repeat(np.array(b).reshape(1, dim), n_pts, axis=0)
    left__ = np.array([Q, np.zeros((n_pts, dim))])
    left_ = np.max(left__, axis = 0)
    left = np.sqrt(np.sum(left_**2, axis=1))
    right_ = np.max(Q, axis=1)
    right = np.min(np.array([right_, np.zeros(n_pts)]), axis=0)
    return left + right 


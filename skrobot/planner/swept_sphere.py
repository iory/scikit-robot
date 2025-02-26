import numpy as np
from sklearn.covariance import EmpiricalCovariance


def compute_swept_sphere(collision_mesh,
                         n_sphere=None,
                         tol=0.1):
    """Compute swept spheres approximating a mesh

    Parameters
    ----------
    collision_mesh : trimesh.Trimesh
        mesh which swept spheres are computed for
    n_sphere : int or None
        number of sphere to approximate the mesh. If it's set to `None`,
        the number of sphere is automatically determined.
    tol : float
        tolerance determines how much mesh jutting-out from the swept-spheres
        are accepted. Let `max_jut` be the maximum jut-distance. Then the
        setting `tol` enforces `max_jut / radius < max_jut`. If some integer
        is set to `n_sphere`, `tol` does not affects the result.

    Returns
    -------
    centers_original_space : numpy.ndarray[float](n_sphere, 3)
        center of the approximating spheres in the space where the
        mesh vertices are defined.

    radius : float
        radius of approximating sphers.
    """

    verts = collision_mesh.vertices

    # first we compute the principal directions of the vertices.
    mean = np.mean(verts, axis=0)
    verts_slided = verts - mean[None, :]
    cov = EmpiricalCovariance().fit(verts_slided)
    eig_vals, basis_tf_mat = np.linalg.eig(cov.covariance_)
    principle_axis = np.argmax(eig_vals)

    # and map to the space spanned by the principle vectors.
    # Also, compute the inverse map, to re-map them to the original
    # splace in the end of this function.
    verts_mapped = verts_slided.dot(basis_tf_mat)

    def inverse_map(verts):
        return verts.dot(basis_tf_mat.T) + mean[None, :]

    # get the indexes of the place axis
    if principle_axis == 0:
        plane_axes = [1, 2]
    elif principle_axis == 1:
        plane_axes = [2, 0]
    else:
        plane_axes = [0, 1]

    # then compute the bounding-circle for vertices projected
    # to the plane.
    def determine_radius(verts_2d_projected):
        X, Y = verts_2d_projected.T
        radius_vec = np.sqrt(X**2 + Y**2)
        radius = np.max(radius_vec)
        return radius

    margin_factor = 1.01
    radius = determine_radius(verts_mapped[:, plane_axes]) * margin_factor

    # compute the maximum and minimum heights (h_center_max, h_center_min)
    # of the sphere centers.
    # Here, height is defined in the principle direction.
    squared_radius_arr = np.sum(verts_mapped[:, plane_axes] ** 2, axis=1)
    h_center_arr = verts_mapped[:, principle_axis]

    h_vert_max = np.max(verts_mapped[:, principle_axis])
    h_vert_min = np.min(verts_mapped[:, principle_axis])

    def get_h_center_max():
        def cond_all_inside_positive(h_center_max):
            sphere_heights = h_center_max +\
                np.sqrt(radius**2 - squared_radius_arr)
            return np.all(sphere_heights > h_center_arr)
        # get first index that satisfies the condition
        h_cand_list = np.linspace(0, h_vert_max, 30)
        idx = np.where([cond_all_inside_positive(h)
                        for h in h_cand_list])[0][0]
        h_center_max = h_cand_list[idx]
        return h_center_max

    def get_h_center_min():
        def cond_all_inside_negative(h_center_min):
            sphere_heights = h_center_min - \
                np.sqrt(radius**2 - squared_radius_arr)
            return np.all(h_center_arr > sphere_heights)
        # get first index that satisfies the condition
        h_cand_list = np.linspace(0, h_vert_min, 30)
        idx = np.where([cond_all_inside_negative(h)
                        for h in h_cand_list])[0][0]
        h_center_min = h_cand_list[idx]
        return h_center_min

    h_center_max = get_h_center_max()
    h_center_min = get_h_center_min()

    # using h_center_min and h_center_max, generate center points in
    # the mapped space.
    def compute_center_pts_mapped_space(n_sphere):
        h_centers = np.linspace(h_center_min, h_center_max, n_sphere)
        centers = np.zeros((n_sphere, 3))
        centers[:, principle_axis] = h_centers
        return centers

    auto_determine_n_sphere = (n_sphere is None)
    if auto_determine_n_sphere:
        n_sphere = 1
        while True:  # iterate until the approximation satisfies tolerance
            centers_pts_mapped_space =\
                compute_center_pts_mapped_space(n_sphere)
            dists_foreach_sphere = np.array([
                np.sqrt(np.sum((verts_mapped - c[None, :])**2, axis=1))
                for c in centers_pts_mapped_space])
            # verts distance to the approximating spheres
            # if this distance is positive value, the vertex is jutting-out
            # from the swept-sphere.
            jut_dists = np.min(dists_foreach_sphere, axis=0) - radius
            max_jut = np.max(jut_dists)
            err_ratio = max_jut / radius
            if err_ratio < tol:
                break
            n_sphere += 1
    else:
        centers_pts_mapped_space = compute_center_pts_mapped_space(n_sphere)

    # map all centers to the original space
    centers_original_space = inverse_map(centers_pts_mapped_space)
    return centers_original_space, radius

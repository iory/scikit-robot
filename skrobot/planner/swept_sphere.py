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


def _sample_surface(mesh, spacing):
    """Points on a mesh's surface, every point of it within ``spacing`` of one.

    Each triangle is split into n x n smaller ones, n from its longest edge,
    and their corners are returned (the mesh's vertices too).
    """
    tri = mesh.triangles
    edge = np.max(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2), axis=1)
    splits = np.maximum(1, np.ceil(edge / spacing).astype(int))
    out = [np.asarray(mesh.vertices)]
    for n in np.unique(splits):
        t = tri[splits == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
        keep = i + j <= n
        a, b = i[keep] / n, j[keep] / n
        pts = (t[:, None, 0] * (1 - a - b)[None, :, None]
               + t[:, None, 1] * a[None, :, None]
               + t[:, None, 2] * b[None, :, None])
        out.append(pts.reshape(-1, 3))
    return np.concatenate(out)


def compute_fitted_spheres(collision_mesh, tol=0.01, res=None,
                           solid='hull', max_candidates=64):
    """Compute spheres of varying radius covering a mesh

    Unlike :func:`compute_swept_sphere`, which puts spheres of one radius
    along the mesh's main axis, each sphere here has its own radius, and
    none reaches more than about ``tol`` (in metres) out of the solid. A box,
    a bracket or a wide palm then takes a few big spheres inside and small
    ones at its corners, rather than many of one size.

    Candidate centres are the cells of a ``res`` grid inside the solid, each
    with the radius that reaches ``tol`` past the solid's surface nearest to
    it, plus ``res``. The mesh's surface is sampled at ``res`` and covered
    greedily: take a sample not covered yet and, of the candidates covering
    it, keep the one covering the most samples not covered yet. A sample
    counts as covered within ``res`` of the sphere's surface, so the whole
    surface is covered, not only the samples.

    Parameters
    ----------
    collision_mesh : trimesh.Trimesh
        Mesh to cover, in its own frame.
    tol : float
        How far a sphere may reach out of the solid [m]. The grid adds up to
        ``2 * res`` to that (a cell's worth for the covering between samples,
        and up to a cell for where the grid puts the solid's surface).
    res : float or None
        Grid cell size [m]; ``tol / 2`` if None. It is never finer than
        1/64 of the mesh's largest extent, which keeps a big mesh fast.
    solid : str
        What the spheres stay in: ``'hull'`` (the convex hull, the default:
        visual meshes are often open shells that enclose nothing) or
        ``'mesh'`` (what the mesh itself encloses, its holes filled).
    max_candidates : int
        Candidates compared for a sample when more cover it: the biggest
        half of them and a random half of the rest.

    Returns
    -------
    centers : numpy.ndarray[float](n_sphere, 3)
        Centres of the spheres, in the mesh's frame.
    radii : numpy.ndarray[float](n_sphere,)
        Their radii.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    if solid not in ('hull', 'mesh'):
        raise ValueError(
            "solid must be 'hull' or 'mesh', got {!r}".format(solid))
    res = tol / 2.0 if res is None else res
    res = max(res, float(np.max(collision_mesh.extents)) / 64.0)
    pts = _sample_surface(collision_mesh, res)
    lo = pts.min(axis=0) - 2 * res
    shape = np.ceil((pts.max(axis=0) + 2 * res - lo) / res).astype(int) + 1
    surface_cells = tuple(np.floor((pts - lo) / res).astype(int).T)
    if solid == 'hull':
        hull = collision_mesh.convex_hull
        normals = hull.face_normals
        offsets = -np.einsum('ij,ij->i', normals, hull.triangles[:, 0])
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape],
                                   indexing='ij'), axis=-1).reshape(-1, 3)
        x = lo + (idx + 0.5) * res
        inside = np.ones(len(x), dtype=bool)
        for k in range(0, len(normals), 256):
            inside &= np.all(
                x @ normals[k:k + 256].T + offsets[k:k + 256] <= 0, axis=1)
        inside = inside.reshape(shape)
        inside[surface_cells] = True
    else:
        occupied = np.zeros(shape, dtype=bool)
        occupied[surface_cells] = True
        inside = ndimage.binary_fill_holes(occupied)

    # how far each cell's centre is from the nearest cell outside, less a cell
    room = np.maximum(ndimage.distance_transform_edt(inside) * res - res, 0.0)
    cells = np.argwhere(inside)
    candidates = lo + (cells + 0.5) * res
    reach = room[tuple(cells.T)] + tol + res
    # a sample covered this far in leaves every surface point within res of
    # it inside the sphere
    covers = reach - res
    candidate_tree = cKDTree(candidates)
    sample_tree = cKDTree(pts)
    uncovered = np.ones(len(pts), dtype=bool)
    rng = np.random.default_rng(0)
    centers, radii = [], []
    while uncovered.any():
        p = pts[np.flatnonzero(uncovered)[0]]
        near = np.array(candidate_tree.query_ball_point(p, covers.max()),
                        dtype=int)
        near = near[np.linalg.norm(candidates[near] - p, axis=1) <= covers[near]]
        if len(near) == 0:
            # beyond every candidate's reach: a small sphere of its own
            centers.append(p)
            radii.append(tol + res)
            uncovered[sample_tree.query_ball_point(p, tol)] = False
            continue
        if len(near) > max_candidates:
            order = np.argsort(-reach[near])
            half = max_candidates // 2
            near = np.concatenate([
                near[order[:half]],
                rng.choice(near[order[half:]], max_candidates - half,
                           replace=False)])
        best, best_count, best_covered = None, -1, None
        for c in near:
            covered = np.array(
                sample_tree.query_ball_point(candidates[c], covers[c]),
                dtype=int)
            count = int(uncovered[covered].sum()) if len(covered) else 0
            if count > best_count:
                best, best_count, best_covered = c, count, covered
        centers.append(candidates[best])
        radii.append(reach[best])
        uncovered[best_covered] = False
    return np.array(centers), np.array(radii)

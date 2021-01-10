import copy

import numpy as np
import scipy
import trimesh

from skrobot.coordinates import CascadedCoords
from skrobot.model.primitives import Sphere
from skrobot.planner.swept_sphere import compute_swept_sphere
from skrobot.planner.utils import forward_kinematics_multi
from skrobot.planner.utils import get_robot_config


class SweptSphereSdfCollisionChecker(object):
    """Collision checker between swept spheres and sdf"""

    def __init__(self, sdf, robot_model):
        self.sdf = sdf
        self.robot_model = robot_model

        self.coll_link_name_list = []
        self.coll_sphere_list = []
        self.coll_radius_list = []
        self.coll_coords_list = []

        self.color_normal_sphere = [250, 250, 10, 200]
        self.color_collision_sphere = [255, 0, 0, 200]

    @property
    def n_feature(self):
        """Return number of collision sphere.

        Returns
        -------
        n_feature : int
            number of collision spheres.
        """
        return len(self.coll_sphere_list)

    def add_coll_spheres_to_viewer(self, viewer):
        """Add collision sheres to viewer

        Parameters
        ----------
        viewer : skrobot.viewers._trimesh.TrimeshSceneViewer
            viewer
        """

        for s in self.coll_sphere_list:
            viewer.add(s)

    def delete_coll_spheres_from_viewer(self, viewer):
        """Delete collision sheres from viewer

        Parameters
        ----------
        viewer : skrobot.viewers._trimesh.TrimeshSceneViewer
            viewer
        """
        for s in self.coll_sphere_list:
            viewer.delete(s)

    def add_collision_link(self, coll_link):
        """Add link for which collision with sdf is checked

        The given `coll_link` will be approximated by swept-spheres
        and these spheres will be added to collision sphere's list.

        Parameters
        ----------
        coll_link : skrobot.model.Link
            link for which collision with sdf is checked
        """

        if coll_link.name in self.coll_link_name_list:
            return
        self.coll_link_name_list.append(coll_link)

        col_mesh = coll_link.collision_mesh
        assert type(col_mesh) == trimesh.base.Trimesh

        centers, R = compute_swept_sphere(col_mesh)
        sphere_list = []
        coords_list = []
        for center in centers:
            link_pos = coll_link.copy_worldcoords()
            coll_coords = CascadedCoords(
                pos=link_pos.worldpos(),
                rot=link_pos.worldrot())
            coll_coords.translate(center)
            coll_link.assoc(coll_coords)
            coords_list.append(coll_coords)

            # add sphere
            sp = Sphere(radius=R, pos=coll_coords.worldpos(),
                        color=self.color_normal_sphere)
            coll_coords.assoc(sp)
            sphere_list.append(sp)

        self.coll_sphere_list.extend(sphere_list)
        self.coll_coords_list.extend(coords_list)
        self.coll_radius_list.extend([R] * len(sphere_list))

    def add_collision_links(self, coll_links):
        """Add links for which collisions with SDF is checked.

        The given `coll_links` will be approximated by swept-spheres
        and these spheres will be added to collision sphere's list.

        Parameters
        ----------
        coll_links : list[skrobot.model.Link]
            link list for which collisions with sdf is checked.
        """
        for coll_link in coll_links:
            self.add_collision_link(coll_link)

    def collision_check(self):
        """Check collision between links and collision spheres.

        Returns
        -------
        is_collision : bool
            `True` if a collision occurred between any pair of links and
            collision spheres and `False` otherwise.
        """
        joint_list = [j for j in self.robot_model.joint_list]
        angle_vector = get_robot_config(
            self.robot_model, joint_list, with_base=True)

        dists, _ = self.compute_batch_sd_vals(
            joint_list, np.array([angle_vector]),
            with_base=True, with_jacobian=False)
        idxes_collide = np.where(dists < 0)[0]
        return len(idxes_collide) > 0

    def update_color(self):  # for debugging
        """Update the color of links under collision

        This method checks the collision between the collision
        spheres and registered sdf. If collision spheres are
        found to be under collision, the color of the spheres
        will be changed to `color_collision_sphere`.

        Returns
        -------
        dists : numpy.ndarray(n_sphere,)
            array of the signed distances for each sphere against sdf.
        """

        joint_list = [j for j in self.robot_model.joint_list]
        angle_vector = get_robot_config(
            self.robot_model, joint_list, with_base=True)

        dists, _ = self.compute_batch_sd_vals(
            joint_list, np.array([angle_vector]),
            with_base=True, with_jacobian=False)
        idxes_collide = np.where(dists < 0)[0]

        for idx in range(self.n_feature):
            sphere = self.coll_sphere_list[idx]
            n_facet = len(sphere._visual_mesh.visual.face_colors)

            color = self.color_collision_sphere if idx in idxes_collide \
                else self.color_normal_sphere
            sphere._visual_mesh.visual.face_colors = np.array(
                [color] * n_facet)
        return dists

    def compute_batch_sd_vals(self,
                              joint_list,
                              angle_vector_seq,
                              with_base=False,
                              with_jacobian=False):
        """Compute sd signed distances of collision spheres

        This method is the core of this class. We assume that this
        method is mainly called from a trajecotyr optimizer or
        path planner.

        Let :math:`n_{wp}` be the number of way-points of a trajectory.
        Let :math:`n_{f}` be the number of collision feature points on
        the robot links.

        Let :math:`f_{i, j} : \\mathbb{R}^{n_{dof}}
        \\ni q \\mapsto x \\in \\mathbb{R}^3`
        be the forward kinematics of collision features point :math:`j`
        at waypoint :math:`i` where :math:`q` is the angle vector and
        :math:`n_{dof}` is the dimension of the angle vector.

        Let :math:`c : \\mathbb{R}^3 \\ni x \\mapsto sd \\in \\mathbb{R}`
        be the signd distance function.

        Then, this fucntion is defined as
        :math:`F : \\mathbb{R}^{n_{wp} n_{dof}} \\ni \\xi \\mapsto
        [
        [f_{1, 1}(q_1), \\ldots, f_{1, n_{f}}(q_1)]^T,
        \\ldots,
        [f_{n_{wp}, 1}(q_{n_{wp}}),\\ldots,f_{n_{wp}, n_{f}}(q_{n_{wp}})]^T
        ]^T
        \\in \\mathbb{R}^{n_{wp} n_{f}}`
        where :math:`\\xi = [q_1^T, \\ldots, q_{n_{wp}}^T]^T` be
        the angle vector sequence of the trajectory.
        The corresponding jacobian is defined as
        :math:`\\frac{\\partial F}{\\partial \\xi}`.

        Parameters
        ----------
        joint_list : list[skrobot.model.Joint]
            joint list to be set
        angle_vector_seq : numpy.ndarray[float](n_wp, n_dof)
            angle vector sequence.
        with_base : bool
            hoge
        with_jacobian : bool
            if `True`, jacobian is copmuted at the same time.
        Returns
        -------
        sd_vals : numpy.ndarray[float](n_wp * n_feature,)
            signed distnaces for all feature points through the trajectory
        sd_vals_jacobi : numpy.ndarray[float](n_wp * n_feature, n_wp * n_dof)
            jacobain of sd_vals with respect to DOF (i.e. n_wp * n_dof) of the
            trajectory
        """

        P, J = self._coll_batch_forward_kinematics(
            joint_list, angle_vector_seq,
            with_base=with_base,
            with_jacobian=with_jacobian)
        sd_vals, sd_vals_jacobi = self._compute_batch_sd_vals_internal(P, J)
        return sd_vals, sd_vals_jacobi

    def _compute_batch_sd_vals_internal(self, pts_batch, jac_batch):
        """This function must be implemented considering performance

        Parameters
        ----------
        pts_batch : numpy.ndarray(n_wp, n_feature, 3)
            all feature points in a trajectory.
        jac_batch : numpy.ndarray(n_wp, n_feature, 3, n_dof) or None
            jacobians for all feature points in a trajectory. If set to
            None, jacobian for `pts_batch` will not be computed.

        Returns
        -------
        sd_vals_batch_flatten_sphere :
        numpy.ndarray[float](n_wp * n_feature,)
            signed distnaces for all feature points through the trajectory
        jac_whole : numpy.ndarray[float](n_wp * n_feature, n_wp * n_dof)
            jacobain of sd_vals with respect to DOF (i.e. n_wp * n_dof) of the
            trajectory. If `jac_batch = None`, `jac_whole = None`.
        """

        with_jacobian = (jac_batch is not None)

        n_wp, n_feature, _ = pts_batch.shape
        n_total_feature = n_wp * n_feature

        # flattening (n_wp, n_feature, 3) -> (n_wp * n_feature, 3)
        # is necessary to copmute sdf() without using loop.
        pts_batch_flatten = pts_batch.reshape((n_total_feature, 3))
        sd_vals_batch_flatten = self.sdf(pts_batch_flatten)

        # we must convert it to sphere's signed distnace
        radius_arr_traj = np.array(self.coll_radius_list * n_wp)
        sd_vals_batch_flatten_sphere = sd_vals_batch_flatten - radius_arr_traj

        if not with_jacobian:
            return sd_vals_batch_flatten_sphere, None

        _, _, _, n_dof = jac_batch.shape
        # numerically grad array of sdf
        eps = 1e-7
        sd_val_grads_batch_flatten = np.zeros((n_total_feature, 3))
        for idx in range(3):
            pts_batch_flatten_plus = copy.copy(pts_batch_flatten)
            pts_batch_flatten_plus[:, idx] += eps
            sd_vals_batch_flatten_plus = self.sdf(pts_batch_flatten_plus)
            diff = sd_vals_batch_flatten_plus - sd_vals_batch_flatten
            sd_val_grads_batch_flatten[:, idx] = diff / eps
        sd_val_grads_batch = sd_val_grads_batch_flatten.reshape(
            n_wp, n_feature, 3)

        # (n_wp, n_feature, 3) x (n_wp, n_feature, 3, n_dof)
        # => (n_wp, n_feature, n_dof)
        jac_whole_batch = np.einsum(
            'ijk,ijkl->ijl', sd_val_grads_batch, jac_batch)

        jac_whole = scipy.linalg.block_diag(*list(jac_whole_batch))
        return sd_vals_batch_flatten_sphere, jac_whole

    def _coll_batch_forward_kinematics(self,
                                       joint_list,
                                       angle_vector_seq,
                                       with_base,
                                       with_jacobian):
        """This function must be implemented considering performance

        Parameters
        ----------
        joint_list : list[skrobot.model.Joint]
            joint list to be considered in the forward kinematics.
        angle_vector_seq : numpy.ndarray[float](n_wp, n_dof)
            angle vector sequence.
        with_base : bool
            hoge
        with_jacobian : bool
            if `True`, jacobian is copmuted at the same time.
        Returns
        -------
        pts_batch : numpy.ndarray(n_wp, n_feature, 3)
            all feature points in a trajectory.
        jac_batch : numpy.ndarray(n_wp, n_feature, 3, n_dof)
            jacobians for all feature points in a trajectory.
        """

        if isinstance(angle_vector_seq, list):
            angle_vector_seq = np.array(angle_vector_seq)

        n_wp, n_dof = angle_vector_seq.shape
        pts_batch = np.zeros((n_wp, self.n_feature, 3))

        if with_jacobian:
            jac_batch = np.zeros((n_wp, self.n_feature, 3, n_dof))
        else:
            jac_batch = None

        for i in range(n_wp):
            av = angle_vector_seq[i]
            pts_arr, jac_arr = forward_kinematics_multi(
                self.robot_model, joint_list, av, self.coll_coords_list,
                with_rot=False, with_base=with_base,
                with_jacobian=with_jacobian)

            pts_batch[i, :, :] = pts_arr
            if with_jacobian:
                jac_batch[i, :, :, :] = jac_arr

        return pts_batch, jac_batch

import tinyfk

from skrobot.planner import SweptSphereSdfCollisionChecker


class TinyfkSweptSphereSdfCollisionChecker(SweptSphereSdfCollisionChecker):
    """Collision checker using tinyfk as a backend"""

    def __init__(self, sdf, robot_model):
        super(TinyfkSweptSphereSdfCollisionChecker, self).__init__(
            sdf, robot_model)
        self.fksolver = tinyfk.RobotModel(robot_model.urdf_path)
        self.coll_sphere_id_list = []

    def add_collision_link(self, coll_link):
        center_list, _, sphere_list, _ =\
            super(TinyfkSweptSphereSdfCollisionChecker, self).\
            add_collision_link(coll_link)
        coll_link_id = self.fksolver.get_link_ids([coll_link.name])[0]
        sphere_name_list = [s.name for s in sphere_list]
        for name, center in zip(sphere_name_list, center_list):
            self.fksolver.add_new_link(name, coll_link_id, center)

        sphere_ids = self.fksolver.get_link_ids(sphere_name_list)
        self.coll_sphere_id_list.extend(sphere_ids)

    def compute_batch_sd_vals(
            self,
            joint_list,
            angle_vector_seq,
            with_base=False,
            with_jacobian=False):
        """Mocking compute_batch_sd_vals of the super class

        get_joint_ids comes with some overhead, which is critical
        in the cpp based trajectory planner. So, Do not call this,
        and instead, call _compute_batchsd_vals in path-planning
        applications.
        """
        joint_name_list = [j.name for j in joint_list]
        joint_ids = self.fksolver.get_joint_ids(joint_name_list)
        return self._compute_batch_sd_vals(
            joint_ids,
            angle_vector_seq,
            with_base=with_base,
            with_jacobian=with_jacobian)

    def _compute_batch_sd_vals(
            self,
            joint_ids,
            angle_vector_seq,
            with_base=False,
            with_jacobian=False):
        """This will be called from the optimization

        As python doesn't supprot multiple dispatch,
        we define _compute_batchsd_vals whcih takes
        different type of arguments.
        """

        n_wp, n_dof = angle_vector_seq.shape
        with_rot = False
        P_tmp, J_tmp = self.fksolver.solve_forward_kinematics(
            angle_vector_seq, self.coll_sphere_id_list, joint_ids,
            with_rot, with_base, with_jacobian)
        # because pybind can handle only 2-dim matrix, we must reshape it
        P = P_tmp.reshape(n_wp, self.n_feature, 3)
        if with_jacobian:
            J = J_tmp.reshape(n_wp, self.n_feature, 3, n_dof)
        else:
            J = None
        sd_vals, sd_vals_jacobi = self._compute_batch_sd_vals_internal(P, J)
        return sd_vals, sd_vals_jacobi

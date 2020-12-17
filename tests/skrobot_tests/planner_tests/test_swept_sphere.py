import unittest

import numpy as np
import trimesh

import skrobot
from skrobot.planner.swept_sphere import compute_swept_sphere


class TestSweptSphere(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        robot_model = skrobot.models.PR2()
        cls.mesh1 = robot_model.r_gripper_palm_link._visual_mesh[0]

        objfile_path = skrobot.data.bunny_objpath()
        bunnymesh = trimesh.load_mesh(objfile_path)
        cls.mesh2 = bunnymesh

    def test_copmute_swept_sphere(self):

        def approximation_accuracy_test(mesh, tol):
            center_pts, radius = compute_swept_sphere(mesh, tol=tol)

            def swept_sphere_sdf(pts):
                dists_list = []
                for c in center_pts:
                    dists = np.sqrt(np.sum((pts - c[None, :])**2, axis=1))
                    dists_list.append(dists)
                dists_arr = np.array(dists_list)
                # union sdf of all spheres
                signed_dists = np.min(dists_arr, axis=0) - radius
                return signed_dists

            jut_arr = swept_sphere_sdf(mesh.vertices)
            max_jut = np.max(jut_arr)
            self.assertLess(max_jut / radius, tol)

        tol = 1e-3
        approximation_accuracy_test(self.mesh1, tol)
        approximation_accuracy_test(self.mesh2, tol)

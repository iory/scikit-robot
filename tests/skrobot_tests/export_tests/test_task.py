import os
import shutil
import tempfile
import unittest

import numpy as np


try:
    from pxr import UsdPhysics
    _HAS_PXR = True
except ImportError:
    _HAS_PXR = False

import skrobot
from skrobot.export import export_for_task
from skrobot.models.urdf import RobotModelFromURDF


@unittest.skipUnless(_HAS_PXR, 'usd-core (pxr) not installed')
class TestExportForTask(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(skrobot.__file__),
                            'data', 'kuka_description', 'kuka.urdf')
        cls.robot = RobotModelFromURDF(urdf_file=path)
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_only_links_near_the_object_are_decomposed(self):
        try:
            import coacd  # noqa: F401
        except ImportError:
            self.skipTest('coacd not installed')
        r = self.robot
        r.angle_vector(np.zeros(len(r.joint_list)))
        ee = r.link_list[-1]
        target = ee.worldpos()               # object at the end-effector
        out = os.path.join(self.tmpdir, 'kuka_task.usdc')
        stage = export_for_task(r, [r.angle_vector()], target, out,
                                threshold=0.15)
        self.assertTrue(os.path.exists(out))
        decomposed = {p.GetParent().GetName() for p in stage.Traverse()
                      if p.HasAPI(UsdPhysics.CollisionAPI)
                      and '_part_' in p.GetName()}
        # the base link is far from the target and must NOT be decomposed
        self.assertNotIn(r.link_list[0].name, decomposed)

    def test_no_object_contact_decomposes_nothing(self):
        r = self.robot
        r.angle_vector(np.zeros(len(r.joint_list)))
        out = os.path.join(self.tmpdir, 'kuka_far.usdc')
        stage = export_for_task(r, [r.angle_vector()], [100.0, 100.0, 100.0],
                                out, threshold=0.05)
        parts = [p for p in stage.Traverse()
                 if p.HasAPI(UsdPhysics.CollisionAPI)
                 and '_part_' in p.GetName()]
        self.assertEqual(parts, [])


if __name__ == '__main__':
    unittest.main()

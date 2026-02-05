import unittest
from unittest.mock import MagicMock

import numpy as np
from numpy import testing

import skrobot


class TestManipulabilityEllipseComputation(unittest.TestCase):
    """Test manipulability computation without Viser server."""

    @classmethod
    def setUpClass(cls):
        cls.robot = skrobot.models.Kuka()
        cls.robot.reset_manip_pose()
        cls.link_list = cls.robot.rarm.link_list
        cls.end_link = cls.robot.rarm_end_coords.parent

    def test_jacobian_computation(self):
        """Test that Jacobian is computed correctly."""
        # Compute Jacobian using robot model
        jacobian = self.robot.calc_jacobian_from_link_list(
            move_target=self.end_link,
            link_list=self.link_list,
            position_mask=[True, True, True],
            rotation_mask=[False, False, False],
        )

        # Jacobian should be 3 x n_joints
        n_joints = len(self.link_list)
        self.assertEqual(jacobian.shape, (3, n_joints))

        # Jacobian should not be all zeros
        self.assertFalse(np.allclose(jacobian, 0))

    def test_manipulability_computation(self):
        """Test manipulability measure computation."""
        jacobian = self.robot.calc_jacobian_from_link_list(
            move_target=self.end_link,
            link_list=self.link_list,
            position_mask=[True, True, True],
            rotation_mask=[False, False, False],
        )

        # Compute Yoshikawa manipulability
        JJT = jacobian @ jacobian.T
        det_JJT = np.linalg.det(JJT)
        manipulability = np.sqrt(max(0.0, det_JJT))

        # Manipulability should be positive for non-singular configuration
        self.assertGreater(manipulability, 0)

    def test_eigendecomposition_for_ellipse(self):
        """Test eigendecomposition for ellipsoid visualization."""
        jacobian = self.robot.calc_jacobian_from_link_list(
            move_target=self.end_link,
            link_list=self.link_list,
            position_mask=[True, True, True],
            rotation_mask=[False, False, False],
        )

        JJT = jacobian @ jacobian.T
        vals, vecs = np.linalg.eigh(JJT)

        # Eigenvalues should all be non-negative
        self.assertTrue(np.all(vals >= -1e-10))

        # Eigenvectors should be orthonormal
        testing.assert_almost_equal(
            vecs @ vecs.T, np.eye(3), decimal=6
        )

    def test_manipulability_changes_with_config(self):
        """Test that manipulability changes with joint configuration."""
        # Store initial manipulability
        jacobian1 = self.robot.calc_jacobian_from_link_list(
            move_target=self.end_link,
            link_list=self.link_list,
            position_mask=[True, True, True],
            rotation_mask=[False, False, False],
        )
        JJT1 = jacobian1 @ jacobian1.T
        manip1 = np.sqrt(max(0.0, np.linalg.det(JJT1)))

        # Change configuration
        for link in self.link_list:
            current = link.joint.joint_angle()
            link.joint.joint_angle(current + 0.3)

        # Compute new manipulability
        jacobian2 = self.robot.calc_jacobian_from_link_list(
            move_target=self.end_link,
            link_list=self.link_list,
            position_mask=[True, True, True],
            rotation_mask=[False, False, False],
        )
        JJT2 = jacobian2 @ jacobian2.T
        manip2 = np.sqrt(max(0.0, np.linalg.det(JJT2)))

        # Manipulability should be different
        self.assertNotAlmostEqual(manip1, manip2, places=5)

        # Reset configuration
        self.robot.reset_manip_pose()


class TestManipulabilityEllipseClass(unittest.TestCase):
    """Test ManipulabilityEllipse class with mocked Viser server."""

    @classmethod
    def setUpClass(cls):
        cls.robot = skrobot.models.Kuka()
        cls.robot.reset_manip_pose()
        cls.link_list = cls.robot.rarm.link_list
        cls.end_link = cls.robot.rarm_end_coords.parent

    def _create_mock_server(self):
        """Create a mock Viser server."""
        mock_server = MagicMock()
        mock_mesh_handle = MagicMock()
        mock_mesh_handle.visible = True
        mock_server.scene.add_mesh_simple.return_value = mock_mesh_handle
        return mock_server, mock_mesh_handle

    def test_initialization(self):
        """Test ManipulabilityEllipse initialization."""
        from skrobot.viewers._manipulability_ellipse import ManipulabilityEllipse

        mock_server, _ = self._create_mock_server()

        ellipse = ManipulabilityEllipse(
            server=mock_server,
            robot_model=self.robot,
            link_list=self.link_list,
            target_link=self.end_link,
        )

        self.assertEqual(ellipse.manipulability, 0.0)
        self.assertIsNotNone(ellipse._mesh_handle)

    def test_update(self):
        """Test ManipulabilityEllipse update."""
        from skrobot.viewers._manipulability_ellipse import ManipulabilityEllipse

        mock_server, mock_mesh = self._create_mock_server()

        ellipse = ManipulabilityEllipse(
            server=mock_server,
            robot_model=self.robot,
            link_list=self.link_list,
            target_link=self.end_link,
            visible=True,
        )

        ellipse.update()

        # Manipulability should be computed
        self.assertGreater(ellipse.manipulability, 0)

        # Mesh vertices should be updated
        self.assertIsNotNone(mock_mesh.vertices)

    def test_set_visibility(self):
        """Test visibility setting."""
        from skrobot.viewers._manipulability_ellipse import ManipulabilityEllipse

        mock_server, mock_mesh = self._create_mock_server()

        ellipse = ManipulabilityEllipse(
            server=mock_server,
            robot_model=self.robot,
            link_list=self.link_list,
            target_link=self.end_link,
            visible=False,
        )

        # Initially not visible
        self.assertFalse(ellipse._visible)

        # Set visible
        ellipse.set_visibility(True)
        self.assertTrue(ellipse._visible)

        # Set invisible
        ellipse.set_visibility(False)
        self.assertFalse(ellipse._visible)

    def test_set_target_link(self):
        """Test setting target link."""
        from skrobot.viewers._manipulability_ellipse import ManipulabilityEllipse

        mock_server, _ = self._create_mock_server()

        ellipse = ManipulabilityEllipse(
            server=mock_server,
            robot_model=self.robot,
            link_list=self.link_list,
            target_link=None,
        )

        # Initially no target
        self.assertIsNone(ellipse._target_link)

        # Set target link
        ellipse.set_target_link(self.end_link)
        self.assertEqual(ellipse._target_link, self.end_link)

        # Clear target link
        ellipse.set_target_link(None)
        self.assertIsNone(ellipse._target_link)

    def test_set_scaling_factor(self):
        """Test scaling factor setting."""
        from skrobot.viewers._manipulability_ellipse import ManipulabilityEllipse

        mock_server, _ = self._create_mock_server()

        ellipse = ManipulabilityEllipse(
            server=mock_server,
            robot_model=self.robot,
            link_list=self.link_list,
            target_link=self.end_link,
            scaling_factor=0.2,
        )

        self.assertEqual(ellipse._scaling_factor, 0.2)

        ellipse.set_scaling_factor(0.5)
        self.assertEqual(ellipse._scaling_factor, 0.5)

    def test_remove(self):
        """Test removing ellipse from scene."""
        from skrobot.viewers._manipulability_ellipse import ManipulabilityEllipse

        mock_server, mock_mesh = self._create_mock_server()

        ellipse = ManipulabilityEllipse(
            server=mock_server,
            robot_model=self.robot,
            link_list=self.link_list,
            target_link=self.end_link,
        )

        ellipse.remove()

        mock_mesh.remove.assert_called_once()
        self.assertIsNone(ellipse._mesh_handle)
        self.assertIsNone(ellipse._target_link)


if __name__ == '__main__':
    unittest.main()

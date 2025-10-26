import unittest

import numpy as np

import skrobot


class TestJupyterNotebookViewer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.viewer = skrobot.viewers.JupyterNotebookViewer(height=500)
        self.robot = skrobot.models.Kuka()

    def test_viewer_creation(self):
        """Test that viewer can be created."""
        self.assertIsNotNone(self.viewer)
        self.assertEqual(self.viewer.height, 500)
        self.assertIsNotNone(self.viewer.scene)

    def test_add_robot(self):
        """Test adding a robot to the viewer."""
        initial_geometry_count = len(self.viewer.scene.geometry)
        self.viewer.add(self.robot)
        # Robot should add geometry to the scene
        self.assertGreater(len(self.viewer.scene.geometry),
                           initial_geometry_count)

    def test_add_link(self):
        """Test adding a single link to the viewer."""
        link = self.robot.link_list[0]
        self.viewer.add(link)
        # Check that link was added
        self.assertGreater(len(self.viewer._links), 0)

    def test_add_primitive(self):
        """Test adding a primitive geometry to the viewer."""
        box = skrobot.model.Box(
            extents=(0.1, 0.1, 0.1),
            face_colors=(1., 0, 0)
        )
        self.viewer.add(box)
        # Box should be added to the scene
        self.assertGreater(len(self.viewer.scene.geometry), 0)

    def test_delete_robot(self):
        """Test deleting a robot from the viewer."""
        self.viewer.add(self.robot)
        geometry_count_with_robot = len(self.viewer.scene.geometry)
        self.viewer.delete(self.robot)
        # Geometry should be removed
        self.assertLess(len(self.viewer.scene.geometry),
                        geometry_count_with_robot)

    def test_set_camera(self):
        """Test setting camera parameters."""
        # Test setting angles
        angles = [np.deg2rad(45), np.deg2rad(30), np.deg2rad(60)]
        self.viewer.set_camera(angles=angles)
        # Camera should be updated (no error should be raised)

        # Test setting distance
        self.viewer.set_camera(distance=5.0)

    def test_to_html(self):
        """Test converting scene to HTML."""
        self.viewer.add(self.robot)
        html = self.viewer.to_html()

        # HTML should be a non-empty string
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 0)

        # HTML should contain expected elements
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('three', html.lower())

    def test_to_html_escape_quotes(self):
        """Test HTML generation with escaped quotes."""
        self.viewer.add(self.robot)
        html = self.viewer.to_html(escape_quotes=True)

        # HTML should be a non-empty string
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 0)

    def test_redraw(self):
        """Test redrawing the scene."""
        self.viewer.add(self.robot)

        # Modify robot pose
        self.robot.reset_manip_pose()

        # Redraw should update transforms without error
        self.viewer.redraw()

    def test_is_active_property(self):
        """Test is_active property."""
        # JupyterNotebookViewer should always be active
        self.assertTrue(self.viewer.is_active)

    def test_has_exit_property(self):
        """Test has_exit property."""
        # JupyterNotebookViewer should never have exit
        self.assertFalse(self.viewer.has_exit)

    def test_save_image_not_implemented(self):
        """Test that save_image raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.viewer.save_image('test.png')

    def test_multiple_robots(self):
        """Test adding multiple robots to the viewer."""
        robot1 = skrobot.models.Kuka()
        robot2 = skrobot.models.Panda()

        self.viewer.add(robot1)
        geometry_count_1 = len(self.viewer.scene.geometry)

        self.viewer.add(robot2)
        geometry_count_2 = len(self.viewer.scene.geometry)

        # Second robot should add more geometry
        self.assertGreater(geometry_count_2, geometry_count_1)

    def test_scene_with_primitives(self):
        """Test creating a scene with both robot and primitives."""
        # Add plane
        plane = skrobot.model.Box(
            extents=(2, 2, 0.01),
            face_colors=(0.75, 0.75, 0.75)
        )
        self.viewer.add(plane)

        # Add robot
        self.viewer.add(self.robot)

        # Add target box
        box = skrobot.model.Box(
            extents=(0.05, 0.05, 0.05),
            face_colors=(1., 0, 0)
        )
        box.translate((0.5, 0, 0.3))
        self.viewer.add(box)

        # All should be in the scene
        self.assertGreater(len(self.viewer.scene.geometry), 2)

        # HTML should be generated successfully
        html = self.viewer.to_html()
        self.assertGreater(len(html), 0)

    def test_update_method(self):
        """Test the update convenience method."""
        self.viewer.add(self.robot)

        # Modify robot pose
        self.robot.reset_manip_pose()

        # Update should work without error
        self.viewer.update()

        # Update with update_in_place
        self.robot.init_pose()
        self.viewer.update(update_in_place=True)

    def test_display_id_tracking(self):
        """Test that display_id is created and tracked."""
        self.viewer.add(self.robot)

        # Initially no display_id
        self.assertIsNone(self.viewer._display_id)

        # After show() in non-IPython environment, returns HTML
        result = self.viewer.show()
        # In testing environment without IPython, should return HTML
        if result is not None:
            self.assertIsInstance(result, str)

    def test_show_with_update_in_place(self):
        """Test show() with update_in_place parameter."""
        self.viewer.add(self.robot)

        # First show
        self.viewer.show()

        # Second show without update_in_place (default)
        # Should create new display
        self.viewer.show(update_in_place=False)

        # Third show with update_in_place
        # Should update existing display
        self.viewer.show(update_in_place=True)

    def test_automatic_camera_setup(self):
        """Test that camera is automatically set on show()."""
        self.viewer.add(self.robot)

        # Initially, camera should not be marked as set by user
        self.assertFalse(self.viewer._camera_set_by_user)

        # Get initial camera position
        initial_camera = self.viewer.scene.camera_transform.copy()

        # Call show() - camera should be auto-set
        self.viewer.show()

        # Camera should be repositioned
        auto_camera = self.viewer.scene.camera_transform
        self.assertFalse(np.allclose(initial_camera, auto_camera))

        # Should still not be marked as user-set
        self.assertFalse(self.viewer._camera_set_by_user)

    def test_user_camera_not_overridden(self):
        """Test that user-set camera is not overridden."""
        self.viewer.add(self.robot)

        # User explicitly sets camera
        custom_angles = [np.deg2rad(30), np.deg2rad(30), np.deg2rad(90)]
        self.viewer.set_camera(angles=custom_angles)

        # Should be marked as user-set
        self.assertTrue(self.viewer._camera_set_by_user)

        # Get camera position after user set it
        user_camera = self.viewer.scene.camera_transform.copy()

        # Call show() - camera should NOT be auto-set
        self.viewer.show()

        # Camera should remain unchanged
        after_show_camera = self.viewer.scene.camera_transform
        self.assertTrue(np.allclose(user_camera, after_show_camera))


if __name__ == '__main__':
    unittest.main()

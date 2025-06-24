"""Test for XML-based URDF root link changer."""

import os
import sys
import tempfile
import unittest

import pytest

from skrobot.data import kuka_urdfpath
from skrobot.urdf import URDFXMLRootLinkChanger


@pytest.mark.skipif(
    sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
    reason="Skip in Python 2 and Python 3.6")
class TestURDFXMLRootLinkChanger(unittest.TestCase):

    def setUp(self):
        self.urdf_path = kuka_urdfpath()
        self.changer = URDFXMLRootLinkChanger(self.urdf_path)

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init_with_valid_urdf_path(self):
        changer = URDFXMLRootLinkChanger(self.urdf_path)
        self.assertIsNotNone(changer.tree)
        self.assertIsNotNone(changer.root)
        self.assertIsInstance(changer.links, dict)
        self.assertIsInstance(changer.joints, dict)
        self.assertIsInstance(changer.joint_tree, dict)

    def test_init_with_invalid_urdf_path(self):
        with self.assertRaises(FileNotFoundError):
            URDFXMLRootLinkChanger("non_existent_file.urdf")

    def test_list_links(self):
        links = self.changer.list_links()
        self.assertIsInstance(links, list)
        self.assertGreater(len(links), 0)

        for link in links:
            self.assertIsInstance(link, str)

    def test_get_current_root_link(self):
        root_link = self.changer.get_current_root_link()
        self.assertIsInstance(root_link, str)
        self.assertIn(root_link, self.changer.list_links())

    def test_change_root_link_invalid_name(self):
        output_path = os.path.join(self.temp_dir, "test_invalid.urdf")
        with self.assertRaises(ValueError):
            self.changer.change_root_link("non_existent_link", output_path)

    def test_change_root_link_valid_name(self):
        links = self.changer.list_links()
        if len(links) > 1:
            current_root = self.changer.get_current_root_link()
            new_root_candidates = [link for link in links
                                   if link != current_root]

            if new_root_candidates:
                new_root = new_root_candidates[0]
                output_path = os.path.join(self.temp_dir, "test_changed.urdf")

                self.changer.change_root_link(new_root, output_path)

                self.assertTrue(os.path.exists(output_path))

                new_changer = URDFXMLRootLinkChanger(output_path)
                actual_new_root = new_changer.get_current_root_link()
                self.assertEqual(actual_new_root, new_root)

    def test_change_root_link_same_as_current(self):
        current_root = self.changer.get_current_root_link()
        output_path = os.path.join(self.temp_dir, "test_same.urdf")

        self.changer.change_root_link(current_root, output_path)

        self.assertTrue(os.path.exists(output_path))

        new_changer = URDFXMLRootLinkChanger(output_path)
        actual_root = new_changer.get_current_root_link()
        self.assertEqual(actual_root, current_root)

    def test_find_path_to_link(self):
        current_root = self.changer.get_current_root_link()
        links = self.changer.list_links()

        if len(links) > 1:
            target_link = None
            for link in links:
                if link != current_root:
                    target_link = link
                    break

            if target_link:
                path = self.changer._find_path_to_link(
                    current_root, target_link)
                self.assertIsInstance(path, list)
                # Path should contain tuples of (parent, child, joint)
                for item in path:
                    self.assertIsInstance(item, tuple)
                    self.assertEqual(len(item), 3)


if __name__ == '__main__':
    unittest.main()

import hashlib
import os
import tempfile
import unittest

from skrobot.data import fetch_urdfpath
from skrobot.data import kuka_urdfpath
from skrobot.data import panda_urdfpath
from skrobot.data import pr2_urdfpath
from skrobot.urdf.hash import get_file_hash
from skrobot.urdf.hash import get_urdf_hash


class TestHashFunctions(unittest.TestCase):
    """Test cases for URDF hash functions."""

    def test_get_file_hash(self):
        """Test basic file hash calculation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('test content')
            temp_path = f.name

        try:
            hash_value = get_file_hash(temp_path)
            expected = hashlib.sha256(b'test content').hexdigest()
            self.assertEqual(hash_value, expected)
        finally:
            os.unlink(temp_path)

        # Test non-existent file
        hash_value = get_file_hash('/nonexistent/file/path')
        self.assertEqual(hash_value, 'file_not_found')

    def test_get_urdf_hash_simple(self):
        """Test URDF hash calculation with simple URDF."""
        urdf_content = '''<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </visual>
  </link>
</robot>'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf',
                                         delete=False) as f:
            f.write(urdf_content)
            urdf_path = f.name

        try:
            hash_value = get_urdf_hash(urdf_path)
            self.assertIsInstance(hash_value, str)
            self.assertEqual(len(hash_value), 64)  # SHA-256 hash length
        finally:
            os.unlink(urdf_path)

    def test_get_urdf_hash_kuka(self):
        """Test URDF hash calculation with KUKA robot."""
        urdf_path = kuka_urdfpath()
        hash_value = get_urdf_hash(urdf_path)

        # Basic checks
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 64)

        # Hash should be consistent for the same file
        hash_value2 = get_urdf_hash(urdf_path)
        self.assertEqual(hash_value, hash_value2)

    def test_get_urdf_hash_pr2(self):
        """Test URDF hash calculation with PR2 robot."""
        urdf_path = pr2_urdfpath()
        hash_value = get_urdf_hash(urdf_path)

        # Basic checks
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 64)

        # Hash should be consistent
        hash_value2 = get_urdf_hash(urdf_path)
        self.assertEqual(hash_value, hash_value2)

    def test_get_urdf_hash_fetch(self):
        """Test URDF hash calculation with Fetch robot."""
        urdf_path = fetch_urdfpath()
        hash_value = get_urdf_hash(urdf_path)

        # Basic checks
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 64)

        # Hash should be consistent
        hash_value2 = get_urdf_hash(urdf_path)
        self.assertEqual(hash_value, hash_value2)

    def test_get_urdf_hash_panda(self):
        """Test URDF hash calculation with Panda robot."""
        urdf_path = panda_urdfpath()
        hash_value = get_urdf_hash(urdf_path)

        # Basic checks
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 64)

        # Hash should be consistent
        hash_value2 = get_urdf_hash(urdf_path)
        self.assertEqual(hash_value, hash_value2)


if __name__ == '__main__':
    unittest.main()

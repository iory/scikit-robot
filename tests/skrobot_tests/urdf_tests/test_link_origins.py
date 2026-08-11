import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from skrobot.coordinates.math import rotation_matrix
from skrobot.coordinates.math import rpy2matrix
from skrobot.urdf import normalize_link_origins
from skrobot.urdf import normalize_link_origins_file


def _origin_matrix(elem):
    """4x4 transform of ``elem``'s ``<origin>``, identity when absent."""
    matrix = np.eye(4)
    if elem is None:
        return matrix
    origin = elem.find('origin')
    if origin is None:
        return matrix
    if 'xyz' in origin.attrib:
        matrix[:3, 3] = [float(v) for v in origin.get('xyz').split()]
    if 'rpy' in origin.attrib:
        roll, pitch, yaw = [float(v) for v in origin.get('rpy').split()]
        matrix[:3, :3] = rpy2matrix(roll, pitch, yaw)
    return matrix


def _joint_motion(joint, value):
    """4x4 transform a joint applies at configuration ``value``."""
    matrix = np.eye(4)
    axis_elem = joint.find('axis')
    if axis_elem is None:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = np.array([float(v) for v in axis_elem.get('xyz').split()])
    axis = axis / np.linalg.norm(axis)
    joint_type = joint.get('type')
    if joint_type in ('revolute', 'continuous'):
        matrix[:3, :3] = rotation_matrix(value, axis)
    elif joint_type == 'prismatic':
        matrix[:3, 3] = axis * value
    return matrix


def world_poses(root, joint_values=None):
    """World pose of every frame in a URDF at a given configuration.

    Independent forward kinematics used as the reference: link frames plus
    every ``<inertial>`` / ``<visual>`` / ``<collision>`` frame, keyed by
    ``'<link>/<tag>/<index>'``.
    """
    joint_values = joint_values or {}
    links = {link.get('name'): link for link in root.findall('link')}
    children = {}
    has_parent = set()
    for joint in root.findall('joint'):
        parent = joint.find('parent').get('link')
        children.setdefault(parent, []).append(joint)
        has_parent.add(joint.find('child').get('link'))

    poses = {}
    stack = [(name, np.eye(4)) for name in links if name not in has_parent]
    while stack:
        name, matrix = stack.pop()
        poses[name] = matrix
        for joint in children.get(name, []):
            value = joint_values.get(joint.get('name'), 0.0)
            stack.append((
                joint.find('child').get('link'),
                matrix.dot(_origin_matrix(joint)).dot(
                    _joint_motion(joint, value))))
    for name, link in links.items():
        for tag in ('inertial', 'visual', 'collision'):
            for i, elem in enumerate(link.findall(tag)):
                poses['{}/{}/{}'.format(name, tag, i)] = \
                    poses[name].dot(_origin_matrix(elem))
    return poses


def max_pose_difference(before, after):
    """Largest absolute entry-wise difference between two pose dicts."""
    assert set(before) == set(after)
    return max(float(np.max(np.abs(before[k] - after[k]))) for k in before)


def expected_poses(before, shifts):
    """The poses a correct re-centring must produce.

    Every frame stays exactly where it was, except the re-centred link
    frames themselves, which move to ``pose @ T(d)`` -- that displacement
    is the whole point of the operation.
    """
    expected = dict(before)
    for name, shift in shifts.items():
        matrix = np.eye(4)
        matrix[:3, 3] = shift
        expected[name] = before[name].dot(matrix)
    return expected


def mesh_centroids(mapping):
    """A ``centroid_fn`` answering from ``{mesh filename: centroid}``."""
    def centroid_fn(geometry):
        mesh = geometry.find('mesh')
        if mesh is None:
            return None
        value = mapping.get(mesh.get('filename'))
        if value is None:
            return None
        return np.array(value, dtype=np.float64)
    return centroid_fn


BRANCHING = """<robot name="branching">
  <link name="base">
    <visual>
      <origin xyz="0.02 0 0.03" rpy="0 0 0"/>
      <geometry><mesh filename="base.stl"/></geometry>
    </visual>
  </link>
  <joint name="base_to_mid" type="fixed">
    <parent link="base"/>
    <child link="mid"/>
    <origin xyz="0.1 0.2 0.3" rpy="0.3 -0.2 0.5"/>
  </joint>
  <link name="mid">
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.3"/>
    </inertial>
    <visual>
      <origin xyz="0.9 -0.4 1.2" rpy="0.1 0.2 -0.3"/>
      <geometry><mesh filename="mid.stl"/></geometry>
    </visual>
    <collision>
      <origin xyz="0.9 -0.4 1.2" rpy="0.1 0.2 -0.3"/>
      <geometry><mesh filename="mid.stl"/></geometry>
    </collision>
  </link>
  <joint name="mid_to_hinge" type="revolute">
    <parent link="mid"/>
    <child link="hinge"/>
    <origin xyz="1.0 -0.5 1.1" rpy="0 0.4 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2" upper="2" effort="10" velocity="1"/>
  </joint>
  <link name="hinge">
    <visual>
      <geometry><mesh filename="hinge.stl"/></geometry>
    </visual>
  </link>
  <joint name="mid_to_slider" type="prismatic">
    <parent link="mid"/>
    <child link="slider"/>
    <origin xyz="0.8 -0.3 1.3" rpy="-0.2 0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <link name="slider"/>
  <joint name="mid_to_pad" type="fixed">
    <parent link="mid"/>
    <child link="pad"/>
    <origin xyz="0.95 -0.45 1.25" rpy="0 0 0.7"/>
  </joint>
  <link name="pad">
    <visual>
      <geometry><mesh filename="pad.stl"/></geometry>
    </visual>
  </link>
</robot>
"""

CONFIGURATION = {'mid_to_hinge': 0.73, 'mid_to_slider': -0.41}


class TestNormalizeLinkOrigins(unittest.TestCase):

    def _round_trip(self, urdf, centroids, joint_values=None,
                    **kwargs):
        """Normalize ``urdf`` and return ``(root, shifts, max_difference)``.

        ``max_difference`` compares the forward kinematics of the whole
        model, every frame included, against :func:`expected_poses`: it is
        the amount by which something moved that should not have.
        """
        root = ET.fromstring(urdf)
        joint_values = joint_values or {}
        before = world_poses(root, joint_values)
        shifts = normalize_link_origins(
            root, centroid_fn=mesh_centroids(centroids), **kwargs)
        after = world_poses(root, joint_values)
        # nothing but the re-centred link frames may have moved
        for name in set(before) - set(shifts):
            np.testing.assert_allclose(
                before[name], after[name], atol=1e-12,
                err_msg='{} moved'.format(name))
        return root, shifts, max_pose_difference(
            expected_poses(before, shifts), after)

    def test_link_with_parent_and_several_children(self):
        centroids = {
            'base.stl': [0.0, 0.0, 0.0],
            'mid.stl': [0.05, -0.02, 0.01],
            'hinge.stl': [0.0, 0.0, 0.0],
            'pad.stl': [0.0, 0.0, 0.0],
        }
        root, shifts, difference = self._round_trip(
            BRANCHING, centroids, CONFIGURATION)

        # 'mid' is the only fixed link whose geometry is off its frame.
        self.assertEqual(set(shifts), {'mid'})
        expected = rpy2matrix(0.1, 0.2, -0.3).dot(
            np.array(centroids['mid.stl'])) + np.array([0.9, -0.4, 1.2])
        np.testing.assert_allclose(shifts['mid'], expected, atol=1e-15)
        self.assertLess(difference, 1e-12)

        # the frame really did move onto the geometry
        visual = root.find(".//link[@name='mid']/visual")
        residual = rpy2matrix(0.1, 0.2, -0.3).dot(
            np.array(centroids['mid.stl'])) \
            + np.array([float(v) for v
                        in visual.find('origin').get('xyz').split()])
        self.assertLess(np.linalg.norm(residual), 1e-14)

        # a pure translation never touches rpy
        joint = root.find(".//joint[@name='base_to_mid']/origin")
        self.assertEqual(joint.get('rpy'), '0.3 -0.2 0.5')
        self.assertEqual(visual.find('origin').get('rpy'), '0.1 0.2 -0.3')

    def test_link_with_parent_and_several_children_many_configurations(self):
        centroids = {
            'base.stl': [0.0, 0.0, 0.0],
            'mid.stl': [0.05, -0.02, 0.01],
            'hinge.stl': [0.0, 0.0, 0.0],
            'pad.stl': [0.0, 0.0, 0.0],
        }
        worst = 0.0
        for hinge, slider in [(0.0, 0.0), (1.4, 0.9), (-2.0, -1.0)]:
            _, _, difference = self._round_trip(
                BRANCHING, centroids,
                {'mid_to_hinge': hinge, 'mid_to_slider': slider})
            worst = max(worst, difference)
        self.assertLess(worst, 1e-12)

    def test_root_link_is_left_alone(self):
        urdf = """<robot name="rooted">
  <link name="base">
    <visual>
      <origin xyz="1.5 -0.5 0.25" rpy="0 0 0"/>
      <geometry><mesh filename="base.stl"/></geometry>
    </visual>
  </link>
  <joint name="base_to_tip" type="fixed">
    <parent link="base"/>
    <child link="tip"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
  </joint>
  <link name="tip"/>
</robot>
"""
        # The root has no parent joint to absorb T(+d), so re-centring it
        # would move the whole robot in the world.
        root, shifts, difference = self._round_trip(
            urdf, {'base.stl': [0.0, 0.0, 0.0]})
        self.assertEqual(shifts, {})
        self.assertEqual(difference, 0.0)
        origin = root.find(".//link[@name='base']/visual/origin")
        self.assertEqual(origin.get('xyz'), '1.5 -0.5 0.25')

    def test_inertial_origin_already_non_zero(self):
        urdf = """<robot name="inertial">
  <link name="base"/>
  <joint name="base_to_body" type="fixed">
    <parent link="base"/>
    <child link="body"/>
    <origin xyz="0 0 0.5" rpy="0 0 1.2"/>
  </joint>
  <link name="body">
    <inertial>
      <origin xyz="0.7 -0.25 0.4" rpy="0.15 -0.35 0.55"/>
      <mass value="3"/>
      <inertia ixx="0.1" ixy="0.01" ixz="0.02" iyy="0.2" iyz="0.03"
               izz="0.3"/>
    </inertial>
    <visual>
      <origin xyz="0.75 -0.2 0.45" rpy="0 0 0"/>
      <geometry><mesh filename="body.stl"/></geometry>
    </visual>
  </link>
</robot>
"""
        root, shifts, difference = self._round_trip(
            urdf, {'body.stl': [-0.05, 0.0, 0.02]})
        np.testing.assert_allclose(
            shifts['body'], [0.7, -0.2, 0.47], atol=1e-15)
        self.assertLess(difference, 1e-12)

        inertial = root.find(".//link[@name='body']/inertial/origin")
        np.testing.assert_allclose(
            [float(v) for v in inertial.get('xyz').split()],
            [0.0, -0.05, -0.07], atol=1e-15)
        # the inertia tensor is expressed in the inertial frame, whose
        # orientation is untouched, so it must not be rewritten
        self.assertEqual(inertial.get('rpy'), '0.15 -0.35 0.55')
        inertia = root.find(".//link[@name='body']/inertial/inertia")
        self.assertEqual(inertia.get('ixy'), '0.01')

    def test_already_centred_geometry_is_a_noop(self):
        urdf = """<robot name="centred">
  <link name="base"/>
  <joint name="base_to_body" type="fixed">
    <parent link="base"/>
    <child link="body"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>
  <link name="body">
    <visual>
      <geometry><mesh filename="body.stl"/></geometry>
    </visual>
  </link>
</robot>
"""
        root = ET.fromstring(urdf)
        original = ET.tostring(root)
        shifts = normalize_link_origins(
            root, centroid_fn=mesh_centroids({'body.stl': [0.0, 0.0, 0.0]}))
        self.assertEqual(shifts, {})
        self.assertEqual(ET.tostring(root), original)

    def test_shift_within_tolerance_is_a_noop(self):
        urdf = """<robot name="nearly">
  <link name="base"/>
  <joint name="base_to_body" type="fixed">
    <parent link="base"/>
    <child link="body"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>
  <link name="body">
    <visual>
      <geometry><mesh filename="body.stl"/></geometry>
    </visual>
  </link>
</robot>
"""
        root = ET.fromstring(urdf)
        original = ET.tostring(root)
        shifts = normalize_link_origins(
            root,
            centroid_fn=mesh_centroids({'body.stl': [1e-11, 0.0, 0.0]}))
        self.assertEqual(shifts, {})
        self.assertEqual(ET.tostring(root), original)

    def test_several_visuals_use_their_common_centroid(self):
        urdf = """<robot name="multi">
  <link name="base"/>
  <joint name="base_to_body" type="fixed">
    <parent link="base"/>
    <child link="body"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <link name="body">
    <visual>
      <origin xyz="0.6 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="left.stl"/></geometry>
    </visual>
    <visual>
      <origin xyz="0 0.4 0" rpy="0 0 0"/>
      <geometry><mesh filename="right.stl"/></geometry>
    </visual>
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry><mesh filename="top.stl"/></geometry>
    </visual>
  </link>
</robot>
"""
        centroids = {'left.stl': [0.0, 0.0, 0.0],
                     'right.stl': [0.0, 0.0, 0.0],
                     'top.stl': [0.0, 0.0, 0.0]}
        _, shifts, difference = self._round_trip(urdf, centroids)
        np.testing.assert_allclose(
            shifts['body'], [0.2, 0.4 / 3.0, 0.2 / 3.0], atol=1e-15)
        # the common centroid is none of the three visual origins
        for xyz in ([0.6, 0, 0], [0, 0.4, 0], [0, 0, 0.2]):
            self.assertGreater(
                np.linalg.norm(shifts['body'] - np.array(xyz)), 0.1)
        self.assertLess(difference, 1e-12)

    def test_movable_parent_joint_is_skipped(self):
        urdf = """<robot name="movable">
  <link name="base"/>
  <joint name="base_to_body" type="revolute">
    <parent link="base"/>
    <child link="body"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <link name="body">
    <visual>
      <origin xyz="0.9 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="body.stl"/></geometry>
    </visual>
  </link>
</robot>
"""
        # Moving a movable link's frame would move its axis of motion, so
        # the model would only agree with itself at the zero configuration.
        root = ET.fromstring(urdf)
        shifts = normalize_link_origins(
            root, centroid_fn=mesh_centroids({'body.stl': [0.0, 0.0, 0.0]}))
        self.assertEqual(shifts, {})

    def test_link_names_restricts_the_operation(self):
        centroids = {
            'base.stl': [0.0, 0.0, 0.0],
            'mid.stl': [0.05, -0.02, 0.01],
            'hinge.stl': [0.0, 0.0, 0.0],
            'pad.stl': [0.3, 0.0, 0.0],
        }
        _, shifts, difference = self._round_trip(
            BRANCHING, centroids, CONFIGURATION)
        self.assertEqual(set(shifts), {'mid', 'pad'})
        self.assertLess(difference, 1e-12)

        _, shifts, difference = self._round_trip(
            BRANCHING, centroids, CONFIGURATION, link_names=['pad'])
        self.assertEqual(set(shifts), {'pad'})
        self.assertLess(difference, 1e-12)

    def test_unmeasurable_geometry_leaves_the_link_alone(self):
        root = ET.fromstring(BRANCHING)
        original = ET.tostring(root)
        shifts = normalize_link_origins(root, centroid_fn=lambda g: None)
        self.assertEqual(shifts, {})
        self.assertEqual(ET.tostring(root), original)

    def test_base_path_or_centroid_fn_is_required(self):
        root = ET.fromstring(BRANCHING)
        with self.assertRaises(ValueError):
            normalize_link_origins(root)


class TestNormalizeLinkOriginsFile(unittest.TestCase):

    URDF = """<robot name="primitives">
  <link name="base"/>
  <joint name="base_to_body" type="fixed">
    <parent link="base"/>
    <child link="body"/>
    <origin xyz="0 0 0.5" rpy="0 0 0.9"/>
  </joint>
  <link name="body">
    <visual>
      <origin xyz="1.2 -0.6 0.4" rpy="0 0 0"/>
      <geometry><box size="0.1 0.2 0.3"/></geometry>
    </visual>
    <collision>
      <origin xyz="1.2 -0.6 0.4" rpy="0 0 0"/>
      <geometry><box size="0.1 0.2 0.3"/></geometry>
    </collision>
  </link>
  <joint name="body_to_tip" type="fixed">
    <parent link="body"/>
    <child link="tip"/>
    <origin xyz="1.3 -0.6 0.4" rpy="0 0 0"/>
  </joint>
  <link name="tip"/>
</robot>
"""

    def test_primitive_geometry_and_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, 'robot.urdf')
            output_file = os.path.join(tmpdir, 'centred.urdf')
            with open(input_file, 'w') as f:
                f.write(self.URDF)

            before = world_poses(ET.fromstring(self.URDF))
            shifts = normalize_link_origins_file(input_file, output_file)
            after = world_poses(ET.parse(output_file).getroot())

        # a box is centred on its own frame, so the shift is the visual origin
        np.testing.assert_allclose(
            shifts['body'], [1.2, -0.6, 0.4], atol=1e-15)
        self.assertLess(
            max_pose_difference(expected_poses(before, shifts), after), 1e-12)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            normalize_link_origins_file('/no/such/robot.urdf')


class TestGeometryCentroid(unittest.TestCase):

    def test_mesh_on_disk(self):
        try:
            import trimesh
        except ImportError:
            self.skipTest('trimesh is not available')

        urdf = """<robot name="meshed">
  <link name="base"/>
  <joint name="base_to_body" type="fixed">
    <parent link="base"/>
    <child link="body"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>
  <link name="body">
    <visual>
      <geometry><mesh filename="body.stl" scale="2 2 2"/></geometry>
    </visual>
  </link>
</robot>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
            box.apply_translation([0.4, -0.15, 0.05])
            box.export(os.path.join(tmpdir, 'body.stl'))
            input_file = os.path.join(tmpdir, 'robot.urdf')
            with open(input_file, 'w') as f:
                f.write(urdf)

            before = world_poses(ET.fromstring(urdf))
            root = ET.parse(input_file).getroot()
            shifts = normalize_link_origins(root, base_path=tmpdir)
            after = world_poses(root)

        # the mesh scale multiplies the measured centroid
        np.testing.assert_allclose(
            shifts['body'], [0.8, -0.3, 0.1], atol=1e-6)
        self.assertLess(
            max_pose_difference(expected_poses(before, shifts), after), 1e-12)


class TestFramesThisCannotMove(unittest.TestCase):
    """A frame is only moved when everything expressed in it moves with it.

    Anything referencing the link from outside -- a Gazebo sensor pose, a
    top-level ``<sensor>`` -- is stated in that link's frame and is not
    rewritten here, so re-centring would silently displace it.  Likewise a
    malformed origin, or a link the document declares or parents twice.
    """

    _URDF = """<robot name="r">
      <link name="base"/>
      <link name="head">
        <visual><origin xyz="0.4 0 0.1"/>
          <geometry><box size="0.1 0.1 0.1"/></geometry></visual>
      </link>
      <joint name="j" type="fixed"><origin xyz="0 0 1"/>
        <parent link="base"/><child link="head"/></joint>
      {}</robot>"""

    def _run(self, extra=''):
        root = ET.fromstring(self._URDF.format(extra))
        shifts = normalize_link_origins(
            root, centroid_fn=lambda geometry: np.zeros(3))
        return root, shifts

    def test_a_plain_link_is_still_recentred(self):
        _root, shifts = self._run()
        np.testing.assert_allclose(shifts['head'], [0.4, 0.0, 0.1])

    def test_a_gazebo_sensor_on_the_link_stops_it(self):
        _root, shifts = self._run(
            '<gazebo reference="head"><sensor name="cam">'
            '<pose>0.5 0 0.12 0 0 0</pose></sensor></gazebo>')
        self.assertEqual(shifts, {})

    def test_a_sensor_parented_to_the_link_stops_it(self):
        _root, shifts = self._run(
            '<sensor name="s"><parent link="head"/>'
            '<origin xyz="0.45 0 0.15"/></sensor>')
        self.assertEqual(shifts, {})

    def test_an_empty_gazebo_tag_does_not_stop_it(self):
        """``<gazebo reference="L"/>`` carrying only material settings names
        no frame, so it must not block a re-centre."""
        _root, shifts = self._run('<gazebo reference="head"/>')
        np.testing.assert_allclose(shifts['head'], [0.4, 0.0, 0.1])

    def test_an_unparsable_origin_leaves_the_tree_untouched(self):
        root, shifts = self._run(
            '<link name="tip"/><joint name="k" type="fixed">'
            '<origin xyz="${tip} 0 0"/>'
            '<parent link="head"/><child link="tip"/></joint>')
        self.assertEqual(shifts, {})
        # the parent joint must not have been moved before the failure
        joint = next(j for j in root.findall('joint')
                     if j.get('name') == 'j')
        self.assertEqual(joint.find('origin').get('xyz'), '0 0 1')

    def test_a_link_declared_twice_is_left_alone(self):
        _root, shifts = self._run('<link name="head"/>')
        self.assertEqual(shifts, {})

    def test_a_link_with_two_parent_joints_is_left_alone(self):
        _root, shifts = self._run(
            '<joint name="j2" type="fixed"><origin xyz="1 0 0"/>'
            '<parent link="base"/><child link="head"/></joint>')
        self.assertEqual(shifts, {})


class TestToleranceIsAboveMeasurementNoise(unittest.TestCase):

    def _shift_for(self, centroid, **kwargs):
        root = ET.fromstring("""<robot name="r">
          <link name="base"/>
          <link name="c"><visual><geometry><box size="1 1 1"/></geometry>
            </visual></link>
          <joint name="j" type="fixed"><parent link="base"/>
            <child link="c"/></joint></robot>""")
        return normalize_link_origins(
            root, centroid_fn=lambda geometry: np.asarray(centroid, float),
            **kwargs)

    def test_a_nanometre_is_noise_and_is_ignored(self):
        """Measuring a centroid off a binary STL carries several nanometres
        of float32 error, so a nanometre shift must not churn the file."""
        self.assertEqual(self._shift_for([1e-9, 0.0, 0.0]), {})
        self.assertEqual(self._shift_for([1e-9 / np.sqrt(3)] * 3), {})

    def test_a_millimetre_is_real_and_is_applied(self):
        shifts = self._shift_for([1e-3, 0.0, 0.0])
        np.testing.assert_allclose(shifts['c'], [1e-3, 0.0, 0.0])

    def test_the_threshold_is_configurable(self):
        self.assertEqual(self._shift_for([1e-3, 0.0, 0.0], tolerance=1.0), {})


if __name__ == '__main__':
    unittest.main()

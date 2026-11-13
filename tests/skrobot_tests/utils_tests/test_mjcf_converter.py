import importlib
import os
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from skrobot.urdf import urdf_to_mjcf


# A CAD export split by material hands the converter three shapes that MuJoCo's
# mesh compiler treats very differently, so the fixture below carries one of
# each on a single link.
def _solid_box():
    return trimesh.creation.box(extents=(0.1, 0.1, 0.1))


def _flat_sheet():
    """Two coplanar triangles: 4 vertices, zero volume, no convex hull.

    This is what a stamped part (a servo horn, a shim) looks like once the
    exporter splits it out. MuJoCo needs ``inertia="shell"`` for it, and
    refuses it outright as a collision geom.
    """
    vertices = np.array([[0.0, 0.0, 0.0],
                         [0.1, 0.0, 0.0],
                         [0.1, 0.1, 0.0],
                         [0.0, 0.1, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _thin_shell():
    """A closed 0.3 x 0.3 x 0.003 mm shim: negligible volume, real convex hull.

    Distinct from _flat_sheet: this one is still usable as a collision geom, so
    it must survive with shell inertia rather than being dropped.
    """
    return trimesh.creation.box(extents=(3e-4, 3e-4, 3e-6))


def _single_triangle():
    """One triangle: 3 vertices. MuJoCo refuses it ("at least 4 vertices")."""
    vertices = np.array([[0.0, 0.0, 0.0],
                         [0.05, 0.0, 0.0],
                         [0.0, 0.05, 0.0]])
    return trimesh.Trimesh(vertices=vertices, faces=np.array([[0, 1, 2]]),
                           process=False)


def _doubled_triangle():
    """One triangle stored twice, front and back: 6 vertices, 3 distinct.

    A COLLADA export keeps the duplicates, so a raw vertex count says 6;
    MuJoCo merges coincident vertices first, sees 3, and refuses the mesh.
    """
    vertices = np.array([[0.0, 0.0, 0.0],
                         [0.05, 0.0, 0.0],
                         [0.0, 0.05, 0.0]] * 2)
    return trimesh.Trimesh(vertices=vertices,
                           faces=np.array([[0, 1, 2], [3, 5, 4]]),
                           process=False)


_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="degenerate">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh}"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh}"/></geometry>
    </collision>
  </link>
</robot>
"""


class TestDegenerateMeshes(unittest.TestCase):
    """A URDF whose meshes MuJoCo cannot compile as solids must still convert.

    Both shapes here come straight out of real CAD exports; before the
    converter handled them, either one made ``MjModel.from_xml_path`` raise and
    took the whole robot with it.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _convert(self, mesh, ext='stl'):
        # STL and OBJ loading merge coincident vertices; COLLADA keeps them
        name = 'part.' + ext
        mesh_path = os.path.join(self.tmpdir, name)
        mesh.export(mesh_path)
        urdf_path = os.path.join(self.tmpdir, 'robot.urdf')
        with open(urdf_path, 'w') as f:
            f.write(_URDF_TEMPLATE.format(mesh=name))
        out_path = os.path.join(self.tmpdir, 'robot.xml')
        urdf_to_mjcf(urdf_path, out_path, add_ground=False)
        return out_path, ET.parse(out_path).getroot()

    def _mesh_assets(self, root):
        asset = root.find('asset')
        return [] if asset is None else asset.findall('mesh')

    def test_solid_mesh_is_not_marked_shell(self):
        # <visual> and <collision> name the same file, which loads once,
        # so they share one asset; it is not a shell.
        _, root = self._convert(_solid_box())
        assets = self._mesh_assets(root)
        self.assertEqual(len(assets), 1)
        self.assertEqual([a.get('inertia') for a in assets], [None])

    def test_zero_volume_mesh_is_marked_shell(self):
        _, root = self._convert(_thin_shell())
        assets = self._mesh_assets(root)
        self.assertEqual(len(assets), 1)
        self.assertEqual([a.get('inertia') for a in assets], ['shell'])

    def test_three_vertex_mesh_is_dropped(self):
        # Too few vertices for MuJoCo, so no asset and no geom referencing one.
        _, root = self._convert(_single_triangle())
        self.assertEqual(len(self._mesh_assets(root)), 0)
        mesh_geoms = [g for g in root.iter('geom') if g.get('type') == 'mesh']
        self.assertEqual(mesh_geoms, [])

    def test_duplicated_vertices_do_not_count(self):
        # 6 stored vertices but only 3 distinct: dropped like a single triangle
        _, root = self._convert(_doubled_triangle(), ext='dae')
        self.assertEqual(len(self._mesh_assets(root)), 0)

    def test_coplanar_mesh_survives_as_visual_only(self):
        # It keeps its <visual> geom (group 2, no contact) but loses the
        # <collision> one (group 3), which MuJoCo could not have compiled.
        _, root = self._convert(_flat_sheet())
        groups = sorted(g.get('group') for g in root.iter('geom')
                        if g.get('type') == 'mesh')
        self.assertEqual(groups, ['2'])
        # the now-unreferenced collision asset is pruned too
        self.assertEqual(len(self._mesh_assets(root)), 1)

    def _require_mujoco(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest('mujoco is not installed')
            return None
        return mujoco

    def test_mujoco_loads_the_flat_sheet_model(self):
        mujoco = self._require_mujoco()
        out_path, _ = self._convert(_flat_sheet())
        model = mujoco.MjModel.from_xml_path(out_path)
        self.assertEqual(model.nbody, 2)  # world + base_link

    def test_mujoco_loads_the_thin_shell_model(self):
        mujoco = self._require_mujoco()
        out_path, _ = self._convert(_thin_shell())
        model = mujoco.MjModel.from_xml_path(out_path)
        self.assertEqual(model.nbody, 2)

    def test_mujoco_loads_the_doubled_triangle_model(self):
        mujoco = self._require_mujoco()
        out_path, _ = self._convert(_doubled_triangle(), ext='dae')
        model = mujoco.MjModel.from_xml_path(out_path)
        self.assertEqual(model.nbody, 2)

    def test_mujoco_loads_the_single_triangle_model(self):
        mujoco = self._require_mujoco()
        out_path, _ = self._convert(_single_triangle())
        model = mujoco.MjModel.from_xml_path(out_path)
        self.assertEqual(model.nbody, 2)


_LEGGED_URDF = """<?xml version="1.0"?>
<robot name="legged">
  <link name="base_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
  <joint name="hip" type="revolute">
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <parent link="base_link"/><child link="foot"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="3.0" velocity="6.0"/>
  </joint>
  <link name="foot">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.1"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <collision><origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry><box size="0.05 0.05 0.05"/></geometry></collision>
  </link>
</robot>
"""


class TestServoDerivedDamping(unittest.TestCase):
    """``joint_damping='backemf'`` takes each joint's damping from its own
    ``<limit>``, instead of putting one guessed number on every joint."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.urdf = os.path.join(self.tmpdir, 'robot.urdf')
        with open(self.urdf, 'w') as f:
            f.write(_LEGGED_URDF)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _convert(self, **kwargs):
        out_path = os.path.join(self.tmpdir, 'robot.xml')
        urdf_to_mjcf(self.urdf, out_path, **kwargs)
        return out_path, ET.parse(out_path).getroot()

    def _model_joints(self, root):
        # NOT root.iter('joint'): that also picks up <default><joint> and the
        # <equality><joint> a mimic produces, neither of which owns a DoF.
        return [j for body in root.iter('body') for j in body.findall('joint')]

    def test_backemf_damping_is_effort_over_velocity(self):
        # effort 3.0 / velocity 6.0 -> 0.5 N*m*s/rad
        _out, root = self._convert(joint_damping='backemf')
        joints = self._model_joints(root)
        self.assertEqual(len(joints), 1)
        self.assertAlmostEqual(float(joints[0].get('damping')), 0.5)

    def test_backemf_damping_writes_no_global_default(self):
        _out, root = self._convert(joint_damping='backemf')
        default = root.find('default')
        defaulted = [] if default is None else default.findall('joint')
        self.assertEqual([j.get('damping') for j in defaulted
                          if j.get('damping')], [])

    def test_a_number_still_goes_on_the_default(self):
        _out, root = self._convert(joint_damping=0.25)
        self.assertAlmostEqual(
            float(root.find('default').find('joint').get('damping')), 0.25)
        self.assertIsNone(self._model_joints(root)[0].get('damping'))

    def test_a_joint_without_usable_limits_is_left_alone(self):
        with open(self.urdf, 'w') as f:
            f.write(_LEGGED_URDF.replace(
                '<limit lower="-1" upper="1" effort="3.0" velocity="6.0"/>',
                '<limit lower="-1" upper="1" effort="3.0" velocity="0"/>'))
        _out, root = self._convert(joint_damping='backemf')
        self.assertIsNone(self._model_joints(root)[0].get('damping'))


class TestAutoHomeBaseHeight(unittest.TestCase):
    """``home_base_height='auto'`` measures the height at which the model rests
    on the floor, instead of the caller having to supply one."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.urdf = os.path.join(self.tmpdir, 'robot.urdf')
        with open(self.urdf, 'w') as f:
            f.write(_LEGGED_URDF)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _qpos(self, home=None, **kwargs):
        out_path = os.path.join(self.tmpdir, 'robot.xml')
        urdf_to_mjcf(self.urdf, out_path, floating_base=True,
                     home=home or {}, **kwargs)
        root = ET.parse(out_path).getroot()
        key = root.find('keyframe').find('key')
        self.assertEqual(key.get('name'), 'home')
        return [float(v) for v in key.get('qpos').split()]

    def test_auto_height_puts_the_lowest_geom_on_the_floor(self):
        # hip 0.2 m below the base, foot box another 0.1 m below that and
        # 0.05 m tall: its underside is 0.325 m down, exactly the height needed
        qpos = self._qpos(home_base_height='auto')
        self.assertAlmostEqual(qpos[2], 0.325, places=6)

    def test_an_explicit_height_is_used_as_given(self):
        self.assertAlmostEqual(self._qpos(home_base_height=0.5)[2], 0.5)

    def test_auto_height_follows_the_home_pose(self):
        # folding the hip to 90 deg swings the foot forward instead of down, so
        # the robot needs to start lower than with the leg hanging straight
        straight = self._qpos(home_base_height='auto')
        folded = self._qpos(home_base_height='auto',
                            home={'hip': np.pi / 2.0})
        self.assertLess(folded[2], straight[2])


_TWO_LINK_URDF = """<?xml version="1.0"?>
<robot name="hand">
  <link name="base_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <collision><geometry><mesh filename="part.glb"/></geometry></collision>
  </link>
  <joint name="finger_joint" type="revolute">
    <parent link="base_link"/><child link="finger"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1.0" velocity="1.0"/>
  </joint>
  <link name="finger">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.1"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <collision><geometry><mesh filename="part.glb"/></geometry></collision>
  </link>
</robot>
"""


class TestDecomposeLinks(unittest.TestCase):
    """``convex_decompose_links`` decomposes only the links it names, and a
    file's sub-meshes as one mesh rather than patch by patch."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Two boxes in one file: two sub-meshes, as a CAD export splits them.
        scene = trimesh.Scene([
            trimesh.creation.box(extents=(0.02, 0.02, 0.02)),
            trimesh.creation.box(extents=(0.02, 0.02, 0.02)).apply_translation(
                (0.05, 0.0, 0.0)),
        ])
        scene.export(os.path.join(self.tmpdir, 'part.glb'))
        self.urdf = os.path.join(self.tmpdir, 'hand.urdf')
        with open(self.urdf, 'w') as f:
            f.write(_TWO_LINK_URDF)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_only_named_link_is_decomposed_as_one_mesh(self):
        from unittest import mock

        calls = []

        def fake_decomposition(mesh, quality='balanced', **params):
            calls.append(len(mesh.vertices))
            return [trimesh.creation.box(extents=(0.02, 0.02, 0.02)),
                    trimesh.creation.box(extents=(0.02, 0.02, 0.02))
                    .apply_translation((0.05, 0.0, 0.0)),
                    trimesh.creation.box(extents=(0.01, 0.01, 0.01))]

        out_path = os.path.join(self.tmpdir, 'hand.xml')
        # By the module object: skrobot.utils also exports a function named
        # convex_decomposition, which shadows the module as an attribute.
        module = importlib.import_module('skrobot.utils.convex_decomposition')
        with mock.patch.object(module, 'convex_decomposition_cached',
                               fake_decomposition):
            urdf_to_mjcf(self.urdf, out_path, add_ground=False,
                         convex_decompose_links=['finger'])
        root = ET.parse(out_path).getroot()

        def collision_geoms(body_name):
            body = [b for b in root.iter('body')
                    if b.get('name') == body_name][0]
            return [g for g in body.findall('geom')
                    if 'collision' in (g.get('name') or '')]

        # One CoACD run, on both boxes merged (8 + 8 vertices).
        self.assertEqual(calls, [16])
        self.assertEqual(len(collision_geoms('finger')), 3)
        self.assertEqual(len(collision_geoms('base_link')), 2)


# The legged robot with frames on it: a tool tip under the foot (yawed 90 deg)
# with a marker under that, and a mount on the base that carries a real
# (massive) link, so it is not a pure frame and must stay a body.
_FRAMED_URDF = _LEGGED_URDF.replace("</robot>", """
  <joint name="tip_joint" type="fixed">
    <origin xyz="0 0 -0.15" rpy="0 0 1.5707963267948966"/>
    <parent link="foot"/><child link="tip"/>
  </joint>
  <link name="tip"/>
  <joint name="marker_joint" type="fixed">
    <origin xyz="0.01 0 0" rpy="0 0 0"/>
    <parent link="tip"/><child link="marker"/>
  </joint>
  <link name="marker"/>
  <joint name="mount_joint" type="fixed">
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
    <parent link="base_link"/><child link="mount"/>
  </joint>
  <link name="mount"/>
  <joint name="camera_joint" type="fixed">
    <origin xyz="0 0 0.02" rpy="0 0 0"/>
    <parent link="mount"/><child link="camera"/>
  </joint>
  <link name="camera">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.05"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
  </link>
</robot>""")


class TestFrameLinksBecomeSites(unittest.TestCase):
    """A link with no geometry and no mass is a frame, and MuJoCo names a point
    on a body with a site: an empty body would also pass for the end of the
    leg to anything that looks for leaf bodies."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.urdf = os.path.join(self.tmpdir, 'robot.urdf')
        with open(self.urdf, 'w') as f:
            f.write(_FRAMED_URDF)
        self.out_path = os.path.join(self.tmpdir, 'robot.xml')
        urdf_to_mjcf(self.urdf, self.out_path, floating_base=True)
        self.root = ET.parse(self.out_path).getroot()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _body(self, name):
        return [b for b in self.root.iter('body') if b.get('name') == name][0]

    def test_frames_are_sites_on_the_real_body(self):
        bodies = {b.get('name') for b in self.root.iter('body')}
        self.assertFalse({'tip', 'marker'} & bodies)
        sites = {s.get('name'): s for s in self._body('foot').findall('site')}
        self.assertEqual(set(sites), {'tip', 'marker'})

        def floats(el, key):
            return np.array([float(v) for v in el.get(key).split()])

        np.testing.assert_allclose(floats(sites['tip'], 'pos'), [0, 0, -0.15],
                                   atol=1e-9)
        half = np.pi / 4.0
        np.testing.assert_allclose(floats(sites['tip'], 'quat'),
                                   [np.cos(half), 0, 0, np.sin(half)],
                                   atol=1e-9)
        # +x of the tip is +y of the foot, so the marker lands 1 cm along +y
        np.testing.assert_allclose(floats(sites['marker'], 'pos'),
                                   [0, 0.01, -0.15], atol=1e-9)

    def test_a_frame_carrying_a_real_link_stays_a_body(self):
        mount = self._body('mount')
        self.assertEqual([b.get('name') for b in mount.findall('body')],
                         ['camera'])

    def test_the_foot_is_the_leaf_body(self):
        leaves = {b.get('name') for b in self.root.iter('body')
                  if b.find('body') is None}
        self.assertEqual(leaves, {'foot', 'camera'})

    def test_mujoco_places_the_sites(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest('mujoco is not installed')
        model = mujoco.MjModel.from_xml_path(self.out_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        foot = data.body('foot').xpos
        np.testing.assert_allclose(data.site('tip').xpos - foot,
                                   [0, 0, -0.15], atol=1e-9)
        np.testing.assert_allclose(data.site('marker').xpos - foot,
                                   [0, 0.01, -0.15], atol=1e-9)


# A two-finger gripper: finger_b mimics finger_a with multiplier -1 and an
# offset, and finger_a carries an inertia whose off-diagonal terms make it
# indefinite (eigenvalues -1e-4, 2e-4, 2e-4), as a mistyped URDF does.
_GRIPPER_URDF = """<?xml version="1.0"?>
<robot name="gripper">
  <link name="palm">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.2"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
  </link>
  <joint name="finger_a_joint" type="revolute">
    <origin xyz="0 0.02 0" rpy="0 0 0"/>
    <parent link="palm"/><child link="finger_a"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1.0" velocity="1.0"/>
  </joint>
  <link name="finger_a">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
      <inertia ixx="1e-4" ixy="-1e-4" ixz="-1e-4"
               iyy="1e-4" iyz="-1e-4" izz="1e-4"/>
    </inertial>
  </link>
  <joint name="finger_b_joint" type="revolute">
    <origin xyz="0 -0.02 0" rpy="0 0 0"/>
    <parent link="palm"/><child link="finger_b"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1.0" velocity="1.0"/>
    <mimic joint="finger_a_joint" multiplier="-1.0" offset="0.1"/>
  </joint>
  <link name="finger_b">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
  </link>
</robot>
"""


class TestInertiaWarningAndMimicQpos(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.urdf = os.path.join(self.tmpdir, 'gripper.urdf')
        with open(self.urdf, 'w') as f:
            f.write(_GRIPPER_URDF)
        self.out_path = os.path.join(self.tmpdir, 'gripper.xml')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _convert(self):
        urdf_to_mjcf(self.urdf, self.out_path, add_ground=False,
                     add_position_actuators=False)

    def test_invalid_inertia_is_reported_with_the_link_name(self):
        with self.assertLogs('skrobot.utils.mjcf_converter',
                             level='WARNING') as logs:
            self._convert()
        warned = [r.getMessage() for r in logs.records]
        self.assertTrue(any('finger_a' in m and 'not positive definite' in m
                            for m in warned), warned)
        # the valid inertias are not reported
        self.assertFalse(any('palm' in m or 'finger_b' in m for m in warned),
                         warned)

    def test_mimic_joint_follows_after_apply_mimic_qpos(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest('mujoco is not installed')
        from skrobot.urdf import apply_mimic_qpos

        self._convert()
        model = mujoco.MjModel.from_xml_path(self.out_path)
        data = mujoco.MjData(model)
        a = model.jnt_qposadr[model.joint('finger_a_joint').id]
        b = model.jnt_qposadr[model.joint('finger_b_joint').id]
        data.qpos[a] = 0.5
        # mj_forward alone does not move the mimic joint
        mujoco.mj_forward(model, data)
        self.assertEqual(data.qpos[b], 0.0)
        apply_mimic_qpos(model, data.qpos)
        self.assertAlmostEqual(data.qpos[b], 0.1 - 0.5)
        mujoco.mj_forward(model, data)
        # the equality is now satisfied, so the solver sees no violation
        self.assertTrue(np.allclose(data.efc_pos[:data.nefc], 0.0))


class TestTexturedMesh(unittest.TestCase):
    """An OBJ + MTL + image keeps its texture in the MJCF."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_texture_material_and_uv(self):
        from PIL import Image

        Image.new('RGB', (8, 4), (10, 200, 30)).save(
            os.path.join(self.tmpdir, 'part.png'))
        box = _solid_box()
        # One texture coordinate per vertex is all the test needs.
        uv = (box.vertices[:, :2] - box.vertices[:, :2].min(axis=0)) / 0.1
        box.visual = trimesh.visual.TextureVisuals(
            uv=uv, image=Image.open(os.path.join(self.tmpdir, 'part.png')))
        box.export(os.path.join(self.tmpdir, 'part.obj'))
        urdf_path = os.path.join(self.tmpdir, 'robot.urdf')
        with open(urdf_path, 'w') as f:
            f.write(_URDF_TEMPLATE.format(mesh='part.obj'))
        out_path = os.path.join(self.tmpdir, 'robot.xml')
        urdf_to_mjcf(urdf_path, out_path, add_ground=False)
        root = ET.parse(out_path).getroot()
        asset = root.find('asset')
        self.assertEqual(len(asset.findall('texture')), 1)
        self.assertEqual(len(asset.findall('material')), 1)
        self.assertTrue(asset.find('mesh').get('file').endswith('.obj'))
        material = asset.find('material').get('name')
        geoms = {g.get('group'): g for g in root.iter('geom')}
        self.assertEqual(geoms['2'].get('material'), material)
        self.assertIsNone(geoms['3'].get('material'))
        try:
            import mujoco
        except ImportError:
            return
        model = mujoco.MjModel.from_xml_path(out_path)
        self.assertEqual(model.ntex, 1)
        self.assertEqual(model.mesh_texcoord.shape[0] > 0, True)


class TestMtlDissolve(unittest.TestCase):
    """An OBJ material with ``d`` (glass) keeps its transparency."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_glass_is_translucent(self):
        with open(os.path.join(self.tmpdir, 'part.mtl'), 'w') as f:
            f.write('newmtl glass\nKd 0.8 0.85 0.88\nd 0.150\n')
        box = _solid_box()
        lines = ['mtllib part.mtl', 'usemtl glass']
        lines += ['v {} {} {}'.format(*v) for v in box.vertices]
        lines += ['f {} {} {}'.format(*(f + 1)) for f in box.faces]
        with open(os.path.join(self.tmpdir, 'part.obj'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        urdf_path = os.path.join(self.tmpdir, 'robot.urdf')
        with open(urdf_path, 'w') as f:
            f.write(_URDF_TEMPLATE.format(mesh='part.obj'))
        out_path = os.path.join(self.tmpdir, 'robot.xml')
        urdf_to_mjcf(urdf_path, out_path, add_ground=False)
        visual = [g for g in ET.parse(out_path).getroot().iter('geom')
                  if g.get('group') == '2'][0]
        self.assertAlmostEqual(float(visual.get('rgba').split()[3]), 0.15)

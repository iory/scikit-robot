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

    def _convert(self, mesh):
        mesh_path = os.path.join(self.tmpdir, 'part.stl')
        mesh.export(mesh_path)
        urdf_path = os.path.join(self.tmpdir, 'robot.urdf')
        with open(urdf_path, 'w') as f:
            f.write(_URDF_TEMPLATE.format(mesh='part.stl'))
        out_path = os.path.join(self.tmpdir, 'robot.xml')
        urdf_to_mjcf(urdf_path, out_path, add_ground=False)
        return out_path, ET.parse(out_path).getroot()

    def _mesh_assets(self, root):
        asset = root.find('asset')
        return [] if asset is None else asset.findall('mesh')

    def test_solid_mesh_is_not_marked_shell(self):
        # <visual> and <collision> load the file separately, so one mesh file
        # becomes two assets; neither is a shell.
        _, root = self._convert(_solid_box())
        assets = self._mesh_assets(root)
        self.assertEqual(len(assets), 2)
        self.assertEqual([a.get('inertia') for a in assets], [None, None])

    def test_zero_volume_mesh_is_marked_shell(self):
        _, root = self._convert(_thin_shell())
        assets = self._mesh_assets(root)
        self.assertEqual(len(assets), 2)
        self.assertEqual([a.get('inertia') for a in assets],
                         ['shell', 'shell'])

    def test_three_vertex_mesh_is_dropped(self):
        # Too few vertices for MuJoCo, so no asset and no geom referencing one.
        _, root = self._convert(_single_triangle())
        self.assertEqual(len(self._mesh_assets(root)), 0)
        mesh_geoms = [g for g in root.iter('geom') if g.get('type') == 'mesh']
        self.assertEqual(mesh_geoms, [])

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

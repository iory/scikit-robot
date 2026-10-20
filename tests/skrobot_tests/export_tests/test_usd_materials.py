"""A glTF visual with a PBR material becomes a bound UsdPreviewSurface."""
import os
import shutil
import tempfile
import unittest
import warnings


try:
    from pxr import UsdShade
    _HAS_PXR = True
except ImportError:
    _HAS_PXR = False

from skrobot.export.usd import urdf_to_usd


URDF = '''<?xml version="1.0"?>
<robot name="pbr">
  <link name="base_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
    <visual><geometry><mesh filename="{glb}"/></geometry></visual>
    <collision><geometry><mesh filename="{glb}"/></geometry></collision>
  </link>
  <link name="plain_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
    <visual><geometry><mesh filename="{stl}"/></geometry>
      <material name="red"><color rgba="1 0 0 1"/></material></visual>
  </link>
  <joint name="j" type="fixed">
    <parent link="base_link"/><child link="plain_link"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
'''


def _box_glb(path, rgba, metallic, roughness, name):
    import trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial
    box = trimesh.creation.box(extents=(0.02, 0.03, 0.04))
    box.visual = TextureVisuals(material=PBRMaterial(
        baseColorFactor=rgba, metallicFactor=metallic,
        roughnessFactor=roughness, name=name))
    scene = trimesh.Scene()
    scene.add_geometry(box, node_name=name, geom_name=name)
    with open(path, 'wb') as f:
        f.write(scene.export(file_type='glb'))


def _bound_shader(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    material, _rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return None
    shaders = [UsdShade.Shader(c) for c in material.GetPrim().GetChildren()
               if UsdShade.Shader(c)]
    return material, shaders[0]


@unittest.skipUnless(_HAS_PXR, 'usd-core (pxr) not installed')
class TestPbrMaterials(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        glb = os.path.join(cls.tmp, 'shiny.glb')
        _box_glb(glb, [128, 128, 128, 255], 1.0, 0.08, 'chrome')
        import trimesh
        stl = os.path.join(cls.tmp, 'plain.stl')
        trimesh.creation.box(extents=(0.02, 0.02, 0.02)).export(stl)
        urdf = os.path.join(cls.tmp, 'pbr.urdf')
        with open(urdf, 'w') as f:
            f.write(URDF.format(glb='shiny.glb', stl='plain.stl'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.stage = urdf_to_usd(urdf, os.path.join(cls.tmp, 'pbr.usda'))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_glb_visual_is_bound_to_a_preview_surface(self):
        found = _bound_shader(self.stage, '/robot/base_link/visual_0_0')
        self.assertIsNotNone(found)
        material, shader = found
        self.assertEqual(shader.GetIdAttr().Get(), 'UsdPreviewSurface')
        self.assertAlmostEqual(shader.GetInput('metallic').Get(), 1.0)
        self.assertAlmostEqual(shader.GetInput('roughness').Get(), 0.08)
        # 128/255 sRGB -> 0.216 linear, the same value displayColor carries
        diffuse = shader.GetInput('diffuseColor').Get()
        self.assertAlmostEqual(diffuse[0], 0.2158605, places=4)
        self.assertIn('chrome', material.GetPath().pathString)
        self.assertTrue(material.GetPath().pathString.startswith('/robot/Looks/'))

    def test_vertex_colour_visual_keeps_no_material(self):
        # an STL with a URDF <material> colour: displayColor only, as before
        self.assertIsNone(_bound_shader(self.stage, '/robot/plain_link/visual_0_0'))

    def test_collision_meshes_get_no_material(self):
        for prim in self.stage.Traverse():
            if 'collision' in prim.GetName():
                material, _rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                self.assertFalse(bool(material), prim.GetPath())

    def test_identical_finishes_share_one_material(self):
        looks = [p for p in self.stage.Traverse()
                 if p.GetPath().pathString.startswith('/robot/Looks/')
                 and p.IsA(UsdShade.Material)]
        self.assertEqual(len(looks), 1)

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
        # glTF's baseColorFactor is linear already: 128/255 stays 0.502
        diffuse = shader.GetInput('diffuseColor').Get()
        self.assertAlmostEqual(diffuse[0], 128 / 255.0, places=4)
        self.assertIn('chrome', material.GetPath().pathString)
        self.assertTrue(material.GetPath().pathString.startswith('/robot/Looks/'))

    def test_glb_display_color_is_the_linear_base_color(self):
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath('/robot/base_link/visual_0_0')
        color = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
        self.assertAlmostEqual(color[0][0], 128 / 255.0, places=4)

    def test_urdf_material_color_is_linearised(self):
        # URDF rgba is an sRGB display colour: 1 0 0 stays 1 0 0
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath('/robot/plain_link/visual_0_0')
        color = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
        self.assertAlmostEqual(color[0][0], 1.0, places=4)
        self.assertAlmostEqual(color[0][1], 0.0, places=4)

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


@unittest.skipUnless(_HAS_PXR, 'usd-core (pxr) not installed')
class TestColladaColorStaysSrgb(unittest.TestCase):

    def test_collada_diffuse_is_linearised(self):
        import trimesh
        from pxr import UsdGeom
        tmp = tempfile.mkdtemp()
        try:
            box = trimesh.creation.box(extents=(0.02, 0.03, 0.04))
            box.visual = trimesh.visual.ColorVisuals(
                box, face_colors=[128, 128, 128, 255])
            with open(os.path.join(tmp, 'grey.dae'), 'wb') as f:
                f.write(trimesh.exchange.dae.export_collada(box))
            urdf = os.path.join(tmp, 'dae.urdf')
            with open(urdf, 'w') as f:
                f.write('<robot name="dae"><link name="base_link"><visual>'
                        '<geometry><mesh filename="grey.dae"/></geometry>'
                        '</visual></link></robot>')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                stage = urdf_to_usd(urdf, os.path.join(tmp, 'dae.usda'))
            prim = stage.GetPrimAtPath('/robot/base_link/visual_0_0')
            color = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
            # COLLADA colours are sRGB: 128/255 -> 0.216 linear
            self.assertAlmostEqual(color[0][0], 0.2158605, places=4)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _textured_box_glb(path):
    import numpy as np
    from PIL import Image
    import trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial
    box = trimesh.creation.box(extents=(0.02, 0.03, 0.04))
    uv = (box.vertices[:, :2] - box.vertices[:, :2].min(0)) / np.ptp(box.vertices[:, :2], axis=0)
    image = Image.new('RGB', (8, 4), (200, 30, 30))
    box.visual = TextureVisuals(uv=uv, material=PBRMaterial(
        baseColorTexture=image, metallicFactor=0.0, roughnessFactor=0.6,
        name='painted'))
    scene = trimesh.Scene()
    scene.add_geometry(box, node_name='painted', geom_name='painted')
    with open(path, 'wb') as f:
        f.write(scene.export(file_type='glb'))


@unittest.skipUnless(_HAS_PXR, 'usd-core (pxr) not installed')
class TestTextures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        _textured_box_glb(os.path.join(cls.tmp, 'painted.glb'))
        import trimesh
        trimesh.creation.box(extents=(0.02, 0.02, 0.02)).export(
            os.path.join(cls.tmp, 'plain.stl'))
        urdf = os.path.join(cls.tmp, 'tex.urdf')
        with open(urdf, 'w') as f:
            f.write(URDF.format(glb='painted.glb', stl='plain.stl'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.stage = urdf_to_usd(urdf, os.path.join(cls.tmp, 'tex.usda'))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_uvs_become_st(self):
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath('/robot/base_link/visual_0_0')
        st = UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
        self.assertTrue(st)
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        self.assertEqual(len(st.Get()), len(points))

    def test_diffuse_reads_the_image(self):
        material, shader = _bound_shader(self.stage, '/robot/base_link/visual_0_0')
        sources = shader.GetInput('diffuseColor').GetConnectedSources()[0]
        texture = UsdShade.Shader(sources[0].source.GetPrim())
        self.assertEqual(texture.GetIdAttr().Get(), 'UsdUVTexture')
        rel = texture.GetInput('file').Get().path
        path = os.path.join(self.tmp, rel)
        self.assertTrue(os.path.exists(path), path)
        from PIL import Image
        self.assertEqual(Image.open(path).convert('RGB').getpixel((0, 0)), (200, 30, 30))

    def test_untextured_link_has_no_st(self):
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath('/robot/plain_link/visual_0_0')
        self.assertFalse(UsdGeom.PrimvarsAPI(prim).GetPrimvar('st'))


URDF_COLOURED = '''<?xml version="1.0"?>
<robot name="coloured">
  <link name="base_link">
    <visual><geometry><mesh filename="{glb}"/></geometry>
      <material name="white"><color rgba="1 1 1 1"/></material></visual>
  </link>
  <link name="plain_link">
    <visual><geometry><mesh filename="{stl}"/></geometry>
      <material name="red"><color rgba="1 0 0 1"/></material></visual>
  </link>
  <joint name="j" type="fixed">
    <parent link="base_link"/><child link="plain_link"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
'''


@unittest.skipUnless(_HAS_PXR, 'usd-core (pxr) not installed')
class TestMeshColoursOverUrdfMaterial(unittest.TestCase):
    """A URDF <material> colours only meshes without colours of their own."""

    @classmethod
    def setUpClass(cls):
        import numpy as np
        import trimesh
        cls.tmp = tempfile.mkdtemp()
        box = trimesh.creation.box(extents=(0.02, 0.03, 0.04))
        colours = np.tile([0, 0, 0, 255], (len(box.vertices), 1))
        colours[box.vertices[:, 2] > 0] = [0, 0, 255, 255]   # black, blue top
        box.visual = trimesh.visual.ColorVisuals(box, vertex_colors=colours)
        box.export(os.path.join(cls.tmp, 'two_tone.glb'))
        trimesh.creation.box(extents=(0.02, 0.02, 0.02)).export(
            os.path.join(cls.tmp, 'plain.stl'))
        urdf = os.path.join(cls.tmp, 'coloured.urdf')
        with open(urdf, 'w') as f:
            f.write(URDF_COLOURED.format(glb='two_tone.glb', stl='plain.stl'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.stage = urdf_to_usd(urdf, os.path.join(cls.tmp, 'coloured.usda'))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _display_colour(self, path):
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath(path)
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar('displayColor')
        return pv.GetInterpolation(), {tuple(round(c, 3) for c in v) for v in pv.Get()}

    def test_vertex_colours_survive_a_urdf_material(self):
        interp, colours = self._display_colour('/robot/base_link/visual_0_0')
        self.assertEqual(interp, 'vertex')
        self.assertEqual(colours, {(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)})

    def test_uncoloured_mesh_takes_the_urdf_material(self):
        interp, colours = self._display_colour('/robot/plain_link/visual_0_0')
        self.assertEqual(interp, 'constant')
        self.assertEqual(colours, {(1.0, 0.0, 0.0)})

import json
import os
import struct
import tempfile

import numpy as np
import pytest

from skrobot.utils.draco import is_dracopy_available


# Check availability at module load time to handle numpy ABI issues
_dracopy_available = is_dracopy_available()

pytestmark = pytest.mark.skipif(
    not _dracopy_available,
    reason="DracoPy is not installed or incompatible"
)


class TestDraco:

    def test_dracopy_available(self):
        """Test that DracoPy is available."""
        assert is_dracopy_available()

    def test_encode_decode_positions_only(self):
        """Test encoding and decoding mesh with positions only."""
        import DracoPy

        vertices = np.array([
            [-0.1, -0.1, 0.0],
            [0.1, -0.1, 0.0],
            [0.0, 0.1, 0.0],
        ], dtype=np.float32)
        faces = np.array([0, 1, 2], dtype=np.uint32)

        compressed = DracoPy.encode(
            points=vertices,
            faces=faces,
            quantization_bits=14
        )
        decoded = DracoPy.decode(compressed)

        decoded_verts = np.array(decoded.points).reshape(-1, 3)

        np.testing.assert_allclose(decoded_verts, vertices, atol=1e-4)

    def test_encode_decode_with_colors(self):
        """Test encoding and decoding mesh with positions and colors."""
        import DracoPy

        vertices = np.array([
            [-0.1, -0.1, 0.0],
            [0.1, -0.1, 0.0],
            [0.0, 0.1, 0.0],
        ], dtype=np.float32)
        faces = np.array([0, 1, 2], dtype=np.uint32)
        colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ], dtype=np.uint8)

        compressed = DracoPy.encode(
            points=vertices,
            faces=faces,
            colors=colors,
            quantization_bits=14
        )
        decoded = DracoPy.decode(compressed)

        decoded_verts = np.array(decoded.points).reshape(-1, 3)
        decoded_colors = np.array(decoded.colors).reshape(-1, 3)

        np.testing.assert_allclose(decoded_verts, vertices, atol=1e-4)
        np.testing.assert_array_equal(decoded_colors, colors)

    def test_draco_attribute_order_with_colors(self):
        """Test that Draco attribute IDs are correctly assigned.

        When colors are provided, DracoPy assigns:
        - unique_id=0 to COLOR
        - unique_id=1 to POSITION

        This test verifies this behavior which is critical for correct
        GLB attribute mapping.
        """
        import DracoPy

        vertices = np.array([
            [-0.1, -0.1, 0.0],
            [0.1, -0.1, 0.0],
            [0.0, 0.1, 0.0],
        ], dtype=np.float32)
        faces = np.array([0, 1, 2], dtype=np.uint32)
        colors = np.array([
            [75, 75, 75],
            [128, 128, 128],
            [200, 200, 200],
        ], dtype=np.uint8)

        compressed = DracoPy.encode(
            points=vertices,
            faces=faces,
            colors=colors,
            quantization_bits=14
        )
        decoded = DracoPy.decode(compressed)

        # Check attribute order
        # attribute_type: 0=POSITION, 2=COLOR
        attributes = decoded.attributes
        assert len(attributes) == 2

        # Find position and color attributes
        position_attr = None
        color_attr = None
        for attr in attributes:
            if attr['attribute_type'] == 0:  # POSITION
                position_attr = attr
            elif attr['attribute_type'] == 2:  # COLOR
                color_attr = attr

        assert position_attr is not None
        assert color_attr is not None

        # Verify unique_ids: COLOR should be 0, POSITION should be 1
        assert color_attr['unique_id'] == 0
        assert position_attr['unique_id'] == 1

        # Verify data types
        assert position_attr['data'].dtype == np.float32
        assert color_attr['data'].dtype == np.uint8

    def test_export_glb_with_draco(self):
        """Test exporting a mesh to GLB with Draco compression."""
        import trimesh

        from skrobot.utils.draco import export_glb_with_draco

        vertices = np.array([
            [-0.1, -0.1, 0.0],
            [0.1, -0.1, 0.0],
            [0.0, 0.1, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.visual.vertex_colors = np.array([
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
        ], dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
            temp_path = f.name

        try:
            export_glb_with_draco([mesh], temp_path)

            assert os.path.exists(temp_path)

            with open(temp_path, 'rb') as f:
                data = f.read()

            # Check GLB header
            magic, version, length = struct.unpack('<4sII', data[:12])
            assert magic == b'glTF'
            assert version == 2

            # Parse JSON chunk
            json_len, json_type = struct.unpack('<II', data[12:20])
            assert json_type == 0x4E4F534A  # "JSON"
            json_data = json.loads(data[20:20 + json_len].decode('utf-8').rstrip())

            # Verify glTF structure
            assert 'extensionsUsed' in json_data
            assert 'KHR_draco_mesh_compression' in json_data['extensionsUsed']

            # Verify accessors
            accessors = json_data['accessors']
            assert len(accessors) >= 2

            # Verify mesh primitive has Draco extension
            mesh_data = json_data['meshes'][0]
            primitive = mesh_data['primitives'][0]
            assert 'extensions' in primitive
            assert 'KHR_draco_mesh_compression' in primitive['extensions']

            draco_ext = primitive['extensions']['KHR_draco_mesh_compression']
            assert 'bufferView' in draco_ext
            assert 'attributes' in draco_ext

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_glb_draco_attribute_mapping(self):
        """Test that GLB Draco attribute mapping is correct for Three.js.

        This test verifies the fix for the attribute mapping bug where
        POSITION and COLOR_0 indices were swapped in the GLB metadata.
        """
        import trimesh

        from skrobot.utils.draco import export_glb_with_draco

        vertices = np.array([
            [-0.05, -0.05, 0.0],
            [0.05, -0.05, 0.0],
            [0.0, 0.05, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.visual.vertex_colors = np.array([
            [100, 100, 100, 255],
            [150, 150, 150, 255],
            [200, 200, 200, 255],
        ], dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
            temp_path = f.name

        try:
            export_glb_with_draco([mesh], temp_path)

            with open(temp_path, 'rb') as f:
                data = f.read()

            # Parse JSON chunk
            json_len = struct.unpack('<I', data[12:16])[0]
            json_data = json.loads(data[20:20 + json_len].decode('utf-8').rstrip())

            # Get Draco extension attributes mapping
            primitive = json_data['meshes'][0]['primitives'][0]
            draco_attrs = primitive['extensions']['KHR_draco_mesh_compression']['attributes']

            # Verify correct mapping:
            # - POSITION should map to Draco unique_id 1 (not 0)
            # - COLOR_0 should map to Draco unique_id 0 (not 1)
            assert draco_attrs['POSITION'] == 1
            assert draco_attrs['COLOR_0'] == 0

            # Also verify accessor metadata has correct bounds
            position_accessor_idx = primitive['attributes']['POSITION']
            position_accessor = json_data['accessors'][position_accessor_idx]

            # Position min/max should be actual vertex coordinates, not color values
            pos_min = position_accessor['min']
            pos_max = position_accessor['max']

            # Vertex coordinates are in range [-0.05, 0.05], not [0, 1] (which would be color)
            assert pos_min[0] < 0  # Should have negative values
            assert pos_max[0] < 1  # Should be less than 1

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_handler_glb_valid_gltf_structure(self):
        """Test that Draco GLB exported via trimesh handler has valid glTF.

        The trimesh primitive_export handler must produce a GLB where:
        - Draco primitives reference valid accessor indices
        - Accessors have required componentType/type/count fields
        - bufferView index in Draco extension is valid
        - extensionsUsed includes KHR_draco_mesh_compression

        This validates the fix for ctx["accessors"] -> ctx["tree"] and
        the OrderedDict buffer_items handling.
        """
        import trimesh

        from skrobot.utils.draco import register_dracopy_handlers

        register_dracopy_handlers()

        mesh = trimesh.Trimesh(
            vertices=np.array([
                [-0.1, -0.1, 0.0],
                [0.1, -0.1, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, -0.1, 0.1],
            ]),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
        )
        mesh.visual.vertex_colors = np.array([
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 255],
        ], dtype=np.uint8)

        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="test_mesh")
        glb_data = scene.export(file_type="glb", extension_draco=True)

        # Parse GLB
        magic = glb_data[:4]
        assert magic == b'glTF'

        json_len = struct.unpack('<I', glb_data[12:16])[0]
        gltf = json.loads(glb_data[20:20 + json_len])

        # Must declare the extension
        assert 'KHR_draco_mesh_compression' in gltf.get('extensionsUsed', [])

        # Validate each mesh primitive
        accessors = gltf.get('accessors', [])
        buffer_views = gltf.get('bufferViews', [])

        for mesh_data in gltf['meshes']:
            for prim in mesh_data['primitives']:
                draco = prim.get('extensions', {}).get(
                    'KHR_draco_mesh_compression')
                assert draco is not None, \
                    "Primitive missing KHR_draco_mesh_compression extension"

                # bufferView index must be valid
                assert draco['bufferView'] < len(buffer_views), \
                    f"Draco bufferView {draco['bufferView']} out of range"

                # Draco attributes must be a non-empty dict
                assert len(draco['attributes']) > 0

                # Primitive attributes must reference valid accessors
                for attr_name, acc_idx in prim['attributes'].items():
                    assert acc_idx < len(accessors), \
                        f"Attribute {attr_name} accessor {acc_idx} out of range"
                    acc = accessors[acc_idx]
                    assert 'componentType' in acc, \
                        f"Accessor {acc_idx} missing componentType"
                    assert 'type' in acc, \
                        f"Accessor {acc_idx} missing type"
                    assert 'count' in acc, \
                        f"Accessor {acc_idx} missing count"

                # indices accessor must be valid
                if 'indices' in prim:
                    idx = prim['indices']
                    assert idx < len(accessors), \
                        f"Indices accessor {idx} out of range"

    def test_handler_roundtrip(self):
        """Test encode via handler then decode back produces same geometry."""
        import trimesh

        from skrobot.utils.draco import register_dracopy_handlers

        register_dracopy_handlers()

        vertices = np.array([
            [-0.05, -0.05, 0.0],
            [0.05, -0.05, 0.0],
            [0.0, 0.05, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        colors = np.array([
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
        ], dtype=np.uint8)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.visual.vertex_colors = colors

        # Export via handler
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="test")
        glb_data = scene.export(file_type="glb", extension_draco=True)

        # Re-import
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
            f.write(glb_data)
            temp_path = f.name

        try:
            loaded = trimesh.load(temp_path)
            if isinstance(loaded, trimesh.Scene):
                loaded_meshes = list(loaded.geometry.values())
            else:
                loaded_meshes = [loaded]

            assert len(loaded_meshes) >= 1
            loaded_mesh = loaded_meshes[0]

            # Geometry should survive the roundtrip
            assert len(loaded_mesh.vertices) == len(vertices)
            assert len(loaded_mesh.faces) == len(faces)
            np.testing.assert_allclose(
                loaded_mesh.vertices, vertices, atol=1e-3)
        finally:
            os.unlink(temp_path)

    def test_export_glb_with_default_colors(self):
        """Test exporting a mesh with trimesh's default colors.

        Note: trimesh assigns a default gray color (main_color) to meshes
        even when no explicit colors are provided. This test verifies that
        the default colors are correctly exported.
        """
        import trimesh

        from skrobot.utils.draco import export_glb_with_draco

        vertices = np.array([
            [-0.1, -0.1, 0.0],
            [0.1, -0.1, 0.0],
            [0.0, 0.1, 0.0],
        ])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Verify trimesh assigns default main_color
        assert hasattr(mesh.visual, 'main_color')
        assert mesh.visual.main_color is not None

        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
            temp_path = f.name

        try:
            export_glb_with_draco([mesh], temp_path)

            with open(temp_path, 'rb') as f:
                data = f.read()

            # Parse JSON chunk
            json_len = struct.unpack('<I', data[12:16])[0]
            json_data = json.loads(data[20:20 + json_len].decode('utf-8').rstrip())

            # Get Draco extension
            primitive = json_data['meshes'][0]['primitives'][0]
            draco_attrs = primitive['extensions']['KHR_draco_mesh_compression']['attributes']

            # With default colors from main_color:
            # - POSITION maps to unique_id 1
            # - COLOR_0 maps to unique_id 0
            assert draco_attrs['POSITION'] == 1
            assert draco_attrs['COLOR_0'] == 0

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

import unittest

import numpy as np
import trimesh

from skrobot.utils.convex_decomposition import convex_decomposition
from skrobot.utils.convex_decomposition import is_coacd_available


# keep the MCTS search shallow so the test suite stays fast; the defaults are
# tuned for real CAD parts, not unit tests
_FAST_PARAMS = {'preprocess_resolution': 20, 'mcts_iterations': 10}


class TestConvexDecomposition(unittest.TestCase):

    def test_is_coacd_available_returns_bool(self):
        self.assertIsInstance(is_coacd_available(), bool)

    def test_unknown_quality_raises(self):
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        with self.assertRaises(ValueError):
            convex_decomposition(mesh, quality='ultra')

    @unittest.skipUnless(is_coacd_available(), 'coacd is not installed')
    def test_convex_box_yields_single_part(self):
        mesh = trimesh.creation.box(extents=(1.0, 0.5, 0.25))
        parts = convex_decomposition(mesh, **_FAST_PARAMS)
        self.assertEqual(len(parts), 1)
        part = parts[0]
        self.assertTrue(part.is_watertight)
        self.assertTrue(part.is_convex)
        # an approximate decomposition of a box should stay close to it
        self.assertAlmostEqual(
            part.volume, mesh.volume, delta=0.15 * mesh.volume)

    @unittest.skipUnless(is_coacd_available(), 'coacd is not installed')
    def test_concave_mesh_yields_multiple_convex_parts(self):
        # a hollow ring is strongly concave: no single convex body fits it
        mesh = trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.3)
        parts = convex_decomposition(mesh, **_FAST_PARAMS)
        self.assertGreaterEqual(len(parts), 2)
        for part in parts:
            self.assertTrue(part.is_watertight)
            self.assertTrue(part.is_convex)
        # the union of the parts should roughly cover the input footprint
        combined = np.vstack([part.bounds for part in parts])
        combined_min = combined.min(axis=0)
        combined_max = combined.max(axis=0)
        extent = combined_max - combined_min
        expected = mesh.bounds[1] - mesh.bounds[0]
        np.testing.assert_allclose(extent, expected, rtol=0.2)


if __name__ == '__main__':
    unittest.main()

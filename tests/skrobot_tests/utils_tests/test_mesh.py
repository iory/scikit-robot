import unittest

import numpy as np
import trimesh

from skrobot.utils.mesh import split_mesh_by_face_color


class TestSplitMeshByFaceColorNormals(unittest.TestCase):
    """Splitting by colour must not throw away the file's vertex normals.

    ``submesh`` builds fresh meshes, so normals loaded from the mesh file are
    dropped and later re-derived by averaging adjacent faces -- which rounds
    off exactly the hard edges the file was recording.
    """

    def _two_colour_box(self):
        mesh = trimesh.creation.box()
        colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
        colors[:, 3] = 255
        colors[: len(mesh.faces) // 2, 0] = 255
        colors[len(mesh.faces) // 2:, 1] = 255
        mesh.visual.face_colors = colors
        return mesh

    def test_authored_normals_survive_the_split(self):
        mesh = self._two_colour_box()
        # Something no averaging would ever produce, so we can tell whether
        # the array came from us or was recomputed.
        authored = np.tile([0.0, 0.0, 1.0], (len(mesh.vertices), 1))
        mesh.vertex_normals = authored

        submeshes = split_mesh_by_face_color(mesh)

        self.assertGreater(len(submeshes), 1)
        for submesh in submeshes:
            self.assertIn('vertex_normals', submesh._cache.cache)
            np.testing.assert_allclose(
                np.asarray(submesh.vertex_normals),
                np.tile([0.0, 0.0, 1.0], (len(submesh.vertices), 1)),
                atol=1e-9)

    def test_normals_follow_their_own_vertices(self):
        """Each submesh keeps only its own vertices, so the normals have to be
        re-indexed the same way rather than merely sliced."""
        mesh = self._two_colour_box()
        # A distinct normal per vertex, so a wrong mapping cannot pass.
        authored = np.zeros((len(mesh.vertices), 3))
        authored[:, 0] = 1.0
        authored[:, 1] = np.arange(len(mesh.vertices))
        authored /= np.linalg.norm(authored, axis=1)[:, None]
        mesh.vertex_normals = authored

        # A box has eight distinct corners, so each submesh vertex maps back
        # to exactly one original -- no need to assume the grouping order.
        for submesh in split_mesh_by_face_color(mesh):
            got = np.asarray(submesh.vertex_normals)
            for vertex, normal in zip(np.asarray(submesh.vertices), got):
                source = np.argmin(
                    np.linalg.norm(np.asarray(mesh.vertices) - vertex, axis=1))
                np.testing.assert_allclose(
                    normal, authored[source], atol=1e-9)

    def test_a_mesh_without_authored_normals_is_untouched(self):
        mesh = self._two_colour_box()
        submeshes = split_mesh_by_face_color(mesh)
        self.assertGreater(len(submeshes), 1)
        for submesh in submeshes:
            self.assertNotIn('vertex_normals', submesh._cache.cache)


if __name__ == '__main__':
    unittest.main()

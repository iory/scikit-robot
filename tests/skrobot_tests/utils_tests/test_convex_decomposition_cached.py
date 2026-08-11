import importlib
import json
import os
import shutil
import tempfile
import threading
import unittest

import numpy as np
import trimesh

from skrobot.utils.convex_decomposition import convex_decomposition_cached


# ``skrobot.utils.convex_decomposition`` names the function, not the module,
# so reach the module itself to swap the (slow, optional) decomposer out.
cd_module = importlib.import_module('skrobot.utils.convex_decomposition')


class _CountingDecomposer(object):
    """Stand-in for ``convex_decomposition`` that counts its own calls.

    CoACD is an optional dependency and is slow even when installed, so the
    cache tests patch it out: what they check is that the expensive callable
    runs exactly as often as the cache says it should.
    """

    def __init__(self, delay=0.0):
        self.calls = 0
        self.seen = []
        self.delay = delay
        self._lock = threading.Lock()

    def __call__(self, mesh, quality='balanced', **params):
        if self.delay:
            import time
            time.sleep(self.delay)
        with self._lock:
            self.calls += 1
            self.seen.append((quality, dict(params)))
        # two parts, so the manifest genuinely holds a list
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        return [box.convex_hull,
                trimesh.creation.box(extents=(0.5, 0.5, 2.0)).convex_hull]


def _explode(*args, **kwargs):
    raise AssertionError('the decomposer must not run on a cache hit')


def _explode_load(*args, **kwargs):
    raise AssertionError('the source mesh must not be loaded on a cache hit')


class TestConvexDecompositionCached(unittest.TestCase):

    def setUp(self):
        self.cache_dir = tempfile.mkdtemp(prefix='skrobot_coacd_cache_')
        self.mesh = trimesh.creation.annulus(
            r_min=0.5, r_max=1.0, height=0.3)
        self._original = cd_module.convex_decomposition

    def tearDown(self):
        cd_module.convex_decomposition = self._original
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _patch(self, decomposer):
        cd_module.convex_decomposition = decomposer

    def _entries(self):
        return sorted(os.listdir(self.cache_dir))

    def test_miss_then_hit_runs_decomposer_once(self):
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        first = convex_decomposition_cached(
            self.mesh, cache_dir=self.cache_dir)
        self.assertEqual(decomposer.calls, 1)
        self.assertEqual(len(self._entries()), 1)

        # a hit must not need the decomposer at all -- that is the point
        self._patch(_explode)
        second = convex_decomposition_cached(
            self.mesh, cache_dir=self.cache_dir)
        self.assertEqual(len(second), len(first))
        for a, b in zip(first, second):
            np.testing.assert_allclose(a.vertices, b.vertices)
            np.testing.assert_array_equal(a.faces, b.faces)
            self.assertTrue(b.is_watertight)

    def test_hit_works_with_the_real_decomposer_in_place(self):
        # the cache is what makes CoACD optional at call time: with the entry
        # warm, the real code path is never reached, so an environment without
        # the ``coacd`` package still gets its parts
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        convex_decomposition_cached(self.mesh, cache_dir=self.cache_dir)
        cd_module.convex_decomposition = self._original
        parts = convex_decomposition_cached(
            self.mesh, cache_dir=self.cache_dir)
        self.assertEqual(len(parts), 2)

    def test_quality_changes_the_key(self):
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        convex_decomposition_cached(
            self.mesh, quality='balanced', cache_dir=self.cache_dir)
        convex_decomposition_cached(
            self.mesh, quality='fine', cache_dir=self.cache_dir)
        self.assertEqual(decomposer.calls, 2)
        self.assertEqual(len(self._entries()), 2)
        # ... and each preset still caches on its own
        self._patch(_explode)
        convex_decomposition_cached(
            self.mesh, quality='balanced', cache_dir=self.cache_dir)
        convex_decomposition_cached(
            self.mesh, quality='fine', cache_dir=self.cache_dir)

    def test_explicit_params_change_the_key(self):
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        convex_decomposition_cached(self.mesh, cache_dir=self.cache_dir)
        convex_decomposition_cached(
            self.mesh, cache_dir=self.cache_dir, threshold=0.42)
        self.assertEqual(decomposer.calls, 2)
        self.assertEqual(len(self._entries()), 2)

    def test_params_matching_the_preset_hit_the_same_entry(self):
        # the key is built from the effective parameters, so naming a preset
        # value explicitly must not miss
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        convex_decomposition_cached(self.mesh, cache_dir=self.cache_dir)
        preset = dict(cd_module.COACD_PRESETS['balanced'])
        convex_decomposition_cached(
            self.mesh, cache_dir=self.cache_dir, **preset)
        self.assertEqual(decomposer.calls, 1)
        self.assertEqual(len(self._entries()), 1)

    def test_different_mesh_changes_the_key(self):
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        convex_decomposition_cached(self.mesh, cache_dir=self.cache_dir)
        other = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
        convex_decomposition_cached(other, cache_dir=self.cache_dir)
        self.assertEqual(decomposer.calls, 2)
        self.assertEqual(len(self._entries()), 2)

    def test_corrupt_manifest_is_rebuilt(self):
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        expected = convex_decomposition_cached(
            self.mesh, cache_dir=self.cache_dir)
        entry = os.path.join(self.cache_dir, self._entries()[0])
        with open(os.path.join(entry, 'parts.json'), 'w') as f:
            f.write('{not json at all')

        parts = convex_decomposition_cached(
            self.mesh, cache_dir=self.cache_dir)
        self.assertEqual(decomposer.calls, 2)
        self.assertEqual(len(parts), len(expected))
        # the rebuild repaired the entry
        self._patch(_explode)
        convex_decomposition_cached(self.mesh, cache_dir=self.cache_dir)

    def test_manifest_of_a_foreign_version_is_rebuilt(self):
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        convex_decomposition_cached(self.mesh, cache_dir=self.cache_dir)
        entry = os.path.join(self.cache_dir, self._entries()[0])
        manifest_path = os.path.join(entry, 'parts.json')
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest['version'] = cd_module._CACHE_VERSION + 99
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        convex_decomposition_cached(self.mesh, cache_dir=self.cache_dir)
        self.assertEqual(decomposer.calls, 2)

    def test_missing_part_file_is_rebuilt(self):
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        convex_decomposition_cached(self.mesh, cache_dir=self.cache_dir)
        entry = os.path.join(self.cache_dir, self._entries()[0])
        os.remove(os.path.join(entry, 'part_1.npz'))

        parts = convex_decomposition_cached(
            self.mesh, cache_dir=self.cache_dir)
        self.assertEqual(decomposer.calls, 2)
        self.assertEqual(len(parts), 2)

    def test_truncated_part_file_is_rebuilt(self):
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        convex_decomposition_cached(self.mesh, cache_dir=self.cache_dir)
        entry = os.path.join(self.cache_dir, self._entries()[0])
        with open(os.path.join(entry, 'part_0.npz'), 'wb') as f:
            f.write(b'PK\x03\x04 truncated')

        parts = convex_decomposition_cached(
            self.mesh, cache_dir=self.cache_dir)
        self.assertEqual(decomposer.calls, 2)
        self.assertEqual(len(parts), 2)

    def test_mesh_path_is_accepted_and_hit_skips_loading_the_mesh(self):
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        tmp_dir = tempfile.mkdtemp(prefix='skrobot_coacd_src_')
        original_load = trimesh.load
        try:
            path = os.path.join(tmp_dir, 'mesh.stl')
            self.mesh.export(path)
            convex_decomposition_cached(path, cache_dir=self.cache_dir)
            self.assertEqual(decomposer.calls, 1)
            # a path is keyed on its bytes, so a hit costs one file read and
            # neither parses the mesh nor runs the decomposer
            self._patch(_explode)
            trimesh.load = _explode_load
            parts = convex_decomposition_cached(
                path, cache_dir=self.cache_dir)
            self.assertEqual(len(parts), 2)
        finally:
            trimesh.load = original_load
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_mesh_path_and_loaded_mesh_are_hashed_separately(self):
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        tmp_dir = tempfile.mkdtemp(prefix='skrobot_coacd_src_')
        try:
            path = os.path.join(tmp_dir, 'mesh.stl')
            self.mesh.export(path)
            convex_decomposition_cached(path, cache_dir=self.cache_dir)
            convex_decomposition_cached(
                trimesh.load(path, force='mesh'), cache_dir=self.cache_dir)
            self.assertEqual(decomposer.calls, 2)
            self.assertEqual(len(self._entries()), 2)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_unknown_quality_raises_and_writes_nothing(self):
        self._patch(_explode)
        with self.assertRaises(ValueError):
            convex_decomposition_cached(
                self.mesh, quality='ultra', cache_dir=self.cache_dir)
        self.assertEqual(self._entries(), [])

    def test_default_cache_dir_follows_skrobot_cache_dir(self):
        decomposer = _CountingDecomposer()
        self._patch(decomposer)
        original = os.environ.get('SKROBOT_CACHE_DIR')
        os.environ['SKROBOT_CACHE_DIR'] = self.cache_dir
        try:
            convex_decomposition_cached(self.mesh)
            expected = cd_module._default_cache_dir()
        finally:
            if original is None:
                del os.environ['SKROBOT_CACHE_DIR']
            else:
                os.environ['SKROBOT_CACHE_DIR'] = original
        self.assertTrue(expected.startswith(self.cache_dir))
        self.assertEqual(decomposer.calls, 1)
        self.assertEqual(len(os.listdir(expected)), 1)

    def test_concurrent_calls_decompose_once_and_agree(self):
        decomposer = _CountingDecomposer(delay=0.05)
        self._patch(decomposer)
        results = {}
        errors = []

        def worker(i):
            try:
                results[i] = convex_decomposition_cached(
                    self.mesh, cache_dir=self.cache_dir)
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(decomposer.calls, 1)
        self.assertEqual(len(results), 8)
        self.assertEqual(len(self._entries()), 1)
        reference = results[0]
        for parts in results.values():
            self.assertEqual(len(parts), len(reference))
            for a, b in zip(reference, parts):
                np.testing.assert_allclose(a.vertices, b.vertices)
                np.testing.assert_array_equal(a.faces, b.faces)

        # the entry the threads left behind is intact and complete
        entry = os.path.join(self.cache_dir, self._entries()[0])
        self.assertEqual(
            sorted(os.listdir(entry)),
            ['part_0.npz', 'part_1.npz', 'parts.json'])
        self._patch(_explode)
        convex_decomposition_cached(self.mesh, cache_dir=self.cache_dir)

    def test_concurrent_calls_on_distinct_meshes_build_distinct_entries(self):
        decomposer = _CountingDecomposer(delay=0.01)
        self._patch(decomposer)
        meshes = [trimesh.creation.box(extents=(1.0, 1.0, 1.0 + 0.1 * i))
                  for i in range(6)]
        errors = []

        def worker(mesh):
            try:
                convex_decomposition_cached(mesh, cache_dir=self.cache_dir)
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(m,))
                   for m in meshes for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertEqual(decomposer.calls, len(meshes))
        self.assertEqual(len(self._entries()), len(meshes))


if __name__ == '__main__':
    unittest.main()

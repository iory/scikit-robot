import unittest

import numpy as np

from skrobot.pycompat import HAS_JAX


def requires_jax(test_func):
    return unittest.skipUnless(HAS_JAX, 'JAX not available')(test_func)


def _fk_chain_factory(robot_model, link_list, backend_name):
    """Return get_link_transforms(angles, base_pos, base_rot) for one chain.

    The returned callable matches the floating-base contract expected
    by :func:`build_world_centroid_fn`: ``base_pos`` / ``base_rot``
    represent the world pose of ``root_link``.  The chain's natural
    parent is captured in ``root_link`` frame and composed with the
    floating-base pose at call time.

    ``backend_name`` is ``'numpy'`` or ``'jax'``.
    """
    from skrobot.backend import get_backend
    from skrobot.kinematics.differentiable import extract_fk_parameters
    from skrobot.kinematics.differentiable import forward_kinematics

    move_target = link_list[-1]
    fk_params = extract_fk_parameters(robot_model, link_list, move_target)
    backend = get_backend(backend_name)

    base_link = robot_model.root_link
    root_T = base_link.worldcoords().T()
    root_T_inv = np.linalg.inv(root_T)
    natural_pos = fk_params['base_position']
    natural_rot = fk_params['base_rotation']
    rel_pos = root_T_inv[:3, :3] @ natural_pos + root_T_inv[:3, 3]
    rel_rot = root_T_inv[:3, :3] @ natural_rot

    def get_link_transforms(angles, base_pos, base_rot):
        new_base_pos = base_pos + base_rot @ rel_pos
        new_base_rot = base_rot @ rel_rot
        params = dict(fk_params)
        params['base_position'] = np.asarray(new_base_pos)
        params['base_rotation'] = np.asarray(new_base_rot)
        return forward_kinematics(backend, angles, params)

    return get_link_transforms, fk_params


class TestExtractRobotCentroidParams(unittest.TestCase):
    """extract_robot_centroid_params bookkeeping."""

    def test_no_chains_total_mass_matches_robot(self):
        from skrobot.dynamics import extract_robot_centroid_params
        from skrobot.models import Fetch

        fetch = Fetch()
        params = extract_robot_centroid_params(fetch, link_lists=None)

        expected_total = sum(
            link.mass for link in fetch.link_list
            if getattr(link, 'mass', 0.0) and link.mass > 0
        )

        self.assertEqual(params['n_chains'], 0)
        self.assertEqual(len(params['chain_data']), 0)
        self.assertAlmostEqual(params['total_mass'], expected_total, places=8)
        n_fixed = params['fixed_link_masses'].shape[0]
        self.assertGreater(n_fixed, 0)
        self.assertEqual(params['fixed_link_body_pos'].shape, (n_fixed, 3))
        self.assertEqual(params['fixed_link_body_rot'].shape, (n_fixed, 3, 3))

    def test_single_chain_split(self):
        from skrobot.dynamics import extract_robot_centroid_params
        from skrobot.models import Fetch

        fetch = Fetch()
        rarm_links = list(fetch.rarm.link_list)
        params = extract_robot_centroid_params(fetch, link_lists=rarm_links)

        chain_link_ids = {id(link) for link in rarm_links
                          if getattr(link, 'mass', 0.0) > 0}
        chain_indices = params['chain_data'][0]['link_indices']

        self.assertEqual(params['n_chains'], 1)
        self.assertEqual(chain_indices.shape[0], len(chain_link_ids))

        n_fixed = params['fixed_link_masses'].shape[0]
        chain_mass = float(params['chain_data'][0]['masses'].sum())
        fixed_mass = float(params['fixed_link_masses'].sum())
        self.assertAlmostEqual(chain_mass + fixed_mass,
                               params['total_mass'], places=8)
        self.assertEqual(n_fixed + chain_indices.shape[0],
                         sum(1 for link in fetch.link_list
                             if getattr(link, 'mass', 0.0) > 0))

    def test_multi_chain_split(self):
        from skrobot.dynamics import extract_robot_centroid_params
        from skrobot.models import PR2

        pr2 = PR2()
        rarm_links = list(pr2.rarm.link_list)
        larm_links = list(pr2.larm.link_list)
        params = extract_robot_centroid_params(
            pr2, link_lists=[rarm_links, larm_links])

        self.assertEqual(params['n_chains'], 2)
        self.assertEqual(len(params['chain_data']), 2)
        c0 = params['chain_data'][0]['link_indices'].shape[0]
        c1 = params['chain_data'][1]['link_indices'].shape[0]
        self.assertGreater(c0, 0)
        self.assertGreater(c1, 0)
        chain_mass = (float(params['chain_data'][0]['masses'].sum())
                      + float(params['chain_data'][1]['masses'].sum()))
        fixed_mass = float(params['fixed_link_masses'].sum())
        self.assertAlmostEqual(chain_mass + fixed_mass,
                               params['total_mass'], places=8)


class TestBuildWorldCentroidFnNumpy(unittest.TestCase):
    """build_world_centroid_fn matches update_mass_properties (NumPy)."""

    def test_no_chains_matches_reference(self):
        from skrobot.dynamics import build_world_centroid_fn
        from skrobot.dynamics import extract_robot_centroid_params
        from skrobot.models import Fetch

        fetch = Fetch()
        fetch.reset_pose()

        ref = fetch.update_mass_properties()['total_centroid']

        params = extract_robot_centroid_params(fetch, link_lists=None)
        cog_fn = build_world_centroid_fn(
            params, get_link_transforms=None, backend=np)

        cog = np.asarray(cog_fn())
        np.testing.assert_allclose(cog, ref, atol=1e-8)

    def test_single_chain_matches_reference(self):
        from skrobot.dynamics import build_world_centroid_fn
        from skrobot.dynamics import extract_robot_centroid_params
        from skrobot.models import Fetch

        fetch = Fetch()
        fetch.reset_pose()
        for link in fetch.rarm.link_list:
            joint = link.joint
            mid = 0.5 * (joint.min_angle + joint.max_angle)
            if not np.isfinite(mid):
                mid = 0.0
            joint.joint_angle(np.clip(0.3, joint.min_angle, joint.max_angle))
        fetch.angle_vector()  # propagate FK
        ref = fetch.update_mass_properties()['total_centroid']

        rarm_links = list(fetch.rarm.link_list)
        params = extract_robot_centroid_params(fetch, link_lists=rarm_links)
        get_lt, _ = _fk_chain_factory(fetch, rarm_links, 'numpy')
        cog_fn = build_world_centroid_fn(
            params, get_link_transforms=get_lt, backend=np)

        angles = np.array([link.joint.joint_angle()
                           for link in rarm_links])
        cog = np.asarray(cog_fn([angles]))
        np.testing.assert_allclose(cog, ref, atol=1e-7)

    def test_multi_chain_matches_reference(self):
        from skrobot.dynamics import build_world_centroid_fn
        from skrobot.dynamics import extract_robot_centroid_params
        from skrobot.models import PR2

        pr2 = PR2()
        pr2.reset_pose()
        for link in list(pr2.rarm.link_list) + list(pr2.larm.link_list):
            joint = link.joint
            joint.joint_angle(
                np.clip(0.2, joint.min_angle, joint.max_angle))
        pr2.angle_vector()
        ref = pr2.update_mass_properties()['total_centroid']

        rarm_links = list(pr2.rarm.link_list)
        larm_links = list(pr2.larm.link_list)
        params = extract_robot_centroid_params(
            pr2, link_lists=[rarm_links, larm_links])
        get_lt_r, _ = _fk_chain_factory(pr2, rarm_links, 'numpy')
        get_lt_l, _ = _fk_chain_factory(pr2, larm_links, 'numpy')
        cog_fn = build_world_centroid_fn(
            params, get_link_transforms=[get_lt_r, get_lt_l], backend=np)

        ang_r = np.array([link.joint.joint_angle() for link in rarm_links])
        ang_l = np.array([link.joint.joint_angle() for link in larm_links])
        cog = np.asarray(cog_fn([ang_r, ang_l]))
        np.testing.assert_allclose(cog, ref, atol=1e-7)

    def test_base_translation_shifts_cog(self):
        from skrobot.dynamics import build_world_centroid_fn
        from skrobot.dynamics import extract_robot_centroid_params
        from skrobot.models import Fetch

        fetch = Fetch()
        fetch.reset_pose()

        params = extract_robot_centroid_params(fetch, link_lists=None)
        cog_fn = build_world_centroid_fn(
            params, get_link_transforms=None, backend=np)

        cog0 = np.asarray(cog_fn())
        shift = np.array([0.1, -0.2, 0.05])
        cog_shifted = np.asarray(cog_fn(base_pos=shift))
        np.testing.assert_allclose(cog_shifted - cog0, shift, atol=1e-8)

    def test_base_rotation_rotates_cog_about_origin(self):
        from skrobot.coordinates.math import rotation_matrix
        from skrobot.dynamics import build_world_centroid_fn
        from skrobot.dynamics import extract_robot_centroid_params
        from skrobot.models import Fetch

        fetch = Fetch()
        fetch.reset_pose()

        params = extract_robot_centroid_params(fetch, link_lists=None)
        cog_fn = build_world_centroid_fn(
            params, get_link_transforms=None, backend=np)

        cog0 = np.asarray(cog_fn())
        R = rotation_matrix(np.deg2rad(35), 'z')
        cog_rot = np.asarray(cog_fn(base_rot=R))
        np.testing.assert_allclose(cog_rot, R @ cog0, atol=1e-8)


class TestBuildWorldCentroidFnJax(unittest.TestCase):
    """JAX backend matches NumPy backend numerically."""

    @requires_jax
    def test_jax_matches_numpy_no_chains(self):
        import jax.numpy as jnp

        from skrobot.dynamics import build_world_centroid_fn
        from skrobot.dynamics import extract_robot_centroid_params
        from skrobot.models import Fetch

        fetch = Fetch()
        fetch.reset_pose()
        params = extract_robot_centroid_params(fetch, link_lists=None)

        cog_np = np.asarray(build_world_centroid_fn(
            params, get_link_transforms=None, backend=np)())
        cog_jax = np.asarray(build_world_centroid_fn(
            params, get_link_transforms=None, backend=jnp)())
        # JAX default precision is float32, so allow a small mismatch.
        np.testing.assert_allclose(cog_jax, cog_np, atol=1e-5)

    @requires_jax
    def test_jax_matches_numpy_single_chain(self):
        import jax.numpy as jnp

        from skrobot.dynamics import build_world_centroid_fn
        from skrobot.dynamics import extract_robot_centroid_params
        from skrobot.models import Fetch

        fetch = Fetch()
        fetch.reset_pose()
        for link in fetch.rarm.link_list:
            joint = link.joint
            joint.joint_angle(
                np.clip(0.3, joint.min_angle, joint.max_angle))
        fetch.angle_vector()

        rarm_links = list(fetch.rarm.link_list)
        params = extract_robot_centroid_params(fetch, link_lists=rarm_links)
        get_lt_np, _ = _fk_chain_factory(fetch, rarm_links, 'numpy')
        get_lt_jax, _ = _fk_chain_factory(fetch, rarm_links, 'jax')

        angles = np.array([link.joint.joint_angle()
                           for link in rarm_links])

        cog_np = np.asarray(build_world_centroid_fn(
            params, get_link_transforms=get_lt_np, backend=np)([angles]))
        cog_jax = np.asarray(build_world_centroid_fn(
            params, get_link_transforms=get_lt_jax, backend=jnp)(
                [jnp.asarray(angles)]))
        np.testing.assert_allclose(cog_jax, cog_np, atol=1e-6)


if __name__ == '__main__':
    unittest.main()

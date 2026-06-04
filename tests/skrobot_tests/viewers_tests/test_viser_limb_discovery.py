import unittest

import skrobot
from skrobot.viewers import ViserViewer


def _viewer():
    # Borrow the limb-discovery methods without starting a viser server.
    return ViserViewer.__new__(ViserViewer)


class TestViserLimbDiscovery(unittest.TestCase):

    def test_discover_model_limbs_includes_whole_body(self):
        robot = skrobot.models.Aero(use_hand=False)
        limbs = _viewer()._discover_model_limbs(robot)

        # Limbs the model declares as RobotModel attributes are discovered,
        # including the lifter-included whole-body chains and the torso.
        for name in ('rarm', 'larm', 'head', 'torso',
                     'rarm_whole_body', 'larm_whole_body'):
            self.assertIn(name, limbs)

        # whole-body chains include the parallel-link lifter, so they have
        # more joints than the arm-only chains.
        arm_links, _ = limbs['rarm']
        whole_links, whole_ec = limbs['rarm_whole_body']
        self.assertGreater(len(whole_links), len(arm_links))
        self.assertEqual(whole_ec.name, 'rarm_end_coords')

    def test_resolve_ik_groups_dedups_aliases(self):
        robot = skrobot.models.Aero(use_hand=False)
        # With no geometry-detected groups, resolution is driven purely by the
        # model's declared limbs.
        resolved = _viewer()._resolve_ik_groups(robot, {})

        # The lifter-included whole-body group is exposed.
        self.assertIn('rarm_whole_body', resolved)

        # Aliases that resolve to the same joint set (arm / right_arm / rarm)
        # collapse to a single group, keeping the canonical name.
        self.assertNotIn('right_arm', resolved)
        self.assertNotIn('arm', resolved)
        self.assertIn('rarm', resolved)

        # Torso-only chains are filtered out (same rule as geometry groups),
        # while the >=3-link head chain is kept.
        self.assertNotIn('torso', resolved)
        self.assertIn('head', resolved)

        # No two groups share the same movable-joint set.
        keys = []
        for _, (link_list, _) in resolved.items():
            keys.append(frozenset(
                id(link.joint) for link in link_list
                if getattr(link, 'joint', None) is not None))
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == '__main__':
    unittest.main()

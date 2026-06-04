"""Tests for LLM-assisted limb grouping (skrobot.urdf.llm_grouping).

A mock ``llm_fn`` is used so the tests are deterministic and need no network.
"""

import importlib.util
import json
import sys
import tempfile
import unittest

import numpy as np
import pytest

from skrobot.coordinates.math import rotation_matrix_from_rpy
from skrobot.data import panda_urdfpath
from skrobot.model import RobotModel
from skrobot.urdf import build_grouping_prompt
from skrobot.urdf import compute_jaw_gripper_frame
from skrobot.urdf import generate_robot_class_from_geometry
from skrobot.urdf import parse_grouping_response


ARM_LINKS = ['panda_link{}'.format(i) for i in range(1, 8)]


def _panda():
    robot = RobotModel()
    robot.load_urdf_file(panda_urdfpath())
    return robot


def _arm_grouping_llm(prompt):
    """Mock LLM that groups the Panda arm into a single ``arm`` limb."""
    limbs = [{'name': 'arm', 'links': ARM_LINKS, 'tip_link': 'panda_hand'}]
    return '```json\n' + json.dumps({'limbs': limbs}) + '\n```'


@pytest.mark.skipif(
    sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
    reason="Skip in Python 2 and Python 3.6")
class TestParseGroupingResponse(unittest.TestCase):

    def setUp(self):
        self.robot = _panda()

    def test_parse_valid(self):
        limbs = parse_grouping_response(_arm_grouping_llm(''), self.robot)
        self.assertEqual(len(limbs), 1)
        self.assertEqual(limbs[0]['name'], 'arm')
        self.assertEqual(limbs[0]['links'], ARM_LINKS)
        self.assertEqual(limbs[0]['tip_link'], 'panda_hand')

    def test_parse_plain_json_without_fences(self):
        text = json.dumps({'limbs': [
            {'name': 'arm', 'links': ARM_LINKS, 'tip_link': 'panda_hand'}]})
        limbs = parse_grouping_response(text, self.robot)
        self.assertEqual(limbs[0]['name'], 'arm')

    def test_non_json_raises(self):
        with self.assertRaises(ValueError):
            parse_grouping_response('I cannot help with that.', self.robot)

    def test_keyword_name_raises(self):
        # 'class'.isidentifier() is True but it is a Python keyword.
        text = json.dumps({'limbs': [
            {'name': 'class', 'links': ARM_LINKS, 'tip_link': 'panda_hand'}]})
        with self.assertRaises(ValueError):
            parse_grouping_response(text, self.robot)

    def test_uppercase_name_raises(self):
        text = json.dumps({'limbs': [
            {'name': 'RightArm', 'links': ARM_LINKS,
             'tip_link': 'panda_hand'}]})
        with self.assertRaises(ValueError):
            parse_grouping_response(text, self.robot)

    def test_unknown_link_raises(self):
        text = json.dumps({'limbs': [
            {'name': 'arm', 'links': ['not_a_real_link'],
             'tip_link': 'panda_hand'}]})
        with self.assertRaises(ValueError):
            parse_grouping_response(text, self.robot)

    def test_non_movable_link_raises(self):
        # panda_link0 is the fixed base, not a movable joint.
        text = json.dumps({'limbs': [
            {'name': 'arm', 'links': ['panda_link0'],
             'tip_link': 'panda_hand'}]})
        with self.assertRaises(ValueError):
            parse_grouping_response(text, self.robot)

    def test_disconnected_tip_raises(self):
        # panda_link0 is the base (an ancestor of the chain), so it is not on
        # the limb and must be rejected as a tip_link.
        text = json.dumps({'limbs': [
            {'name': 'arm', 'links': ARM_LINKS, 'tip_link': 'panda_link0'}]})
        with self.assertRaises(ValueError):
            parse_grouping_response(text, self.robot)

    def test_tip_below_chain_is_accepted(self):
        # panda_hand hangs below the arm chain (it is a descendant of the last
        # arm link), so it is a valid tip even though it is not in 'links'.
        text = json.dumps({'limbs': [
            {'name': 'arm', 'links': ARM_LINKS, 'tip_link': 'panda_hand'}]})
        limbs = parse_grouping_response(text, self.robot)
        self.assertEqual(limbs[0]['tip_link'], 'panda_hand')

    def test_tip_defaults_to_last_link(self):
        text = json.dumps({'limbs': [{'name': 'arm', 'links': ARM_LINKS}]})
        limbs = parse_grouping_response(text, self.robot)
        self.assertEqual(limbs[0]['tip_link'], ARM_LINKS[-1])

    def test_jaw_gripper_end_coords_parsed(self):
        text = json.dumps({'limbs': [
            {'name': 'arm', 'links': ARM_LINKS, 'tip_link': 'panda_hand',
             'end_coords': {'type': 'jaw_gripper', 'wrist_link': 'panda_hand',
                            'jaw_links': ['panda_leftfinger',
                                          'panda_rightfinger']}}]})
        limbs = parse_grouping_response(text, self.robot)
        ec = limbs[0]['end_coords']
        self.assertEqual(ec['kind'], 'jaw_gripper')
        self.assertEqual(ec['wrist_link'], 'panda_hand')
        self.assertEqual(len(ec['jaw_links']), 2)

    def test_jaw_gripper_bad_jaw_raises(self):
        text = json.dumps({'limbs': [
            {'name': 'arm', 'links': ARM_LINKS, 'tip_link': 'panda_hand',
             'end_coords': {'type': 'jaw_gripper', 'wrist_link': 'panda_hand',
                            'jaw_links': ['panda_leftfinger', 'nope']}}]})
        with self.assertRaises(ValueError):
            parse_grouping_response(text, self.robot)


@pytest.mark.skipif(
    sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
    reason="Skip in Python 2 and Python 3.6")
class TestBuildGroupingPrompt(unittest.TestCase):

    def test_prompt_contains_tree_and_links(self):
        prompt = build_grouping_prompt(_panda())
        self.assertIn('Kinematic Tree', prompt)
        self.assertIn('panda_link1', prompt)
        self.assertIn('Actuated chains:', prompt)
        self.assertIn('JSON', prompt)


@pytest.mark.skipif(
    sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
    reason="Skip in Python 2 and Python 3.6")
class TestGenerateWithLLMGrouping(unittest.TestCase):

    def test_generate_and_instantiate(self):
        robot = _panda()
        code = generate_robot_class_from_geometry(
            robot, grouping='llm', llm_fn=_arm_grouping_llm,
            class_name='LlmPanda', urdf_path=panda_urdfpath())
        self.assertIn('def arm(self)', code)

        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            path = f.name

        spec = importlib.util.spec_from_file_location('llm_panda', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        instance = module.LlmPanda()
        self.assertEqual(len(instance.arm.link_list), 7)
        self.assertIsNotNone(instance.arm.end_coords)

    def test_jaw_gripper_end_coords_generated(self):
        # Include the jaw links in 'links' on purpose: they must be stripped
        # from the arm group so IK does not drive the gripper joints.
        grouping = {'limbs': [
            {'name': 'arm',
             'links': ARM_LINKS + ['panda_leftfinger', 'panda_rightfinger'],
             'tip_link': 'panda_hand',
             'end_coords': {'type': 'jaw_gripper', 'wrist_link': 'panda_hand',
                            'jaw_links': ['panda_leftfinger',
                                          'panda_rightfinger']}}]}

        def llm_fn(_prompt):
            return json.dumps(grouping)

        code = generate_robot_class_from_geometry(
            _panda(), grouping='llm', llm_fn=llm_fn,
            class_name='GripPanda', urdf_path=panda_urdfpath())
        # The grasp frame is baked as plain CascadedCoords values (pos + rot);
        # no runtime helper is referenced.
        self.assertNotIn('jaw_gripper_end_coords', code)
        self.assertIn('CascadedCoords(', code)
        self.assertIn('rot=', code)

        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            path = f.name
        spec = importlib.util.spec_from_file_location('grip_panda', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        instance = module.GripPanda()
        self.assertIsNotNone(instance.arm.end_coords)
        # Jaw (finger) joints were stripped: only the 7 arm joints remain.
        self.assertEqual(len(instance.arm.joint_list), 7)

    def test_missing_llm_fn_raises(self):
        with self.assertRaises(TypeError):
            generate_robot_class_from_geometry(
                _panda(), grouping='llm', llm_fn=None)

    def test_invalid_grouping_raises(self):
        with self.assertRaises(ValueError):
            generate_robot_class_from_geometry(_panda(), grouping='nonsense')


def _rotation_deviation_deg(rot_a, rot_b):
    """Geodesic angle (deg) between two 3x3 rotation matrices."""
    rel = np.asarray(rot_a).T.dot(np.asarray(rot_b))
    cos = (np.trace(rel) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


@pytest.mark.skipif(
    sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
    reason="Skip in Python 2 and Python 3.6")
class TestComputeJawGripperFrame(unittest.TestCase):
    """The analyzed jaw-gripper frame must agree with the orientation of the
    hand-written ``end_coords`` shipped in :mod:`skrobot.models`.

    A large rotation deviation (> 5 deg) means the analysis picked the wrong
    closing/approach direction, so it is the regression signal here.
    """

    def _check(self, robot, wrist, jaws, reference, tol_deg=5.0):
        before = robot.angle_vector().copy()
        frame = compute_jaw_gripper_frame(robot, wrist, jaws)
        self.assertEqual(frame['parent_link'], wrist)

        wrist_link = getattr(robot, wrist)
        computed = np.asarray(wrist_link.worldrot()).dot(
            rotation_matrix_from_rpy(frame['rot']))
        deviation = _rotation_deviation_deg(
            np.asarray(reference.worldrot()), computed)
        self.assertLessEqual(
            deviation, tol_deg,
            '{} gripper frame is {:.1f} deg from the reference '
            'end_coords'.format(wrist, deviation))

        # The robot pose (including gripper joints not in angle_vector) must be
        # restored exactly.
        np.testing.assert_allclose(robot.angle_vector(), before)

    def test_panda(self):
        import skrobot.models as models
        robot = models.Panda()
        self._check(robot, 'panda_hand',
                    ['panda_leftfinger', 'panda_rightfinger'],
                    robot.rarm_end_coords)

    def test_pr2_both_arms(self):
        import skrobot.models as models
        robot = models.PR2()
        self._check(robot, 'r_gripper_palm_link',
                    ['r_gripper_l_finger_tip_link',
                     'r_gripper_r_finger_tip_link'],
                    robot.rarm_end_coords)
        self._check(robot, 'l_gripper_palm_link',
                    ['l_gripper_l_finger_tip_link',
                     'l_gripper_r_finger_tip_link'],
                    robot.larm_end_coords)

    def test_fetch(self):
        import skrobot.models as models
        robot = models.Fetch()
        self._check(robot, 'gripper_link',
                    ['l_gripper_finger_link', 'r_gripper_finger_link'],
                    robot.rarm_end_coords)

    def test_jaxon_both_arms(self):
        import skrobot.models as models
        robot = models.JaxonJVRC()
        self._check(robot, 'RARM_LINK7',
                    ['RARM_FINGER0', 'RARM_FINGER1'], robot.rarm_end_coords)
        self._check(robot, 'LARM_LINK7',
                    ['LARM_FINGER0', 'LARM_FINGER1'], robot.larm_end_coords)


if __name__ == '__main__':
    unittest.main()

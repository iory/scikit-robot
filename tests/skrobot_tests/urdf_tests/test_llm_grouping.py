"""Tests for LLM-assisted limb grouping (skrobot.urdf.llm_grouping).

A mock ``llm_fn`` is used so the tests are deterministic and need no network.
"""

import importlib.util
import json
import sys
import tempfile
import unittest

import pytest

from skrobot.data import panda_urdfpath
from skrobot.model import RobotModel
from skrobot.urdf import build_grouping_prompt
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

    def test_tip_defaults_to_last_link(self):
        text = json.dumps({'limbs': [{'name': 'arm', 'links': ARM_LINKS}]})
        limbs = parse_grouping_response(text, self.robot)
        self.assertEqual(limbs[0]['tip_link'], ARM_LINKS[-1])


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

    def test_missing_llm_fn_raises(self):
        with self.assertRaises(TypeError):
            generate_robot_class_from_geometry(
                _panda(), grouping='llm', llm_fn=None)

    def test_invalid_grouping_raises(self):
        with self.assertRaises(ValueError):
            generate_robot_class_from_geometry(_panda(), grouping='nonsense')


if __name__ == '__main__':
    unittest.main()

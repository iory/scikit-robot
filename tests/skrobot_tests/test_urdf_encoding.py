"""A URDF is UTF-8; loading it must not depend on the platform default.

A URDF declares its own encoding and defaults to UTF-8. Reading one with the
platform default encoding instead (``cp932`` on a Japanese Windows) either
raises or silently returns the wrong characters, before the parser ever sees
the declaration.
"""

import ast
import os
import unittest

from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF
import skrobot.utils.urdf


# Half-width katakana: the UTF-8 bytes start with 0xEF, which is not a
# valid cp932 lead byte, so reading them as cp932 raises.
ALUMINIUM = u'ｱﾙﾐ'

# Full-width: the UTF-8 bytes are valid cp932 and decode to something else,
# so reading them as cp932 gives a name that is wrong without saying so.
RUBBER = u'ゴム'

URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data',
    'japanese_material.urdf')


def material_names(robot_model):
    return [m.name for m in robot_model.urdf_robot_model.materials]


class TestURDFEncoding(unittest.TestCase):

    def test_load_from_path(self):
        robot_model = RobotModelFromURDF(urdf_file=URDF_PATH)
        self.assertEqual([ALUMINIUM, RUBBER],
                         material_names(robot_model))

    def test_load_from_utf8_text_stream(self):
        robot_model = RobotModel()
        with open(URDF_PATH, 'r', encoding='utf-8') as f:
            robot_model.load_urdf_file(f)
        self.assertEqual([ALUMINIUM, RUBBER],
                         material_names(robot_model))

    def test_load_from_binary_stream(self):
        robot_model = RobotModel()
        with open(URDF_PATH, 'rb') as f:
            robot_model.load_urdf_file(f)
        self.assertEqual([ALUMINIUM, RUBBER],
                         material_names(robot_model))

    def test_load_from_cp932_text_stream(self):
        # What ``open(path)`` gives you on a Japanese Windows. The parser
        # must read the undecoded bytes and honour the declaration.
        robot_model = RobotModel()
        with open(URDF_PATH, 'r', encoding='cp932') as f:
            robot_model.load_urdf_file(f)
        self.assertEqual([ALUMINIUM, RUBBER],
                         material_names(robot_model))

    def test_from_urdf(self):
        robot_model = RobotModel.from_urdf(URDF_PATH)
        self.assertEqual('japanese_material', robot_model.name)
        self.assertEqual([ALUMINIUM, RUBBER],
                         material_names(robot_model))

    def test_load_urdf_string(self):
        robot_model = RobotModel()
        with open(URDF_PATH, 'r', encoding='utf-8') as f:
            robot_model.load_urdf(f.read())
        self.assertEqual([ALUMINIUM, RUBBER],
                         material_names(robot_model))


class TestNoDefaultEncoding(unittest.TestCase):
    """Every text file skrobot reads or writes must name its encoding."""

    @staticmethod
    def _keyword(call, name):
        for keyword in call.keywords:
            if keyword.arg == name:
                return keyword
        return None

    @classmethod
    def _mode(cls, call, position, default):
        keyword = cls._keyword(call, 'mode')
        if keyword is not None:
            node = keyword.value
        elif len(call.args) > position:
            node = call.args[position]
        else:
            return default
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None  # computed at runtime; nothing to check statically

    @staticmethod
    def _function_name(call):
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return ''

    def _offenders(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), path)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = self._function_name(node)
            if name == 'open' and isinstance(node.func, ast.Name):
                mode = self._mode(node, 1, 'r')
            elif name in ('NamedTemporaryFile', 'TemporaryFile'):
                mode = self._mode(node, 0, 'w+b')
            elif name in ('read_text', 'write_text'):
                mode = 'r'
            else:
                continue
            if mode is None or 'b' in mode:
                continue
            if self._keyword(node, 'encoding') is None:
                offenders.append('{}:{}: {}()'.format(path, node.lineno, name))
        return offenders

    def test_no_open_without_encoding(self):
        package_dir = os.path.dirname(
            os.path.abspath(skrobot.utils.urdf.__file__))
        package_dir = os.path.dirname(package_dir)
        offenders = []
        for dirpath, _, filenames in os.walk(package_dir):
            for filename in sorted(filenames):
                if filename.endswith('.py'):
                    offenders.extend(
                        self._offenders(os.path.join(dirpath, filename)))
        self.assertEqual(
            [], offenders,
            'text-mode file I/O without an explicit encoding falls back to '
            'the platform default:\n' + '\n'.join(offenders))

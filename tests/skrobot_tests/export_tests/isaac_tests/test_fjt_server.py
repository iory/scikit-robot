import ast
import unittest

from skrobot.export.isaac import generate_fjt_server
from skrobot.export.isaac import write_fjt_server


class TestGenerateFjtServer(unittest.TestCase):

    CONTROLLERS = {
        'arm_controller': ['j0', 'j1', 'j2'],
        'head_controller': ['head_pan', 'head_tilt'],
    }

    @staticmethod
    def _get_execute_function(tree):
        class_node = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == 'FJTServer':
                class_node = node
                break
        if class_node is None:
            raise AssertionError('FJTServer class not found')

        make_execute = None
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '_make_execute':
                make_execute = node
                break
        if make_execute is None:
            raise AssertionError('_make_execute method not found')

        for node in make_execute.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'execute':
                return node
        raise AssertionError('execute callback not found')

    @staticmethod
    def _has_method_call(nodes, object_name, method_name):
        for node in nodes:
            for subnode in ast.walk(node):
                if (isinstance(subnode, ast.Call)
                        and isinstance(subnode.func, ast.Attribute)
                        and isinstance(subnode.func.value, ast.Name)
                        and subnode.func.value.id == object_name
                        and subnode.func.attr == method_name):
                    return True
        return False

    @staticmethod
    def _has_result_error_code_assignment(nodes, code_name):
        for node in nodes:
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == 'result'
                    and target.attr == 'error_code'):
                continue
            value = node.value
            if (isinstance(value, ast.Attribute)
                    and value.attr == code_name
                    and isinstance(value.value, ast.Attribute)
                    and value.value.attr == 'Result'
                    and isinstance(value.value.value, ast.Name)
                    and value.value.value.id == 'FollowJointTrajectory'):
                return True
        return False

    def test_output_is_valid_python(self):
        src = generate_fjt_server(self.CONTROLLERS)
        compile(src, '<gen>', 'exec')
        ast.parse(src)

    def test_controllers_and_topics_are_embedded(self):
        src = generate_fjt_server(self.CONTROLLERS,
                                  joint_command_topic='/robot/joint_command')
        self.assertIn('arm_controller', src)
        self.assertIn('head_pan', src)
        self.assertIn('/robot/joint_command', src)
        self.assertIn('follow_joint_trajectory', src)

    def test_parameters_flow_through(self):
        src = generate_fjt_server(self.CONTROLLERS, node_name='my_fjt',
                                  publish_hz=50.0, goal_tolerance=0.02)
        self.assertIn("NODE_NAME = 'my_fjt'", src)
        self.assertIn('PUBLISH_HZ = 50.0', src)
        self.assertIn('GOAL_TOLERANCE = 0.02', src)

    def test_empty_controllers_rejected(self):
        with self.assertRaises(ValueError):
            generate_fjt_server({})
        with self.assertRaises(ValueError):
            generate_fjt_server({'c': []})

    def test_execute_rejects_joints_outside_owned_set(self):
        src = generate_fjt_server(self.CONTROLLERS)
        execute = self._get_execute_function(ast.parse(src))

        invalid_names_guard = None
        for node in execute.body:
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
                if node.test.id == 'invalid_names':
                    invalid_names_guard = node
                    break

        if invalid_names_guard is None:
            self.fail('missing invalid_names guard in execute callback')

        self.assertTrue(self._has_method_call(invalid_names_guard.body,
                                              'goal_handle', 'abort'))
        self.assertTrue(self._has_result_error_code_assignment(
            invalid_names_guard.body, 'INVALID_JOINTS'))

    def test_settle_timeout_aborts_goal(self):
        src = generate_fjt_server(self.CONTROLLERS)
        execute = self._get_execute_function(ast.parse(src))

        timeout_guard = None
        for node in ast.walk(execute):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not isinstance(test, ast.Compare):
                continue
            if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Gt):
                continue
            if len(test.comparators) != 1:
                continue
            if (isinstance(test.left, ast.Name)
                    and test.left.id == 'elapsed'
                    and isinstance(test.comparators[0], ast.Name)
                    and test.comparators[0].id == 'SETTLE_TIMEOUT'):
                timeout_guard = node
                break

        if timeout_guard is None:
            self.fail('missing settle-timeout guard in execute callback')

        self.assertTrue(self._has_method_call(timeout_guard.body,
                                              'goal_handle', 'abort'))
        self.assertFalse(self._has_method_call(timeout_guard.body,
                                               'goal_handle', 'succeed'))
        self.assertTrue(self._has_result_error_code_assignment(
            timeout_guard.body, 'GOAL_TOLERANCE_VIOLATED'))

    def test_write_fjt_server(self):
        import os
        import tempfile
        d = tempfile.mkdtemp()
        try:
            p = write_fjt_server(self.CONTROLLERS, os.path.join(d, 'srv.py'))
            self.assertTrue(os.path.exists(p))
            ast.parse(open(p).read())
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()

import os
import os.path as osp
import subprocess
import sys
import tempfile
import unittest

import pytest

from skrobot.data import fetch_urdfpath
from skrobot.data import kuka_urdfpath
from skrobot.urdf import URDFXMLRootLinkChanger


class TestConsoleScripts(unittest.TestCase):

    @pytest.mark.skipif(
        sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
        reason="Skip in Python 2 and Python 3.6")
    def test_convert_urdf_mesh(self):
        with tempfile.TemporaryDirectory() as tmp_output:
            os.environ['SKROBOT_CACHE_DIR'] = tmp_output
            urdfpath = fetch_urdfpath()
            urdf_dir = osp.dirname(urdfpath)

            out_urdfpath = osp.join(urdf_dir, 'fetch_0.urdf')
            out_stl_urdfpath = osp.join(
                urdf_dir, 'fetch_stl.urdf')

            relative_input_path = osp.join(osp.basename(urdf_dir), osp.basename(urdfpath))
            relative_output_path = osp.join(osp.basename(urdf_dir), osp.basename(out_urdfpath))

            cmds = [
                'convert-urdf-mesh {}'.format(urdfpath),
                'convert-urdf-mesh {} --voxel-size 0.001'.format(urdfpath),
                'convert-urdf-mesh {} -d 0.98'.format(urdfpath),
                # inplace option should be used last
                'convert-urdf-mesh {} --output {} -f stl'.format(
                    out_urdfpath, out_stl_urdfpath),
                'convert-urdf-mesh {} --inplace'.format(out_urdfpath),
                'cd .. && convert-urdf-mesh {} --output {}'.format(
                    relative_input_path, relative_output_path),
            ]
            failures = []
            env = os.environ.copy()

            for cmd in cmds:
                print("Executing: {}".format(cmd))

                exec_dir = osp.dirname(urdf_dir) if 'cd ..' in cmd else urdf_dir

                result = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=exec_dir
                )

                if result.returncode != 0:
                    failures.append(
                        (
                            cmd,
                            result.returncode,
                            result.stdout.decode(),
                            result.stderr.decode(),
                        )
                    )

            if failures:
                messages = []
                for cmd, code, stdout, stderr in failures:
                    messages.append(
                        "Command '{}' failed with exit code ".format(cmd)
                        + "{}\nstdout:\n{}\nstderr:\n{}".format(
                            code, stdout, stderr)
                    )
                self.fail("\n\n".join(messages))

    @pytest.mark.skipif(
        sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
        reason="Skip in Python 2 and Python 3.6")
    def test_change_urdf_root(self):
        """Test change-urdf-root command line script."""
        urdf_path = kuka_urdfpath()

        # Get available links for testing
        changer = URDFXMLRootLinkChanger(urdf_path)
        available_links = changer.list_links()
        current_root = changer.get_current_root_link()

        # Find a different link to use as new root
        new_root = None
        for link in available_links:
            if link != current_root:
                new_root = link
                break

        with tempfile.TemporaryDirectory() as tmp_output:
            def _run_command(args):
                cmd = [sys.executable, '-m',
                       'skrobot.apps.change_urdf_root'] + args
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(urdf_path)
                )
                return result.returncode, result.stdout, result.stderr

            # Test --list option
            returncode, stdout, stderr = _run_command([urdf_path, '--list'])
            self.assertEqual(returncode, 0)
            self.assertIn('Current root link:', stdout)
            self.assertIn('Total links:', stdout)
            self.assertIn('All links:', stdout)
            self.assertIn('(current root)', stdout)
            for link in available_links:
                self.assertIn(link, stdout)

            # Test --help option
            returncode, stdout, stderr = _run_command(['--help'])
            self.assertEqual(returncode, 0)
            self.assertIn('Change the root link of a URDF file', stdout)
            self.assertIn('Examples:', stdout)
            self.assertIn('change_urdf_root', stdout)

            # Test missing input file
            returncode, stdout, stderr = _run_command([
                'non_existent_file.urdf', 'some_link', 'output.urdf'
            ])
            self.assertNotEqual(returncode, 0)
            self.assertIn('not found', stderr)

            # Test missing required arguments
            returncode, stdout, stderr = _run_command([urdf_path])
            self.assertNotEqual(returncode, 0)
            self.assertIn('new_root_link is required', stderr)

            # Test invalid link name
            output_path = os.path.join(tmp_output, 'invalid_test.urdf')
            returncode, stdout, stderr = _run_command([
                urdf_path, 'non_existent_link', output_path
            ])
            self.assertNotEqual(returncode, 0)
            self.assertIn('not found in URDF', stderr)
            self.assertIn('Available links:', stderr)

            # Test successful root change (if we have alternative links)
            if new_root is not None:
                output_path = os.path.join(tmp_output, 'changed_root.urdf')
                returncode, stdout, stderr = _run_command([
                    urdf_path, new_root, output_path
                ])
                self.assertEqual(returncode, 0)
                self.assertIn('Successfully changed root link', stdout)
                self.assertIn("from '{}' to '{}'".format(
                    current_root, new_root), stdout)

                # Verify the output file exists
                self.assertTrue(os.path.exists(output_path))

                # Verify the change by loading the result
                result_changer = URDFXMLRootLinkChanger(output_path)
                actual_root = result_changer.get_current_root_link()
                self.assertEqual(actual_root, new_root)

                # Test verbose option
                output_path2 = os.path.join(tmp_output, 'verbose_test.urdf')
                returncode, stdout, stderr = _run_command([
                    urdf_path, new_root, output_path2, '--verbose'
                ])
                self.assertEqual(returncode, 0)
                self.assertIn('Loading URDF:', stdout)
                self.assertIn('Current root link:', stdout)
                self.assertIn('New root link:', stdout)
                self.assertIn('Output file:', stdout)
                self.assertIn('Changing root link...', stdout)
                self.assertIn('Verifying result...', stdout)

                # Test force overwrite option
                output_path3 = os.path.join(tmp_output, 'force_test.urdf')

                # Create a dummy file first
                with open(output_path3, 'w') as f:
                    f.write('dummy content')

                # Without --force, should fail
                returncode, stdout, stderr = _run_command([
                    urdf_path, new_root, output_path3
                ])
                self.assertNotEqual(returncode, 0)
                self.assertIn('already exists', stderr)

                # With --force, should succeed
                returncode, stdout, stderr = _run_command([
                    urdf_path, new_root, output_path3, '--force'
                ])
                self.assertEqual(returncode, 0)
                self.assertIn('Successfully changed root link', stdout)

            # Test changing to the same root link
            output_path4 = os.path.join(tmp_output, 'same_root.urdf')
            returncode, stdout, stderr = _run_command([
                urdf_path, current_root, output_path4, '--verbose'
            ])
            self.assertEqual(returncode, 0)
            self.assertIn('same as current root link', stdout)
            self.assertIn('Successfully changed root link', stdout)

            # Verify the output file exists and has correct root
            self.assertTrue(os.path.exists(output_path4))
            result_changer = URDFXMLRootLinkChanger(output_path4)
            actual_root = result_changer.get_current_root_link()
            self.assertEqual(actual_root, current_root)

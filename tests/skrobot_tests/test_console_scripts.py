import os
import os.path as osp
import subprocess
import sys
import tempfile
import unittest

import pytest

from skrobot.data import fetch_urdfpath


class TestConsoleScripts(unittest.TestCase):

    @pytest.mark.skipif(
        sys.version_info[0] == 2 or sys.version_info[:2] == (3, 6),
        reason="Skip in Python 2 and Python 3.6")
    def test_convert_urdf_mesh(self):
        with tempfile.TemporaryDirectory() as tmp_output:
            os.environ['SKROBOT_CACHE_DIR'] = tmp_output
            urdfpath = fetch_urdfpath()

            out_urdfpath = osp.join(osp.dirname(urdfpath), 'fetch_0.urdf')
            out_stl_urdfpath = osp.join(
                osp.dirname(urdfpath), 'fetch_stl.urdf')

            cmds = [
                'convert-urdf-mesh {}'.format(urdfpath),
                'convert-urdf-mesh {} --voxel-size 0.001'.format(urdfpath),
                'convert-urdf-mesh {} -d 0.98'.format(urdfpath),
                # inplace option should be used last
                'convert-urdf-mesh {} --output {} -f stl'.format(
                    out_urdfpath, out_stl_urdfpath),
                'convert-urdf-mesh {} --inplace'.format(out_urdfpath),
            ]

            failures = []
            env = os.environ.copy()

            for cmd in cmds:
                print("Executing: {}".format(cmd))
                result = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env
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

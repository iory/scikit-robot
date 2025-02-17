import os
import os.path as osp
import subprocess
import sys
import tempfile
import unittest

import pytest

from skrobot.data import fetch_urdfpath


class TestConsoleScripts(unittest.TestCase):

    @pytest.mark.skipif(sys.version_info[0] == 2, reason="Skip in Python 2")
    def test_convert_urdf_mesh(self):
        tmp_output = tempfile.TemporaryDirectory()
        os.environ['SKROBOT_CACHE_DIR'] = tmp_output.name
        urdfpath = fetch_urdfpath()

        # fetch_0.urdf will be create after f'convert-urdf-mesh {urdfpath}'
        out_urdfpath = osp.join(osp.dirname(urdfpath), 'fetch_0.urdf')
        out_stl_urdfpath = osp.join(osp.dirname(urdfpath), 'fetch_stl.urdf')

        cmds = ['convert-urdf-mesh {}'.format(urdfpath),
                'convert-urdf-mesh {} --output {} -f stl'.format(
                    out_urdfpath,
                    out_stl_urdfpath),
                'convert-urdf-mesh {} --voxel-size 0.001'.format(urdfpath),
                'convert-urdf-mesh {} -d 0.98'.format(urdfpath),
                # inplace option should be used at last
                'convert-urdf-mesh {} --inplace'.format(out_urdfpath),
                ]
        kwargs = {}
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

        for cmd in cmds:
            print('Executing {}'.format(cmd))
            result = subprocess.run(cmd,
                                    shell=True,
                                    **kwargs)
            assert result.returncode == 0

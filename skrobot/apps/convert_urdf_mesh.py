#!/usr/bin/env python

import argparse
import os.path as osp
from pathlib import Path
import shutil

from skrobot.model import RobotModel
from skrobot.utils.urdf import export_mesh_format


def main():
    parser = argparse.ArgumentParser(description='Convert URDF Mesh.')
    parser.add_argument('urdf',
                        help='Path to the input URDF file')
    parser.add_argument('--format', '-f',
                        default='dae',
                        choices=['dae', 'obj', 'stl'],
                        help='Mesh format for export. Default is dae.')
    parser.add_argument('--output', help='Path for the output URDF file. If not specified, a filename is automatically generated based on the input URDF file.')  # NOQA
    parser.add_argument('--inplace', '-i', action='store_true',
                        help='Modify the input URDF file inplace. If not specified, a new file is created.')  # NOQA

    args = parser.parse_args()

    base_path = Path(args.urdf).parent
    urdf_path = Path(args.urdf)

    if args.output is None:
        fn, _ = osp.splitext(args.urdf)
        index = 0
        pattern = fn + "_%i.urdf"
        outfile = pattern % index
        while osp.exists(outfile):
            index += 1
            outfile = pattern % index
        args.output = outfile
    output_path = Path(args.output)

    r = RobotModel()
    with open(base_path / urdf_path) as f:
        r.load_urdf_file(f)

    with export_mesh_format('.' + args.format):
        r.urdf_robot_model.save(str(base_path / output_path))
    if args.inplace:
        shutil.move(str(base_path / output_path),
                    base_path / urdf_path)


if __name__ == '__main__':
    main()

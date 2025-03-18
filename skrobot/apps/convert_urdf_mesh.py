#!/usr/bin/env python

import argparse
import contextlib
from distutils.version import StrictVersion
import os.path as osp
from pathlib import Path
import shutil
import sys

import pkg_resources

from skrobot.model import RobotModel
from skrobot.utils.package import is_package_installed
from skrobot.utils.urdf import export_mesh_format
from skrobot.utils.urdf import force_visual_mesh_origin_to_zero


def main():
    parser = argparse.ArgumentParser(description='Convert URDF Mesh.')
    parser.add_argument('urdf',
                        help='Path to the input URDF file')
    parser.add_argument('--format', '-f',
                        default='dae',
                        choices=['dae', 'stl'],
                        help='Mesh format for export. Default is dae.')
    parser.add_argument('--output', '-o', help='Path for the output URDF file. If not specified, a filename is automatically generated based on the input URDF file.')  # NOQA
    parser.add_argument('--inplace', '-i', action='store_true',
                        help='Modify the input URDF file inplace. If not specified, a new file is created.')  # NOQA
    parser.add_argument('--force-zero-origin', action='store_true',
                        help='Force the visual mesh origin to zero.')
    decimation_help = """
Specifies the minimum area ratio threshold for the mesh simplification process.
This threshold determines the minimum proportion of the original mesh area
that must be preserved in the simplified mesh. It is a float value
between 0 and 1.
A higher value means more of the original mesh area is preserved,
resulting in less simplification. Default is None."""
    parser.add_argument(
        '-d', "--decimation-area-ratio-threshold", type=float, default=None,
        help=decimation_help)
    parser.add_argument(
        '--voxel-size', default=None, type=float,
        help='Specifies the voxel size for the simplify_vertex_clustering'
        ' function in open3d. When this value is provided, '
        'it is used as the voxel size in the function to perform '
        'mesh simplification. This process reduces the complexity'
        ' of the mesh by clustering vertices within the specified voxel size.')

    args = parser.parse_args()

    trimesh_version = pkg_resources.get_distribution("trimesh").version
    if StrictVersion(trimesh_version) < StrictVersion("4.0.10"):
        print(
            '[WARNING] With `trimesh` < 4.0.10, the output dae is not '
            + 'colored. Please `pip install trimesh -U`')
        sys.exit(1)
    if args.decimation_area_ratio_threshold:
        disable_decimation = False
        if is_package_installed('open3d') is False:
            print("[ERROR] open3d is not installed. "
                  + "Please install it as 'pip install scikit-robot[all]' "
                  + "to include open3d or 'pip install open3d'")
            disable_decimation = True
        if is_package_installed('fast-simplification') is False:
            print("[ERROR] fast-simplification is not installed. "
                  + "Please install it with 'pip install fast-simplification'")
            disable_decimation = True
        if disable_decimation:
            sys.exit(1)

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

    if args.force_zero_origin:
        force_visual_mesh_origin_to_zero_or_not = \
            force_visual_mesh_origin_to_zero
    else:
        try:
            from contextlib import nullcontext
        except ImportError:
            # for python3.6
            @contextlib.contextmanager
            def nullcontext(enter_result=None):
                yield enter_result
        force_visual_mesh_origin_to_zero_or_not = nullcontext

    r = RobotModel()
    with open(base_path / urdf_path) as f:
        with force_visual_mesh_origin_to_zero_or_not():
            r.load_urdf_file(f)

    with export_mesh_format(
            '.' + args.format,
            decimation_area_ratio_threshold=args.decimation_area_ratio_threshold,  # NOQA
            simplify_vertex_clustering_voxel_size=args.voxel_size):
        r.urdf_robot_model.save(str(base_path / output_path))

    if args.inplace:
        shutil.move(str(base_path / output_path),
                    base_path / urdf_path)


if __name__ == '__main__':
    main()

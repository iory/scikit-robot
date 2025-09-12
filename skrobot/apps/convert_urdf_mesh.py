#!/usr/bin/env python

import argparse
import contextlib
import os.path as osp
from pathlib import Path
import shutil
import sys

from lxml import etree
from packaging.version import Version

from skrobot import determine_version
from skrobot.model import RobotModel
from skrobot.urdf.modularize_urdf import find_root_link
from skrobot.urdf.modularize_urdf import print_xacro_usage_instructions
from skrobot.urdf.modularize_urdf import transform_urdf_to_macro
from skrobot.utils.package import is_package_installed
from skrobot.utils.urdf import apply_scale
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
    parser.add_argument(
        '--overwrite-mesh',
        action='store_true',
        help='Overwrite existing mesh files during export. '
             + 'If not specified, existing files are preserved.')
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
    parser.add_argument(
        '--target-triangles', type=int, default=None,
        help='Target number of triangles for texture-preserving decimation. '
        'When specified, uses enhanced decimation that preserves texture '
        'colors by converting them to vertex colors, performing decimation, '
        'and converting back to texture format. Default is None.')
    parser.add_argument(
        '--xacro', action='store_true',
        help='Generate xacro macro file with modularized URDF '
        'in addition to converted URDF.')
    parser.add_argument(
        '--xacro-no-prefix', action='store_true',
        help='When generating xacro, do not use prefix parameter '
        '(only works with --xacro option).')
    parser.add_argument(
        '--scale', type=float, default=1.0,
        help='Scale factor for link lengths and mesh geometries. '
        'Default is 1.0 (no scaling).')

    args = parser.parse_args()

    trimesh_version = determine_version('trimesh')
    if Version(trimesh_version) < Version("4.0.10"):
        print(
            '[WARNING] With `trimesh` < 4.0.10, the output dae is not '
            + 'colored. Please `pip install trimesh -U`')
        sys.exit(1)
    if args.decimation_area_ratio_threshold or args.target_triangles:
        disable_decimation = False
        if is_package_installed('open3d') is False:
            print("[ERROR] open3d is not installed. "
                  + "Please install it as 'pip install scikit-robot[all]' "
                  + "to include open3d or 'pip install open3d'")
            disable_decimation = True
        if args.decimation_area_ratio_threshold and is_package_installed('fast-simplification') is False:
            print("[ERROR] fast-simplification is not installed. "
                  + "Please install it with 'pip install fast-simplification'")
            disable_decimation = True
        if disable_decimation:
            sys.exit(1)

    urdf_path = Path(args.urdf)

    if args.output is None:
        fn, _ = osp.splitext(args.urdf)
        index = 0
        pattern = fn + "_%i.urdf"
        outfile = pattern % index
        while osp.exists(outfile):
            index += 1
            outfile = pattern % index
        output_path = Path(outfile)
    else:
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

    with force_visual_mesh_origin_to_zero_or_not():
        print(f"Loading URDF from: {urdf_path}")
        r = RobotModel.from_urdf(urdf_path)

    with export_mesh_format(
            '.' + args.format,
            decimation_area_ratio_threshold=args.decimation_area_ratio_threshold,  # NOQA
            simplify_vertex_clustering_voxel_size=args.voxel_size,
            target_triangles=args.target_triangles,
            overwrite_mesh=args.overwrite_mesh), apply_scale(args.scale):
        print(f"Saving new URDF to: {output_path}")
        # Ensure output directory exists
        output_dir = output_path.parent
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        r.urdf_robot_model.save(str(output_path))

    if args.inplace:
        print(f"Moving {output_path} to {urdf_path} (inplace)")
        shutil.move(output_path, urdf_path)

    if args.xacro:
        root_link = find_root_link(str(output_path) if not args.inplace else str(urdf_path))
        xacro_root, robot_name = transform_urdf_to_macro(
            str(output_path) if not args.inplace else str(urdf_path),
            root_link,
            args.xacro_no_prefix
        )

        xacro_output_path = output_path.with_suffix('.xacro') if not args.inplace else urdf_path.with_suffix('.xacro')
        etree.ElementTree(xacro_root).write(
            str(xacro_output_path),
            pretty_print=True,
            xml_declaration=True,
            encoding="utf-8"
        )
        print_xacro_usage_instructions(str(xacro_output_path), robot_name, args.xacro_no_prefix)


if __name__ == '__main__':
    main()

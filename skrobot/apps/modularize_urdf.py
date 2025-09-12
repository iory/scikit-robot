#!/usr/bin/env python

import argparse
import os

from lxml import etree

from skrobot.urdf.modularize_urdf import find_root_link
from skrobot.urdf.modularize_urdf import print_xacro_usage_instructions
from skrobot.urdf.modularize_urdf import transform_urdf_to_macro


def main():
    parser = argparse.ArgumentParser(description="Modularize URDF to xacro macro")
    parser.add_argument("input_urdf", help="Input URDF file path")
    parser.add_argument("--no-prefix", action="store_true",
                        help="Remove 'prefix' parameter and do not use ${prefix} in names")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()

    root_link = find_root_link(args.input_urdf)
    xacro_root, robot_name = transform_urdf_to_macro(args.input_urdf, root_link, args.no_prefix)

    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input_urdf)[0]
        output_path = base_name + "_modularized.xacro"

    etree.ElementTree(xacro_root).write(output_path, pretty_print=True, xml_declaration=True, encoding="utf-8")

    print_xacro_usage_instructions(output_path, robot_name, args.no_prefix)


if __name__ == "__main__":
    main()

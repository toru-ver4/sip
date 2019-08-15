#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
.hdr ファイルを .exr ファイルに変換
"""

import os
import sys
import numpy as np
import TyImageIO as tyio
import OpenImageIO as oiio


def dump_attr(fname, attr):
    print("attr data of {} is as follows.".format(fname))
    print(attr)


def dump_img_info(img):
    print("shape: ", img.shape)
    print("dtype: ", img.dtype)
    print("min: ", np.min(img))
    print("max: ", np.max(img))


def get_out_fname(in_fname):
    root, ext = os.path.splitext(in_fname)
    return root + ".exr"


def show_usage():
    usage_str = """
usage:
    python convert_from_hdr_to_exr.py source_file

options:
    source_file     A .hdr source file.
"""
    print(usage_str)


def get_input_file_name():
    if len(sys.argv) != 2:
        show_usage()
        sys.exit(1)

    return sys.argv[1]


def main():
    in_fname = get_input_file_name()
    out_fname = get_out_fname(in_fname)
    reader = tyio.TyReader(in_fname)
    img = reader.read()
    dump_attr(in_fname, reader.get_attr())
    dump_img_info(img)
    writer = tyio.TyWriter(img, out_fname)
    writer.write(out_img_type_desc=oiio.FLOAT)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

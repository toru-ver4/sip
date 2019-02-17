#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ひたすら連番のdpxを作る。コピーで。
"""

import os
import shutil


SOURCE_DPX = "../test_pattern/img/SMPTE ST2084_ITU-R BT.2020_D65_3840x2160_rev01_type1.dpx"
# SOURCE_DPX = "../test_pattern/img/Gamma 2.4_ITU-R BT.709_D65_3840x2160_rev00_type1.dpx"


def copy_dpx_file_and_rename_sequential():
    src = SOURCE_DPX
    sec = 5
    fps = 60
    frame = sec * fps
    file_str = "./img/file_{:08d}.dpx"
    for idx in range(frame):
        dst = file_str.format(idx)
        print(dst)
        shutil.copyfile(src, dst)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    copy_dpx_file_and_rename_sequential()

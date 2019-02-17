#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HDR用のテストパターンを作る
"""

import os
import subprocess


def gen_dpx():

    bg_luminance = 0.2  # unit is "nits"
    width = [1920, 3840, 4096]
    height = [1080, 2160, 2160]

    for idx in range(len(width)):
        file_str = "HDR_TEST_PATTEN_{:d}x{:d}_bg_{:.02f}nits.{:s}"
        tiff_name = file_str.format(width[idx], height[idx], bg_luminance,
                                    "tiff")
        tiff_name = os.path.join("..", tiff_name)
        dpx_name = file_str.format(width[idx], height[idx], bg_luminance,
                                   "dpx")

        ext_cmd = ['magick', '-depth', '16', tiff_name,
                   '-define', 'dpx:file.copyright=EIZO Corporation',
                   '-depth', '10', dpx_name,
                   ]
        p = subprocess.Popen(ext_cmd, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True)
        for line in p.stdout:
            print(line.rstrip())

        p.wait()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    gen_dpx()

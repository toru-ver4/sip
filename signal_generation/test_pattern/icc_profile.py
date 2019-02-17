#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ICC Profile絡みの動作検証
"""

import os
from PIL import Image
import subprocess

P3_PQ_PROFILE_INCLUDED_IMAGE = "./data/icc.tif"
P3_PQ_PROFILE_NAME = "./data/P3_D65_PQ.icc"


def save_icc_profile_from_image_file(img_name, icc_name):
    img = Image.open(img_name)
    with open(icc_name, 'wb') as f:
        f.write(img.info['icc_profile'])


def save_p3_pq_icc_profile():
    img_name = P3_PQ_PROFILE_INCLUDED_IMAGE
    icc_name = P3_PQ_PROFILE_NAME
    save_icc_profile_from_image_file(img_name, icc_name)


def add_icc_profile(img_file_name, profile_file_name):
    """
    画像ファイルに ICC Profile を埋めこむ
    """
    cmd = ['convert', img_file_name, '-profile', profile_file_name,
           img_file_name]
    subprocess.run(cmd)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    save_p3_pq_icc_profile()
    add_icc_profile("./img/SMPTE ST2084_DCI-P3_D65_1920x1080_rev00_type1.tiff",
                    P3_PQ_PROFILE_NAME)

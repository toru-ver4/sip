#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
動画を作ったりするよ
"""

import os
import shutil
import cv2


def make_source_tiff():
    fps = 24
    second = 5
    frame_num = fps * second

    for idx in range(frame_num):
        dst_file = "img/test_img_{:04d}.tif".format(idx)
        source_file = "img/source.tif"
        shutil.copyfile(source_file, dst_file)


def verify_10bit_data():
    file_name = "verify/hoge_0001.tif"
    img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    h_st = 448
    h_ed = 448 + 1024
    v_st = 280
    v_ed = 281
    gray_img = img[v_st:v_ed, h_st:h_ed, 0]
    for data in gray_img[0]:
        print("{:04X}".format(data))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    verify_10bit_data()

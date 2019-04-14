#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
画像ファイルの特定領域切り出し
"""

import os
import cv2


def trim_img(in_name, out_name):
    st = 290
    width = 3260
    img = cv2.imread(in_name)
    out_img = img[:, st: st+width, :]
    cv2.imwrite(out_name, out_img)


def main_func():
    src_dir = "./src"
    dst_dir = './dst'

    for idx, name in enumerate(os.listdir(src_dir)):
        in_name = os.path.join(src_dir, name)
        out_name = os.path.join(dst_dir, "{:03d}.png".format(idx))
        print(idx, in_name, out_name)
        trim_img(in_name, out_name)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()

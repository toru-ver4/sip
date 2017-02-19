#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
静止画テストパターンの確認

"""

import os
import sys
import cv2
import numpy as np
from PIL import ImageCms
from PIL import Image
import test_pattern_generator as tpg
import imp
imp.reload(tpg)


fg_array_sample = [tpg.const_gray_array_higher,
                   tpg.const_red_grad_array_higher,
                   tpg.const_green_grad_array_higher,
                   tpg.const_blue_grad_array_higher]

bg_array_sample = [tpg.const_black_array, tpg.const_black_array,
                   tpg.const_black_array, tpg.const_black_array]


def make_rgbk_crosshatch(fg_array=fg_array_sample,
                         bg_array=bg_array_sample,
                         angle=30,
                         debug=False):
    """
    # 概要
    良い感じに RGBK のクロスハッチを書くよ
    """
    h_unit = 2
    v_unit = 2

    v_img_array = []
    for v_idx in range(v_unit):
        h_img_array = []
        for h_idx in range(h_unit):
            idx = (v_idx * h_unit) + h_idx
            img = tpg.make_multi_crosshatch(width=2048, height=1080,
                                            h_block=4, v_block=2,
                                            fragment_width=64,
                                            fragment_height=64,
                                            linewidth=1, linetype=cv2.LINE_8,
                                            bg_color_array=bg_array[idx],
                                            fg_color_array=fg_array[idx],
                                            angle=angle, debug=False)
            h_img_array.append(img)
        v_img_array.append(cv2.hconcat((h_img_array)))
    img = cv2.vconcat((v_img_array))

    if debug:
        tpg.preview_image(img[:, :, ::-1])


if __name__ == '__main__':
    make_rgbk_crosshatch(debug=True)

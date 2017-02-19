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


def make_crosshatch(width=4096, height=2160,
                    h_block=16, v_block=8,
                    fragment_width=64, fragment_height=64,
                    linewidth=1, linetype=cv2.LINE_AA,
                    bg_color_array=tpg.const_gray_array_lower,
                    fg_color_array=tpg.const_white_array,
                    angle=30, debug=False):

    # rgbcmykk pattern
    # -----------------------------------------
    bg_color_line = np.concatenate((tpg.const_black_array,
                                    tpg.const_gray_array_lower))
    bg_color_array = np.array([bg_color_line for x in range(v_block)])
    after_shape = (bg_color_array.shape[0] * bg_color_array.shape[1],
                   bg_color_array.shape[2])
    bg_color_array = bg_color_array.reshape(after_shape)
    fg_color_array = [tpg.const_red_grad_array_higher,
                      tpg.const_red_array,
                      tpg.const_green_grad_array_higher,
                      tpg.const_green_array,
                      tpg.const_blue_grad_array_higher,
                      tpg.const_blue_array,
                      tpg.const_cyan_grad_array_higher,
                      tpg.const_cyan_array,
                      tpg.const_magenta_grad_array_higher,
                      tpg.const_magenta_array,
                      tpg.const_yellow_grad_array_higher,
                      tpg.const_yellow_array,
                      tpg.const_gray_array_higher,
                      tpg.const_white_array,
                      tpg.const_gray_array_higher,
                      tpg.const_white_array]
    fg_color_array = np.array(fg_color_array)
    after_shape = (fg_color_array.shape[0] * fg_color_array.shape[1],
                   fg_color_array.shape[2])
    fg_color_array = fg_color_array.reshape(after_shape)

    img = tpg.make_multi_crosshatch(width=width, height=height,
                                    h_block=h_block, v_block=v_block,
                                    fragment_width=fragment_width,
                                    fragment_height=fragment_height,
                                    linewidth=linewidth,
                                    linetype=linetype,
                                    bg_color_array=bg_color_array,
                                    fg_color_array=fg_color_array,
                                    angle=angle, debug=debug)

    return img


def make_and_save_crosshatch():
    # linewidth 1
    # ----------------------------------------------
    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=64, fragment_height=64,
                          linewidth=1, linetype=cv2.LINE_8,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=0, debug=False)
    fname = "./figure/crosshatch_linewidth-1_antialiasing-off_angle-00.png"
    cv2.imwrite(fname, img[:, :, ::-1])

    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=64, fragment_height=64,
                          linewidth=1, linetype=cv2.LINE_AA,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=30, debug=False)
    fname = "./figure/crosshatch_linewidth-1_antialiasing-on_angle-30.png"
    cv2.imwrite(fname, img[:, :, ::-1])

    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=64, fragment_height=64,
                          linewidth=1, linetype=cv2.LINE_AA,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=45, debug=False)
    fname = "./figure/crosshatch_linewidth-1_antialiasing-on_angle-45.png"
    cv2.imwrite(fname, img[:, :, ::-1])

    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=64, fragment_height=64,
                          linewidth=1, linetype=cv2.LINE_AA,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=60, debug=False)
    fname = "./figure/crosshatch_linewidth-1_antialiasing-on_angle-60.png"
    cv2.imwrite(fname, img[:, :, ::-1])

    # linewidth 2
    # ----------------------------------------------
    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=64, fragment_height=64,
                          linewidth=2, linetype=cv2.LINE_8,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=0, debug=False)
    fname = "./figure/crosshatch_linewidth-2_antialiasing-off_angle-00.png"
    cv2.imwrite(fname, img[:, :, ::-1])

    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=64, fragment_height=64,
                          linewidth=2, linetype=cv2.LINE_AA,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=30, debug=False)
    fname = "./figure/crosshatch_linewidth-2_antialiasing-on_angle-30.png"
    cv2.imwrite(fname, img[:, :, ::-1])

    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=64, fragment_height=64,
                          linewidth=2, linetype=cv2.LINE_AA,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=45, debug=False)
    fname = "./figure/crosshatch_linewidth-2_antialiasing-on_angle-45.png"
    cv2.imwrite(fname, img[:, :, ::-1])

    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=64, fragment_height=64,
                          linewidth=2, linetype=cv2.LINE_AA,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=60, debug=False)
    fname = "./figure/crosshatch_linewidth-2_antialiasing-on_angle-60.png"
    cv2.imwrite(fname, img[:, :, ::-1])

    # linewidth 4
    # ----------------------------------------------
    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=96, fragment_height=96,
                          linewidth=4, linetype=cv2.LINE_8,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=0, debug=False)
    fname = "./figure/crosshatch_linewidth-4_antialiasing-off_angle-00.png"
    cv2.imwrite(fname, img[:, :, ::-1])

    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=96, fragment_height=96,
                          linewidth=4, linetype=cv2.LINE_AA,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=30, debug=False)
    fname = "./figure/crosshatch_linewidth-4_antialiasing-on_angle-30.png"
    cv2.imwrite(fname, img[:, :, ::-1])


    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=96, fragment_height=96,
                          linewidth=4, linetype=cv2.LINE_AA,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=45, debug=False)
    fname = "./figure/crosshatch_linewidth-4_antialiasing-on_angle-45.png"
    cv2.imwrite(fname, img[:, :, ::-1])

    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=96, fragment_height=96,
                          linewidth=4, linetype=cv2.LINE_AA,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=60, debug=False)
    fname = "./figure/crosshatch_linewidth-4_antialiasing-on_angle-60.png"
    cv2.imwrite(fname, img[:, :, ::-1])

    # 実験用。debug=True で目視確認すること
    # ---------------------------------------------
    img = make_crosshatch(width=4096, height=2160,
                          h_block=16, v_block=8,
                          fragment_width=96, fragment_height=96,
                          linewidth=4, linetype=cv2.LINE_8,
                          bg_color_array=tpg.const_gray_array_lower,
                          fg_color_array=tpg.const_white_array,
                          angle=0, debug=False)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_and_save_crosshatch()

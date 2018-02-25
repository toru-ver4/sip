#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
静止画テストパターンの確認

"""

import os
import cv2
from scipy import linalg
import numpy as np
import color_convert as cc
import test_pattern_generator as tpg
import colour
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imp
imp.reload(tpg)
imp.reload(cc)


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


def make_crosshatch_easily():
    pass


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


def _convert_array_for_multi_pattern(array):
    array = np.array(array)
    after_shape = (array.shape[0] * array.shape[1],
                   array.shape[2])
    array = array.reshape(after_shape)

    return array


def make_complex_circle_pattern():
    """
    # 概要
    複数のパラメータの円形パターンを作る
    """
    h_block = 16
    v_block = 8
    bg_array = [tpg.const_black_array_16 for x in range(v_block)]
    bg_array = _convert_array_for_multi_pattern(bg_array)
    fg_array = [tpg.red_grad_array_decrement_16,
                tpg.green_grad_array_decrement_16,
                tpg.blue_grad_array_decrement_16,
                tpg.cyan_grad_decrement_16,
                tpg.magenta_grad_decrement_16,
                tpg.yellow_grad_decrement_16,
                tpg.gray_grad_decrement_16,
                tpg.const_white_array_16]
    fg_array = _convert_array_for_multi_pattern(fg_array)

    size_param = [[1, 64], [4, 64], [16, 64], [64, 64], [64, 128]]

    file_str = "figure/circle_size-{}_frag_size-{}.png"

    for size in size_param:
        img = tpg.make_multi_circle(width=4096, height=2160,
                                    h_block=h_block, v_block=v_block,
                                    circle_size=size[0],
                                    fragment_width=size[1],
                                    fragment_height=size[1],
                                    bg_color_array=bg_array,
                                    fg_color_array=fg_array,
                                    debug=False)
        img = np.uint8(img)
        file_name = file_str.format(size[0], size[1])
        cv2.imwrite(file_name, img[:, :, ::-1])


def make_complex_rectangle_pattern():
    """
    # 概要
    複数のパラメータの円形パターンを作る
    """
    h_block = 16
    v_block = 8
    bg_array = [tpg.const_black_array_16 for x in range(v_block)]
    bg_array = _convert_array_for_multi_pattern(bg_array)
    fg_array = [tpg.red_grad_array_decrement_16,
                tpg.green_grad_array_decrement_16,
                tpg.blue_grad_array_decrement_16,
                tpg.cyan_grad_decrement_16,
                tpg.magenta_grad_decrement_16,
                tpg.yellow_grad_decrement_16,
                tpg.gray_grad_decrement_16,
                tpg.const_white_array_16]
    fg_array = _convert_array_for_multi_pattern(fg_array)

    len_param = [[2, 64], [8, 64], [32, 64], [64, 64], [64, 128]]
    angle_param = [[0, cv2.LINE_8], [30, cv2.LINE_AA],
                   [45, cv2.LINE_AA], [60, cv2.LINE_AA]]

    file_str = "figure/rectangle_len-{}_angle-{}.png"
    for len in len_param:
        for angle in angle_param:
            img = tpg.make_multi_rectangle(width=4096, height=2160,
                                           h_block=h_block, v_block=v_block,
                                           h_side_len=len[0],
                                           v_side_len=len[0],
                                           angle=angle[0],
                                           linetype=angle[1],
                                           fragment_width=len[1],
                                           fragment_height=len[1],
                                           bg_color_array=bg_array,
                                           fg_color_array=fg_array,
                                           debug=False)
            img = np.uint8(img)
            file_name = file_str.format(len[0], angle[0])
            cv2.imwrite(file_name, img[:, :, ::-1])


def get_krgbcmy_array(h_block=16, order='decrement', gain=1.0):
    color_set = [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                 (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    a = [tpg.get_color_array(order=order,
                             color=color,
                             div_num=h_block) for color in color_set]
    a = np.array(a)
    a = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
    a = a * gain

    return a


def get_gray_array(h_block=16, order='decrement', gain=1.0):
    color_set = [(1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1),
                 (1, 1, 1), (1, 1, 1), (1, 1, 1)]
    a = [tpg.get_color_array(order=order,
                             color=color,
                             div_num=h_block) for color in color_set]
    a = np.array(a)
    a = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
    a = a * gain

    return a


def test_complex_crosshatch():
    width = 1920
    height = 1080
    h_block = 16
    v_block = 7
    linewidth = 1
    fragment_width = 64
    fragment_height = 64
    angle = 0
    if angle == 0:
        linetype = cv2.LINE_8
    else:
        linetype = cv2.LINE_AA

    # 背景が暗い場合
    # ------------------------
    # fg_array = get_krgbcmy_array(h_block=h_block, order='static', gain=1.0)
    # bg_array = get_gray_array(h_block=h_block, order='decrement', gain=0.5)

    # 背景が明るい場合
    # ------------------------
    fg_array = get_gray_array(h_block=h_block, order='static', gain=0.0)
    bg_array = get_krgbcmy_array(h_block=h_block, order='decrement', gain=1.0)

    img = tpg.make_multi_crosshatch(width=width, height=height,
                                    h_block=h_block, v_block=v_block,
                                    fragment_width=fragment_width,
                                    fragment_height=fragment_height,
                                    linewidth=linewidth, linetype=linetype,
                                    bg_color_array=bg_array,
                                    fg_color_array=fg_array,
                                    angle=angle, debug=False)
    tpg.preview_image(img, 'rgb')
    print(img.shape)


def make_complex_crosshatch():
    h_block = 16
    v_block = 7

    color_fixed = get_krgbcmy_array(h_block=h_block, order='static', gain=1.0)
    color_dec = get_krgbcmy_array(h_block=h_block, order='decrement',
                                  gain=1.0)
    black_fixed = get_gray_array(h_block=h_block, order='static', gain=0.0)
    white_dec_half = get_gray_array(h_block=h_block, order='decrement',
                                    gain=0.5)
    fg_bg_array = [('fg_fix_bg_dec', color_fixed, white_dec_half),
                   ('fg_dec_bg_fix', color_dec, black_fixed),
                   ('reverse_fg_fix_bg_dec', white_dec_half, color_fixed),
                   ('reverse_fg_dec_bg_fix', black_fixed, color_dec)]
    size_list = [(1920, 1080), (3840, 2160), (4096, 2160)]
    linewidth_list = [1, 2, 4, 8]
    fragment_size = [2, 4, 8, 16, 32, 64]
    angle_list = [0, 30, 45, 60]
    f_str = "./figure/chrosshatch_{}x{}_fsize-{}_lwidth-{}_angle-{}_{}.png"
    for fg_bg in fg_bg_array:
        for size in size_list:
            for fsize in fragment_size:
                for angle in angle_list:
                    if angle == 0:
                        linetype = cv2.LINE_8
                    else:
                        linetype = cv2.LINE_AA
                    for linewidth in linewidth_list[::-1]:
                        if linewidth >= fsize:
                            continue
                        fname = f_str.format(size[0], size[1], fsize,
                                             linewidth, angle, fg_bg[0])
                        img = tpg.make_multi_crosshatch(width=size[0],
                                                        height=size[1],
                                                        h_block=h_block,
                                                        v_block=v_block,
                                                        fragment_width=fsize,
                                                        fragment_height=fsize,
                                                        linewidth=linewidth,
                                                        linetype=linetype,
                                                        fg_color_array=fg_bg[1],
                                                        bg_color_array=fg_bg[2],
                                                        angle=angle,
                                                        debug=False)
                        cv2.imwrite(fname, img[:, :, ::-1])


def test_complex_rectangle():
    width = 1920
    height = 1080
    h_block = 16
    v_block = 7
    linewidth = 1
    fragment_size = 64
    angle = 0
    if angle == 0:
        linetype = cv2.LINE_8
        # linetype = cv2.LINE_AA
    else:
        linetype = cv2.LINE_AA

    # 背景が暗い場合
    # ------------------------
    fg_array = get_krgbcmy_array(h_block=h_block, order='static', gain=1.0)
    bg_array = get_gray_array(h_block=h_block, order='decrement', gain=0.5)

    # 背景が明るい場合
    # ------------------------
    # fg_array = get_gray_array(h_block=h_block, order='static', gain=0.0)
    # bg_array = get_krgbcmy_array(h_block=h_block, order='decrement', gain=1.0)

    img = tpg.make_multi_rectangle(width=width, height=height,
                                   h_block=h_block, v_block=v_block,
                                   h_side_len=linewidth, v_side_len=linewidth,
                                   angle=angle,
                                   linetype=linetype,
                                   fragment_width=fragment_size,
                                   fragment_height=fragment_size,
                                   bg_color_array=bg_array,
                                   fg_color_array=fg_array,
                                   debug=False)

    tpg.preview_image(img, 'rgb')
    cv2.imwrite("hoge.png", img[:, :, ::-1])


def make_complex_rectangle():
    h_block = 16
    v_block = 7

    color_fixed = get_krgbcmy_array(h_block=h_block, order='static', gain=1.0)
    color_dec = get_krgbcmy_array(h_block=h_block, order='decrement',
                                  gain=1.0)
    black_fixed = get_gray_array(h_block=h_block, order='static', gain=0.0)
    white_dec_half = get_gray_array(h_block=h_block, order='decrement',
                                    gain=0.5)
    fg_bg_array = [('fg_fix_bg_dec', color_fixed, white_dec_half),
                   ('fg_dec_bg_fix', color_dec, black_fixed),
                   ('reverse_fg_fix_bg_dec', white_dec_half, color_fixed),
                   ('reverse_fg_dec_bg_fix', black_fixed, color_dec)]
    size_list = [(1920, 1080), (3840, 2160), (4096, 2160)]
    linewidth_list = [1, 2, 4, 8]
    fragment_size = [2, 4, 8, 16, 32, 64]
    angle_list = [0, 30, 45, 60]
    f_str = "./figure/rectangle_{}x{}_fsize-{}_lwidth-{}_angle-{}_{}.png"
    for fg_bg in fg_bg_array:
        for size in size_list:
            for fsize in fragment_size:
                for angle in angle_list:
                    if angle == 0:
                        linetype = cv2.LINE_8
                    else:
                        linetype = cv2.LINE_AA
                    for linewidth in linewidth_list[::-1]:
                        if linewidth >= fsize:
                            continue
                        fname = f_str.format(size[0], size[1], fsize,
                                             linewidth, angle, fg_bg[0])
                        img = tpg.make_multi_crosshatch(width=size[0],
                                                        height=size[1],
                                                        h_block=h_block,
                                                        v_block=v_block,
                                                        angle=angle,
                                                        linetype=linetype,
                                                        fragment_width=fsize,
                                                        fragment_height=fsize,
                                                        bg_color_array=fg_bg[2],
                                                        fg_color_array=fg_bg[1],
                                                        debug=False)
                        cv2.imwrite(fname, img[:, :, ::-1])


def test_complex_circle():
    width = 1920
    height = 1080
    h_block = 16
    v_block = 7
    circle_size = 8
    fragment_size = 64

    # 背景が暗い場合
    # ------------------------
    fg_array = get_krgbcmy_array(h_block=h_block, order='static', gain=1.0)
    bg_array = get_gray_array(h_block=h_block, order='decrement', gain=0.5)

    # 背景が明るい場合
    # ------------------------
    # fg_array = get_gray_array(h_block=h_block, order='static', gain=0.0)
    # bg_array = get_krgbcmy_array(h_block=h_block, order='decrement', gain=1.0)

    img = tpg.make_multi_circle(width=width,
                                height=height,
                                h_block=h_block,
                                v_block=v_block,
                                circle_size=circle_size,
                                fragment_width=fragment_size,
                                fragment_height=fragment_size,
                                bg_color_array=bg_array,
                                fg_color_array=fg_array,
                                debug=False)

    tpg.preview_image(img, 'rgb')
    cv2.imwrite("hoge.png", img[:, :, ::-1])


def make_complex_circle():
    h_block = 16
    v_block = 7

    color_fixed = get_krgbcmy_array(h_block=h_block, order='static', gain=1.0)
    color_dec = get_krgbcmy_array(h_block=h_block, order='decrement',
                                  gain=1.0)
    black_fixed = get_gray_array(h_block=h_block, order='static', gain=0.0)
    white_dec_half = get_gray_array(h_block=h_block, order='decrement',
                                    gain=0.5)
    fg_bg_array = [('fg_fix_bg_dec', color_fixed, white_dec_half),
                   ('fg_dec_bg_fix', color_dec, black_fixed),
                   ('reverse_fg_fix_bg_dec', white_dec_half, color_fixed),
                   ('reverse_fg_dec_bg_fix', black_fixed, color_dec)]
    size_list = [(1920, 1080), (3840, 2160), (4096, 2160)]
    linewidth_list = [1, 2, 4, 8]
    fragment_size = [2, 4, 8, 16, 32, 64]
    f_str = "./figure/rectangle_{}x{}_fsize-{}_lwidth-{}_{}.png"
    for fg_bg in fg_bg_array:
        for size in size_list:
            for fsize in fragment_size:
                for linewidth in linewidth_list[::-1]:
                    if linewidth >= fsize:
                        continue
                    fname = f_str.format(size[0], size[1], fsize,
                                         linewidth, fg_bg[0])
                    img = tpg.make_multi_circle(width=size[0],
                                                height=size[1],
                                                h_block=h_block,
                                                v_block=v_block,
                                                circle_size=linewidth,
                                                fragment_width=fsize,
                                                fragment_height=fsize,
                                                bg_color_array=fg_bg[2],
                                                fg_color_array=fg_bg[1],
                                                debug=False)
                    cv2.imwrite(fname, img[:, :, ::-1])


def plot_color_patch(data, v_num=3, h_num=5):
    figsize_base = 5
    plt.rcParams["font.size"] = 18
    f, axarr = plt.subplots(v_num, h_num, sharex='col', sharey='row',
                            figsize=(figsize_base*h_num, 5*v_num))
    for idx in range(v_num * h_num):
        color = "#{:02X}{:02X}{:02X}".format(data[idx][0],
                                             data[idx][1],
                                             data[idx][2])
        h_idx = idx % h_num
        v_idx = idx // h_num
        axarr[v_idx, h_idx].add_patch(
            patches.Rectangle(
                (0, 0), 1.0, 1.0, facecolor=color
            )
        )
    plt.show()


def make_ebu_test_colour_patch():
    luv_file = "./doc/ebu_test_colour_value.csv"
    luv_data = np.loadtxt(luv_file, delimiter=",",
                          skiprows=1, usecols=(5, 6, 7))

    # convert from Yu'v' to Yuv
    luv_data[:, 2] = luv_data[:, 2] * 2/3
    
    # Yuv to XYZ
    xy = colour.UCS_uv_to_xy(luv_data[:, 1:])
    xyY = np.stack((xy[:, 0], xy[:, 1], (luv_data[:, 0] / 100.0))).T
    # print(xyY)
    large_xyz = colour.xyY_to_XYZ(xyY)
    # print(large_xyz)
    rgb_name = 'ITU-R BT.709'
    illuminant_XYZ = colour.RGB_COLOURSPACES[rgb_name].whitepoint
    illuminant_RGB = colour.RGB_COLOURSPACES[rgb_name].whitepoint
    chromatic_adaptation_transform = 'Bradford'
    xyz_to_rgb_mtx = colour.RGB_COLOURSPACES[rgb_name].XYZ_to_RGB_matrix
    rgb_val = colour.XYZ_to_RGB(large_xyz, illuminant_XYZ, illuminant_RGB,
                                xyz_to_rgb_mtx, chromatic_adaptation_transform)
    print(rgb_val)
    # rgb_val = np.uint16(np.round((rgb_val ** 1/2.35) * 0xFFFF))
    # print(rgb_val)
    # print(rgb_val ** (1/2.2))
    rgb_val = np.uint8(np.round((rgb_val ** (1/2.35)) * 0xFF))
    plot_color_patch(rgb_val, v_num=3, h_num=5)


def make_ebu_color_patch_from_yuv():
    yuv_file = "./doc/ebu_test_colour_value.csv"
    ycbcr = np.loadtxt(yuv_file, delimiter=",",
                       skiprows=1, usecols=(2, 3, 4))
    yuv = cc.ycbcr_to_yuv(ycbcr, bit_depth=10)

    rgb2yuv_mtx = [[0.2126, 0.7152, 0.0722],
                   [-0.2126/1.8556, -0.7152/1.8556, 0.9278/1.8556],
                   [0.7874/1.5748, -0.7152/1.5748, -0.0722/1.5748]]
    yuv2rgb_mtx = linalg.inv(np.array(rgb2yuv_mtx))
    print(yuv2rgb_mtx)
    yuv2rgb_mtx = np.array(yuv2rgb_mtx)

    rgb_dash = cc.color_cvt(yuv.reshape((1, yuv.shape[0], yuv.shape[1])),
                            yuv2rgb_mtx)
    rgb_dash = rgb_dash.reshape((rgb_dash.shape[1], rgb_dash.shape[2]))
    # rgb_dash = np.uint8(np.round(rgb_dash * 0xFF))
    # plot_color_patch(rgb_dash, v_num=3, h_num=5)

    rgb = rgb_dash ** 2.35
    rgb_name = 'ITU-R BT.709'
    illuminant_XYZ = colour.RGB_COLOURSPACES[rgb_name].whitepoint
    illuminant_RGB = colour.RGB_COLOURSPACES[rgb_name].whitepoint
    chromatic_adaptation_transform = 'Bradford'
    rgb_to_xyz_mtx = colour.RGB_COLOURSPACES[rgb_name].RGB_to_XYZ_matrix
    large_xyz = colour.RGB_to_XYZ(rgb, illuminant_XYZ, illuminant_RGB,
                                  rgb_to_xyz_mtx,
                                  chromatic_adaptation_transform)
    large_ucs = colour.XYZ_to_UCS(large_xyz)
    uv = colour.UCS_to_uv(large_ucs)
    uv[:, 1] = uv[:, 1] * 3/2
    print(uv)


def get_ebu_color_rgb_from_XYZ():
    xyz_file = "./doc/ebu_test_colour_value2.csv"
    large_xyz = np.loadtxt(xyz_file, delimiter=",",
                           skiprows=1, usecols=(1, 2, 3)) / 100.0
    rgb_val = large_xyz_to_rgb(large_xyz, 'ITU-R BT.2020')
    print(rgb_val)
    rgb_val = rgb_val ** (1/2.35)
    ycbcr = colour.RGB_to_YCbCr(RGB=rgb_val,
                                K=colour.YCBCR_WEIGHTS['ITU-R BT.2020'],
                                in_bits=10, out_bits=10, out_legal=True,
                                out_int=True)
    print(ycbcr)
    yuv2rgb_mtx = np.array(cc.rgb2yuv_rec2020mtx)
    yuv =\
        cc.color_cvt(rgb_val.reshape((1, rgb_val.shape[0], rgb_val.shape[1])),
                     yuv2rgb_mtx)
    ycbcr = cc.yuv_to_ycbcr(yuv.reshape((yuv.shape[1], yuv.shape[2])),
                            bit_depth=10)
    print(ycbcr)
    # plot_color_patch(rgb_val, v_num=3, h_num=5)
    print(colour.RGB_COLOURSPACES['ITU-R BT.709'].RGB_to_XYZ_matrix)


def large_xyz_to_rgb(large_xyz, gamut_str='ITU-R BT.709'):
    illuminant_XYZ = colour.RGB_COLOURSPACES[gamut_str].whitepoint
    illuminant_RGB = colour.RGB_COLOURSPACES[gamut_str].whitepoint
    chromatic_adaptation_transform = 'Bradford'
    xyz_to_rgb_mtx = colour.RGB_COLOURSPACES[gamut_str].XYZ_to_RGB_matrix
    rgb_val = colour.XYZ_to_RGB(large_xyz, illuminant_XYZ, illuminant_RGB,
                                xyz_to_rgb_mtx, chromatic_adaptation_transform)

    return rgb_val


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # make_and_save_crosshatch()
    # make_complex_circle_pattern()
    # make_complex_rectangle_pattern()
    # test_complex_crosshatch()
    # make_complex_crosshatch()
    # test_complex_rectangle()
    # make_complex_rectangle()
    # test_complex_circle()
    # make_complex_circle()
    # make_ebu_test_colour_patch()
    # make_ebu_color_patch_from_yuv()
    get_ebu_color_rgb_from_XYZ()

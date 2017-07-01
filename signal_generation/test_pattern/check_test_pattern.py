#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
静止画テストパターンの確認

"""

import os
import cv2
import numpy as np
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
    fg_normal = get_krgbcmy_array(h_block=h_block, order='static', gain=1.0)
    fg_reverse = get_gray_array(h_block=h_block, order='static', gain=0.0)
    bg_normal = get_gray_array(h_block=h_block, order='decrement', gain=0.5)
    bg_reverse = get_krgbcmy_array(h_block=h_block, order='decrement',
                                   gain=1.0)
    fg_bg_array = [('normal', fg_normal, bg_normal),
                   ('reverse', fg_reverse, bg_reverse)]
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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # make_and_save_crosshatch()
    # make_complex_circle_pattern()
    # make_complex_rectangle_pattern()
    # test_complex_crosshatch()
    make_complex_crosshatch()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 概要
評価用のテストパターン作成ツール集

# 使い方

"""

import cv2
import numpy as np

const_default_color = np.array([1.0, 1.0, 0.0])
const_white = np.array([1.0, 1.0, 1.0])
const_black = np.array([0.0, 0.0, 0.0])
const_red = np.array([1.0, 0.0, 0.0])
const_green = np.array([0.0, 1.0, 0.0])
const_blue = np.array([0.0, 0.0, 1.0])
const_cyan = np.array([0.0, 1.0, 1.0])
const_majenta = np.array([1.0, 0.0, 1.0])
const_yellow = np.array([1.0, 1.0, 0.0])


def parameter_error_message(param_name):
    print('parameter "{}" is not valid.'.format(param_name))


def gen_window_pattern(width=1920, height=1080,
                       color=const_default_color, size=0.5, debug=False):
    """
    # 概要
    Windowパターンを作成する。

    # 詳細仕様
    以下の図のように引数の size に応じて中央に window パターンを表示する。
    size=1.0 は 全画面が window となる。

    size = 0.6                          size = 1.0
    +----------------------------+      +---------------------------+
    |                            |      |                           |
    |       +------------+       |      |                           |
    |       |   window   |       |  =>  |          window           |
    |       |            |       |  =>  |                           |
    |       +------------+       |      |                           |
    |                            |      |                           |
    +----------------------------+      +---------------------------+
    """

    window_img = np.zeros((height, width, 3))
    win_height = round(height * size)
    win_width = round(width * size)
    pt1 = (round((width - win_width) / 2.), round((height - win_height) / 2.))
    pt2 = (pt1[0] + win_width, pt1[1] + win_height)

    cv2.rectangle(window_img, pt1, pt2, color, -1)

    if debug:
        cv2.imshow('preview', window_img[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return window_img


def gen_gradation_bar(width=1920, height=1080,
                      color=const_default_color, direction='h',
                      offset=0.0, debug=False):
    """
    # 概要
    グラデーションパターンを作る関数

    # 特徴
    offset=1.0 に指定すれば saturation のグラデーションにもなるよ！
    """

    slope = color - offset
    inv_color = 1 - color
    if direction == 'h':
        x = np.arange(width) / (width - 1)
        r, g, b = [(x * s_val) + offset for s_val in slope]
        gradation_line = np.dstack((r, g, b))
        gradation_bar = np.vstack([gradation_line for idx in range(height)])

        if debug:
            cv2.imshow('preview', gradation_bar[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif direction == 'v':
        x = np.arange(height) / (height - 1)
        r, g, b = [(x * s_val) + offset for s_val in slope]
        gradation_line = np.dstack((r, g, b)).reshape((height, 1, 3))
        gradation_bar = np.hstack([gradation_line for idx in range(width)])

        if debug:
            cv2.imshow('preview', gradation_bar[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        parameter_error_message("direction")
        gradation_bar = None

    return gradation_bar


def check_abl_pattern(width=1920, height=1080,
                      color_bar_width=100, color_bar_offset=0.0,
                      color_bar_gain=0.5,
                      window_size=0.5, debug=False):
    """
    # 概要
    例のアレの確認用パターンを作ってみます。
    """
    # rgbcmy の color bar をこしらえる
    # --------------------------------
    color_list_org = [const_red, const_green, const_blue,
                      const_cyan, const_majenta, const_yellow]
    color_list = [x * color_bar_gain for x in color_list_org]
    color_bar_list = [gen_gradation_bar(width=color_bar_width,
                                        height=height,
                                        direction='v',
                                        offset=color_bar_offset,
                                        color=c_val) for c_val in color_list]
    # 目隠し用の黒領域をこしらえる
    # ----------------------------
    black_bar_width = color_bar_width * 2
    black_bar = np.zeros((height, black_bar_width, 3))
    color_bar_list.insert(0, black_bar)

    # 白ベタ Window をこしらえる
    # ----------------------------
    win_width = width - (color_bar_width * len(color_list_org))\
        - black_bar_width
    win_height = height
    window_img = gen_window_pattern(width=win_width, height=win_height,
                                    color=const_white, size=window_size)
    color_bar_list.insert(0, window_img)

    # 各パーツを結合
    # ----------------------------
    out_img = cv2.hconcat(color_bar_list)

    if debug:
        cv2.imshow('preview', out_img[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out_img


if __name__ == '__main__':
    gen_gradation_bar(width=1920, height=1080,
                      color=np.array([1.0, 0.7, 0.3]),
                      direction='v', offset=0.8, debug=False)

    gen_window_pattern(width=1920, height=1080,
                       color=np.array([1.0, 1.0, 1.0]), size=0.9, debug=False)

    img = check_abl_pattern(width=1920, height=1080,
                            color_bar_width=50, color_bar_offset=0.0,
                            color_bar_gain=1.0,
                            window_size=1.0, debug=True)
    img *= 0xFFFF
    img[img < 4096] = 4096
    img[img > 60160] = 60160
    img = np.uint16(np.round(img))
    cv2.imwrite("hoge.tiff", img)
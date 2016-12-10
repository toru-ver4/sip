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


def parameter_error_message(param_name):
    print('parameter "{}" is not valid.'.format(param_name))


def gen_window_pattern(width=1920, height=1080,
                       color=const_default_color, size=0.5, debug=True):
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

    base_img = np.zeros((height, width, 3))
    win_height = round(height * size)
    win_width = round(width * size)
    pt1 = (round((width - win_width) / 2.), round((height - win_height) / 2.))
    pt2 = (pt1[0] + win_width, pt1[1] + win_height)

    print(pt1, pt2)

    cv2.rectangle(base_img, pt1, pt2, color, -1)

    if debug:
        cv2.imshow('preview', base_img[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
    print(inv_color)
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


if __name__ == '__main__':
    gen_gradation_bar(width=1920, height=1080,
                      color=np.array([1.0, 0.7, 0.3]),
                      direction='v', offset=0.8, debug=False)
    gen_window_pattern(width=1920, height=1080,
                       color=np.array([1.0, 1.0, 1.0]), size=0.9, debug=True)

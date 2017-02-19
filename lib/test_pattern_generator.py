#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 概要
評価用のテストパターン作成ツール集

# 使い方

"""

import os
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


def preview_image(img):
    cv2.imshow('preview', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    # color_list_org = [const_red, const_green, const_blue,
    #                   const_cyan, const_majenta, const_yellow]
    # color_list_org = [const_white, const_green, const_blue, const_red]
    color_list_org = [const_white, const_red, const_green, const_blue,
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


def get_bt2100_pq_curve(x, debug=False):
    """
    参考：ITU-R Recommendation BT.2100-0
    """
    m1 = 2610/16384
    m2 = 2523/4096 * 128
    c1 = 3424/4096
    c2 = 2413/4096 * 32
    c3 = 2392/4096 * 32

    x = np.array(x)
    bunsi = (x ** (1/m2)) - c1
    bunsi[bunsi < 0] = 0
    bunbo = c2 - (c3 * (x ** (1/m2)))
    luminance = (bunsi / bunbo) ** (1/m1)

    return luminance * 10000


def gen_youtube_hdr_test_pattern(high_bit_num=5, window_size=0.05):
    width = 3840
    height = 2160

    # calc mask bit
    # -------------------------------
    bit_shift = 16 - high_bit_num
    bit_mask = (0xFFFF >> bit_shift) << bit_shift
    text_num = 2 ** high_bit_num
    coef = 0xFFFF / bit_mask

    img = check_abl_pattern(width=width, height=height,
                            color_bar_width=60, color_bar_offset=0.0,
                            color_bar_gain=1.0,
                            window_size=window_size, debug=False)
    img = img * 0xFFFF
    img = np.uint32(np.round(img))
    img = img & bit_mask
    img = img * coef  # 上の階調が暗くならないための処置
    img[img > 0xFFFF] = 0xFFFF
    img = np.uint16(np.round(img))

    # ここで輝度情報を書き込む
    # ---------------------------------
    x = np.arange(text_num) / (text_num - 1)
    x = (np.uint32(np.round(x * 0xFFFF)) & bit_mask) * coef / 0xFFFF
    luminance = get_bt2100_pq_curve(x)
    text_pos = [(3180, int(x * height / text_num) + 28)
                for x in range(text_num)]
    # font = cv2.FONT_HERSHEY_PLAIN
    font = cv2.FONT_HERSHEY_DUPLEX
    font_color = (0x0000, 0x8000, 0x8000)
    for idx in range(text_num):
        text = "{:>8.2f} nits".format(luminance[idx])
        pos = text_pos[idx]
        cv2.putText(img, text, pos, font, 1, font_color)

    return img


def _croshatch_fragment(width=256, height=128, linewidth=1,
                        bg_color=const_black, fg_color=const_white,
                        debug=False):
    """
    # 概要
    クロスハッチの最小パーツを作る
    線は上側、左側にしか引かない。後に結合することを考えて。
    """

    # make base rectanble
    # ----------------------------------
    fragment = np.ones((height, width, 3))
    for idx in range(3):
        fragment[:, :, idx] *= bg_color[idx]

    # add fg lines
    # ----------------------------------
    cv2.line(fragment, (0, 0), (0, height - 1), fg_color, linewidth)
    cv2.line(fragment, (0, 0), (width - 1, 0), fg_color, linewidth)

    if debug:
        preview_image(fragment[:, :, ::-1])

    return fragment


def make_crosshatch(width=1920, height=1080, linewidth=1,
                    fragment_width=64, fragment_height=64,
                    bg_color=const_black, fg_color=const_white,
                    angle=30):
    # make base rectanble
    # ----------------------------------
    img = np.ones((height, width, 3))
    for idx in range(3):
        img[:, :, idx] *= bg_color[idx]

    # plot horizontal lines
    # -----------------------------
    rad = np.array([angle * np.pi / 180.0])
    end_v_init = width * np.tan(rad)
    first_roop_max = int(np.round(np.cos(rad) * height / fragment_height)) + 1
    second_roop_max = int(end_v_init / (fragment_height / np.cos(rad)))

    for idx in range(first_roop_max):
        st_v = (fragment_height * idx) / np.cos(rad)
        ed_v = end_v_init + st_v
        cv2.line(img, (0, st_v), (width, ed_v),
                 fg_color, linewidth, cv2.LINE_AA)

    for idx in range(second_roop_max):
        st_v = (fragment_height * (idx + 1)) / np.cos(rad) * -1
        ed_v = end_v_init + st_v
        cv2.line(img, (0, st_v), (width, ed_v),
                 fg_color, linewidth, cv2.LINE_AA)

    # plot vertical lines
    # -----------------------------
    end_h_init = height * np.tan(rad) * -1
    offset = fragment_width / np.cos(rad)
    roop_max = int(np.round((width - end_h_init) / offset) + 1)
    for idx in range(roop_max):
        st_h = idx * offset
        ed_h = end_h_init + (idx * offset)
        cv2.line(img, (st_h, 0), (ed_h, height),
                 fg_color, linewidth, cv2.LINE_AA)

    preview_image(img[:, :, ::-1])



if __name__ == '__main__':
    # gen_gradation_bar(width=1920, height=1080,
    #                   color=np.array([1.0, 0.7, 0.3]),
    #                   direction='v', offset=0.8, debug=False)

    # gen_window_pattern(width=1920, height=1080,
    #                    color=np.array([1.0, 1.0, 1.0]), size=0.9, debug=False)

    # check_abl_pattern(width=3840, height=2160,
    #                   color_bar_width=200, color_bar_offset=0.0,
    #                   color_bar_gain=1.0,
    #                   window_size=0.1, debug=False)

    # img = gen_youtube_hdr_test_pattern(high_bit_num=6, window_size=0.05)
    # _crosshatch_fragment()
    make_crosshatch()

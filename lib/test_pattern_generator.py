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
import common
# from PIL import ImageCms
import plot_utility as pu
import matplotlib.pyplot as plt
import color_convert as ccv
from scipy import ndimage
# import fire

increment_8bit_16 = [x for x in range(0, 256, 16)]
decrement_8bit_16 = [x for x in range(256, 0, -16)]
decrement_8bit_16[0] = 255
gray_grad_increment_16 = np.array([(x/255, x/255, x/255)
                                   for x in increment_8bit_16])
gray_grad_decrement_16 = np.array([(x/255, x/255, x/255)
                                   for x in decrement_8bit_16])
const_black_array_16 = np.array([(0.0, 0.0, 0.0)
                                 for x in increment_8bit_16])
const_white_array_16 = np.array([(1.0, 1.0, 1.0)
                                 for x in increment_8bit_16])
red_grad_array_decrement_16 = np.array([(x/255, 0, 0)
                                        for x in decrement_8bit_16])
green_grad_array_decrement_16 = np.array([(0, x/255, 0)
                                          for x in decrement_8bit_16])
blue_grad_array_decrement_16 = np.array([(0, 0, x/255)
                                         for x in decrement_8bit_16])
magenta_grad_decrement_16 = np.array([(x/255, 0, x/255)
                                      for x in decrement_8bit_16])
yellow_grad_decrement_16 = np.array([(x/255, x/255, 0)
                                     for x in decrement_8bit_16])
cyan_grad_decrement_16 = np.array([(0, x/255, x/255)
                                   for x in decrement_8bit_16])

const_default_color = np.array([1.0, 1.0, 0.0])
const_white = np.array([1.0, 1.0, 1.0])
const_black = np.array([0.0, 0.0, 0.0])
const_red = np.array([1.0, 0.0, 0.0])
const_green = np.array([0.0, 1.0, 0.0])
const_blue = np.array([0.0, 0.0, 1.0])
const_cyan = np.array([0.0, 1.0, 1.0])
const_majenta = np.array([1.0, 0.0, 1.0])
const_yellow = np.array([1.0, 1.0, 0.0])
const_black_array = np.array([(0.0, 0.0, 0.0) for x in range(0, 128, 16)])
const_white_array = np.array([(1.0, 1.0, 1.0) for x in range(0, 128, 16)])
const_red_array = np.array([(1.0, 0.0, 0.0) for x in range(0, 128, 16)])
const_green_array = np.array([(0.0, 1.0, 0.0) for x in range(0, 128, 16)])
const_blue_array = np.array([(0.0, 0.0, 1.0) for x in range(0, 128, 16)])
const_cyan_array = np.array([(0.0, 1.0, 1.0) for x in range(0, 128, 16)])
const_magenta_array = np.array([(1.0, 0.0, 1.0) for x in range(0, 128, 16)])
const_yellow_array = np.array([(1.0, 1.0, 0.0) for x in range(0, 128, 16)])

const_gray_array_lower = np.array([(x/255, x/255, x/255)
                                   for x in range(0, 128, 16)])
const_gray_array_higher = np.array([(x/255, x/255, x/255)
                                    for x in range(255, 128, -16)])
const_red_grad_array_higher = np.array([(x/255, 0, 0)
                                        for x in range(255, 128, -16)])
const_green_grad_array_higher = np.array([(0, x/255, 0)
                                          for x in range(255, 128, -16)])
const_blue_grad_array_higher = np.array([(0, 0, x/255)
                                         for x in range(255, 128, -16)])
const_magenta_grad_array_higher = np.array([(x/255, 0, x/255)
                                            for x in range(255, 128, -16)])
const_yellow_grad_array_higher = np.array([(x/255, x/255, 0)
                                           for x in range(255, 128, -16)])
const_cyan_grad_array_higher = np.array([(0, x/255, x/255)
                                         for x in range(255, 128, -16)])

"""src:https://en.wikipedia.org/wiki/ColorChecker#Colors"""
const_color_checker_xyY = [
    [0.400, 0.350, 10.1],
    [0.377, 0.345, 35.8],
    [0.247, 0.251, 19.3],
    [0.337, 0.422, 13.3],
    [0.265, 0.240, 24.3],
    [0.261, 0.343, 43.1],
    [0.506, 0.407, 30.1],
    [0.211, 0.175, 12.0],
    [0.453, 0.306, 19.8],
    [0.285, 0.202, 6.6],
    [0.380, 0.489, 44.3],
    [0.473, 0.438, 43.1],
    [0.187, 0.129, 6.1],
    [0.305, 0.478, 23.4],
    [0.539, 0.313, 12.0],
    [0.448, 0.470, 59.1],
    [0.364, 0.233, 19.8],
    [0.196, 0.252, 19.8],
    [0.310, 0.316, 90.0],
    [0.310, 0.316, 59.1],
    [0.310, 0.316, 36.2],
    [0.310, 0.316, 19.8],
    [0.310, 0.316, 9.0],
    [0.310, 0.316, 3.1],
]


def get_color_array(order='static', color=[1, 1, 1],
                    div_num=8, endpoint=True):
    """
    # 概要
    fg_color, bg_color に使うデータを作るのが大変面倒なので、
    楽ちんに作れる関数を用意する。

    # 入力
    order    : 'static', 'increment', 'decrement' を指定
    color    : [r, g, b] で指定。値域は 0.0 ～ 1.0
    min      : increment, decrement の 最小値。最後に max で正規化する
    max      : increment, decrement の 最大値。最後に max で正規化する
    div_num  : 生成する配列の数
    endpoint : np.linspace() 用の引数。使いたければご自由に。

    # 出力
    色情報が入った numpy 配列(N, 3)。
    """

    base_color = np.array(color)
    if order == 'static':
        color_array = np.array([base_color for x in range(div_num)],
                               dtype=np.float)

    elif order == 'increment' or order == 'decrement':
        val_array = np.linspace(1.0, 0.0, div_num, endpoint=endpoint)
        color_array = np.array([base_color * x for x in val_array],
                               dtype=np.float)
        if order == 'increment':
            color_array = color_array[::-1, :]
        color_array = color_array
    return color_array


def preview_image(img, order=None):
    if order == 'rgb':
        cv2.imshow('preview', img[:, :, ::-1])
    elif order == 'bgr':
        cv2.imshow('preview', img)
    else:
        raise ValueError("order parameter is invalid")
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


def get_bt2100_hlg_curve(x, debug=False):
    """
    参考：ITU-R Recommendation BT.2100-0
    """
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073

    under = (x <= 0.5) * 4 * (x ** 2)
    over = (x > 0.5) * (np.exp((x - c) / a) + b)

    y = (under + over) / 12.0

    if debug:
        ax1 = pu.plot_1_graph()
        ax1.plot(x, y)
        plt.show()

    return y


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


def make_crosshatch(width=1920, height=1080,
                    linewidth=1, linetype=cv2.LINE_AA,
                    fragment_width=64, fragment_height=64,
                    bg_color=const_black, fg_color=const_white,
                    angle=30, debug=False):
    """
    # 概要
    クロスハッチパターンを作る。

    # 注意事項
    アンチエイリアシングが8bitしか効かないので
    本関数では強制的に8bitになる。

    アンチエイリアシングをOFFにすななら、
    ```linetype = cv2.LINE_8``` で関数をコールすること。
    """

    # convert float to uint8
    # ---------------------------------
    bg_color = np.uint8(np.round(bg_color * np.iinfo(np.uint8).max))
    fg_color = np.round(fg_color * np.iinfo(np.uint8).max)

    # make base rectanble
    # ----------------------------------
    img = np.ones((height, width, 3), dtype=np.uint8)
    for idx in range(3):
        img[:, :, idx] *= bg_color[idx]

    # plot horizontal lines
    # -----------------------------
    rad = np.array([angle * np.pi / 180.0])
    end_v_init = width * np.tan(rad)
    first_roop_max = int(np.round(np.cos(rad) * height / fragment_height)) + 1
    second_roop_max = int(end_v_init / (fragment_height / np.cos(rad)))

    # H方の直線を下に向かって書く。
    # --------------------------------
    for idx in range(first_roop_max):
        st_v = (fragment_height * idx) / np.cos(rad)
        ed_v = end_v_init + st_v
        cv2.line(img, (0, st_v), (width, ed_v),
                 fg_color, linewidth, linetype)

    # 回転がある場合、最初のループでは右上の領に
    # 書き損が生じる。これを救うために上に向かってH方のの線を書く
    # --------------------------------
    for idx in range(second_roop_max):
        st_v = (fragment_height * (idx + 1)) / np.cos(rad) * -1
        ed_v = end_v_init + st_v
        cv2.line(img, (0, st_v), (width, ed_v),
                 fg_color, linewidth, linetype)

    # plot vertical lines
    # -----------------------------
    end_h_init = height * np.tan(rad) * -1
    offset = fragment_width / np.cos(rad)
    roop_max = int(np.round((width - end_h_init) / offset) + 1)
    for idx in range(roop_max):
        st_h = idx * offset
        ed_h = end_h_init + (idx * offset)
        cv2.line(img, (st_h, 0), (ed_h, height),
                 fg_color, linewidth, linetype)

    # add information of the video level.
    # ------------------------------------
    _add_text_infomation(img,
                         bg_pos=(15, 20), fg_pos=(15, 40),
                         fg_color=fg_color, bg_color=bg_color)
    if debug:
        preview_image(img, 'rgb')

    return img


def make_multi_crosshatch(width=1920, height=1080,
                          h_block=4, v_block=2,
                          fragment_width=64, fragment_height=64,
                          linewidth=1, linetype=cv2.LINE_AA,
                          bg_color_array=const_gray_array_lower,
                          fg_color_array=const_white_array,
                          angle=30, debug=False):
    """
    # 概要
    欲張って複数パターンのクロスハッチを1枚の画像に入れるようにした。
    複数パターンは bg_color_array, fg_color_array にリストとして記述する。

    # 注意事項
    bg_color_array, fg_color_array の shape は
    h_block * v_block の値と一致させること。
    さもないと冒頭のパラメータチェックで例外飛びます。
    """
    # parameter check
    # -----------------------
    if bg_color_array.shape[0] != (h_block * v_block):
        raise TypeError("bg_color_array.shape is invalid.")
    if fg_color_array.shape[0] != (h_block * v_block):
        raise TypeError("fg_color_array.shape is invalid.")

    # block_width = width // h_block
    # block_height = height // v_block
    block_width_array = common.equal_devision(width, h_block)
    block_height_array = common.equal_devision(height, v_block)

    v_img_list = []
    for v_idx in range(v_block):
        h_img_list = []
        block_height = block_height_array[v_idx] 
        for h_idx in range(h_block):
            block_width = block_width_array[h_idx]
            idx = (v_idx * h_block) + h_idx
            img = make_crosshatch(width=block_width, height=block_height,
                                  linewidth=linewidth, linetype=linetype,
                                  fragment_width=fragment_width,
                                  fragment_height=fragment_height,
                                  bg_color=bg_color_array[idx],
                                  fg_color=fg_color_array[idx],
                                  angle=angle)
            h_img_list.append(img)

        v_img_list.append(cv2.hconcat(h_img_list))
    img = cv2.vconcat((v_img_list))

    if debug:
        preview_image(img, 'rgb')

    return img


def _get_center_address(width, height):
    """
    # 概要
    格子の中心座標(整数値)を求める
    """
    h_pos = width // 2
    v_pos = height // 2

    return h_pos, v_pos


def _add_text_infomation(img,
                         bg_pos=(15, 20), fg_pos=(15, 40),
                         fg_color=const_white, bg_color=const_black):
    """
    # 概要
    fg_color, bg_color 情報を画像に直書きする

    # 注意事項
    色かぶりを防ぐために、フォントカラーは (fg + bg) / 2 を
    さらに反転させて 128 で正規化してる。
    反転が嫌だったらコードを直接編集して。
    """

    # add information of the video level.
    # ------------------------------------
    text_color = np.array([256, 256, 256]) - (fg_color + bg_color) / 2
    text_color = text_color / np.max(text_color) * 80
    font = cv2.FONT_HERSHEY_DUPLEX
    text_format = "bg:({:03d}, {:03d}, {:03d})"
    text = text_format.format(bg_color[0], bg_color[1], bg_color[2])
    cv2.putText(img, text, (15, 20), font, 0.30, text_color, 1, cv2.LINE_AA)
    text_format = "fg:({:03.0f}, {:03.0f}, {:03.0f})"
    text = text_format.format(fg_color[0], fg_color[1], fg_color[2])
    cv2.putText(img, text, (15, 40), font, 0.30, text_color, 1, cv2.LINE_AA)


def make_circle_pattern(width=1920, height=1080,
                        circle_size=1,
                        fragment_width=96, fragment_height=96,
                        bg_color=const_black, fg_color=const_white,
                        debug=False):

    # convert float to uint8
    # ---------------------------------
    bg_color = np.uint8(np.round(bg_color * np.iinfo(np.uint8).max))
    fg_color = np.round(fg_color * np.iinfo(np.uint8).max)
    if circle_size <= 1:
        linetype = cv2.LINE_8
    else:
        linetype = cv2.LINE_AA

    img = np.ones((height, width, 3), dtype=np.uint8)
    for idx in range(3):
        img[:, :, idx] *= bg_color[idx]

    fragment_h_num = (width // fragment_width) + 1
    fragment_v_num = (height // fragment_height) + 1
    st_pos_h, st_pos_v = _get_center_address(fragment_width, fragment_height)
    for v_idx in range(fragment_v_num):
        pos_v = st_pos_v + v_idx * fragment_height
        for h_idx in range(fragment_h_num):
            idx = v_idx * fragment_h_num + h_idx
            pos_h = st_pos_h + h_idx * fragment_width
            cv2.circle(img, (pos_h, pos_v), circle_size,
                       fg_color, -1, linetype)

    # add information of the video level.
    # ------------------------------------
    _add_text_infomation(img,
                         bg_pos=(15, 20), fg_pos=(15, 40),
                         fg_color=fg_color, bg_color=bg_color)

    if debug:
        preview_image(img, 'rgb')

    return img


def _rotate_coordinate(pos, angle=30):
    """
    # 概要
    座ををθ℃だけ回転させるよ！
    # 注意事項
    pos は (x, y) 的なタプルな。
    """
    rad = np.array([angle * np.pi / 180.0])
    x = np.cos(rad) * pos[0] + -np.sin(rad) * pos[1]
    y = np.sin(rad) * pos[0] + np.cos(rad) * pos[1]

    return (x[0], y[0])


def make_rectangle_pattern(width=1920, height=1080,
                           h_side_len=32, v_side_len=32,
                           angle=45,
                           linetype=cv2.LINE_AA,
                           fragment_width=96, fragment_height=96,
                           bg_color=const_black, fg_color=const_white,
                           debug=False):

    # convert float to uint8
    # ---------------------------------
    bg_color = np.uint8(np.round(bg_color * np.iinfo(np.uint8).max))
    fg_color = np.round(fg_color * np.iinfo(np.uint8).max)

    img = np.ones((height, width, 3), dtype=np.uint8)
    for idx in range(3):
        img[:, :, idx] *= bg_color[idx]

    st_offset_h = (fragment_width // 2) - (h_side_len // 2)
    st_offset_v = (fragment_height // 2) - (v_side_len // 2)

    # 回転の前に Rectangle の中心が (0, 0) となるよう座標変換
    # ----------------------------------------------------
    center = (h_side_len / 2.0, v_side_len / 2.0)
    pt1_h = 0 - center[0]
    pt1_v = 0 - center[1]
    pt2_h = pt1_h + h_side_len
    pt2_v = pt1_v
    pt3_h = pt1_h
    pt3_v = pt1_v + v_side_len
    pt4_h = pt1_h + h_side_len
    pt4_v = pt1_v + v_side_len

    # 回転する。そのあと、center を足して元の座標に戻す
    # ついでに fragment の中心に配置されるよう offset 足す
    # -------------------------------------------------
    ptrs = [(pt1_h, pt1_v), (pt2_h, pt2_v), (pt4_h, pt4_v), (pt3_h, pt3_v)]
    ptrs = [_rotate_coordinate(x, angle) for x in ptrs]
    ptrs = [(x[0] + center[0] + st_offset_h, x[1] + center[1] + st_offset_v)
            for x in ptrs]

    # 描画ループ
    # ------------------------------------------------
    fragment_h_num = (width // fragment_width) + 1
    fragment_v_num = (height // fragment_height) + 1
    for v_idx in range(fragment_v_num):
        for h_idx in range(fragment_h_num):
            ptrs_current = [(x[0] + h_idx * fragment_width,
                            x[1] + v_idx * fragment_width)
                            for x in ptrs]
            ptrs_current = np.array(ptrs_current, np.int32)
            cv2.fillConvexPoly(img, ptrs_current, fg_color, linetype)

    # add information of the video level.
    # ------------------------------------
    _add_text_infomation(img,
                         bg_pos=(15, 20), fg_pos=(15, 40),
                         fg_color=fg_color, bg_color=bg_color)

    if debug:
        preview_image(img, 'rgb')

    return img


def make_multi_circle(width=1920, height=1080,
                      h_block=4, v_block=2,
                      circle_size=1,
                      fragment_width=96, fragment_height=96,
                      bg_color_array=const_black_array,
                      fg_color_array=const_white_array,
                      debug=False):
    """
    # 概要
    欲張って複数パターンの円形画像を１枚に収める
    """
    # parameter check
    # -----------------------
    if bg_color_array.shape[0] != (h_block * v_block):
        raise TypeError("bg_color_array.shape is invalid.")
    if fg_color_array.shape[0] != (h_block * v_block):
        raise TypeError("fg_color_array.shape is invalid.")

    block_width = width // h_block
    block_height = height // v_block

    v_img_list = []
    for v_idx in range(v_block):
        h_img_list = []
        for h_idx in range(h_block):
            idx = (v_idx * h_block) + h_idx
            img = make_circle_pattern(width=block_width, height=block_height,
                                      circle_size=circle_size,
                                      fragment_width=fragment_width,
                                      fragment_height=fragment_height,
                                      bg_color=bg_color_array[idx],
                                      fg_color=fg_color_array[idx],
                                      debug=False)
            h_img_list.append(img)

        v_img_list.append(cv2.hconcat(h_img_list))
    img = cv2.vconcat((v_img_list))

    if debug:
        preview_image(img, 'rgb')

    return img


def make_multi_rectangle(width=1920, height=1080,
                         h_block=4, v_block=2,
                         h_side_len=32, v_side_len=32,
                         angle=45,
                         linetype=cv2.LINE_AA,
                         fragment_width=96, fragment_height=96,
                         bg_color_array=const_black_array,
                         fg_color_array=const_white_array,
                         debug=False):
    """
    # 概要
    欲張って複数パターンの四角画像を１枚に収める
    """
    # parameter check
    # -----------------------
    if bg_color_array.shape[0] != (h_block * v_block):
        raise TypeError("bg_color_array.shape is invalid.")
    if fg_color_array.shape[0] != (h_block * v_block):
        raise TypeError("fg_color_array.shape is invalid.")

    block_width = width // h_block
    block_height = height // v_block

    v_img_list = []
    for v_idx in range(v_block):
        h_img_list = []
        for h_idx in range(h_block):
            idx = (v_idx * h_block) + h_idx
            img = make_rectangle_pattern(width=block_width,
                                         height=block_height,
                                         h_side_len=h_side_len,
                                         v_side_len=v_side_len,
                                         angle=angle,
                                         linetype=linetype,
                                         fragment_width=fragment_width,
                                         fragment_height=fragment_height,
                                         bg_color=bg_color_array[idx],
                                         fg_color=fg_color_array[idx],
                                         debug=False)
            h_img_list.append(img)

        v_img_list.append(cv2.hconcat(h_img_list))
    img = cv2.vconcat((v_img_list))

    if debug:
        preview_image(img, 'rgb')

    return img


def change_8bit_to_16bit(data):
    return data * 256


def change_10bit_to_16bit(data):
    return data * 64


def change_12bit_to_16bit(data):
    return data * 16


def gen_csf_pattern(width=640, height=480, bar_num=17,
                    a=(32768, 32768, 0), b=(32768, 0, 32768),
                    dtype=np.uint16, debug=False):
    """
    # 概要
    CSF(Contrast Sensitivity Function) のパターンを作る
    """
    lut = [a, b]
    bar_length_list = common.equal_devision(width, bar_num)
    line_bar_list = []
    for bar_idx, length in enumerate(bar_length_list):
        # LUT値をトグルして1次元のbarの値を作る
        # -----------------------------------
        bar = [np.ones((length), dtype=dtype) * lut[bar_idx % 2][color]
               for color in range(3)]
        bar = np.dstack(bar)
        line_bar_list.append(bar)

    # a と b をトグルして作った bar を結合
    # -----------------------------------
    line = np.hstack(line_bar_list)

    # v方向にも stack して 1次元画像を2次元画像にする
    # --------------------------------------------
    img = line * np.ones((height, 1, 3))

    if debug:
        preview_image(img, 'rgb')

    return img


def gen_step_gradation(width=1024, height=128, step_num=17,
                       bit_depth=10, color=(1.0, 1.0, 1.0),
                       direction='h', debug=False):
    """
    # 概要
    階段状に変化するグラデーションパターンを作る。
    なお、引数の調整により滑らからに変化するグラデーションも作れる

    # 注意事項
    1階調ずつ滑らかに階調が変わるグラデーションを作る場合は
    ```(2 ** bit_depth) == (step_num + 1)```
    となるようにパラメータを指定すること。

    """
    max = 2 ** bit_depth

    # グラデーション方向設定
    # ----------------------
    if direction == 'h':
        pass
    else:
        temp = height
        height = width
        width = temp

    # 階段状に変化するグラデーションかどうか判定
    # -------------------------------------
    if (max + 1 != step_num):
        val_list = np.linspace(0, max, step_num)
        # このままだと最終ブロックが 256 とか 1024 なので 1引く
        # --------------------------------------------------
        val_list[-1] -= 1
    else:
        """
        滑らかに変化させる場合は末尾のデータが 256 や 1024 に
        なるため除外する。
        """
        val_list = np.linspace(0, max, step_num)[0:-1]
        step_num -= 1  # step_num は 引数で余計に +1 されてるので引く

        # 念のため1階調ずつの変化か確認
        # ---------------------------
        diff = val_list[1:] - val_list[0:-1]
        if (diff == 1).all():
            pass
        else:
            raise ValueError("calculated value is invalid.")

    # まずは水平1LINEのグラデーションを作る
    # -----------------------------------
    step_length_list = common.equal_devision(width, step_num)
    step_bar_list = []
    for step_idx, length in enumerate(step_length_list):
        step = [np.ones((length)) * color[c_idx] * val_list[step_idx]
                for c_idx in range(3)]
        if direction == 'h':
            step = np.dstack(step)
            step_bar_list.append(step)
            step_bar = np.hstack(step_bar_list)
        else:
            step = np.dstack(step).reshape((length, 1, 3))
            step_bar_list.append(step)
            step_bar = np.vstack(step_bar_list)

    # ブロードキャストを利用して2次元に拡張する
    # ------------------------------------------
    if direction == 'h':
        img = step_bar * np.ones((height, 1, 3))
    else:
        img = step_bar * np.ones((1, height, 3))

    # np.uint16 にコンバート
    # ------------------------------
    img = np.uint16(np.round(img * (2 ** (16 - bit_depth))))

    if debug:
        preview_image(img, 'rgb')

    return img


def composite_gray_scale(img, width, height):
    rate = height / 2160
    grad_width = int(2048 * rate)
    grad_height = int(256 * rate)
    grad_start_v = int(128 * rate)
    grad_space_v = int(64 * rate)
    mk_space_v = int(32 * rate)
    marker_width = int(30 * rate) + 1
    marker_height = int(20 * rate) + 1
    text_offset_h = int(56 * rate)
    text_offset_v = int(16 * rate)

    text_scale = 0.5 * (rate ** 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_color = (0x8000, 0x8000, 0x0000)

    # 8bit, 10bit のグラデーション表示
    # -------------------------------
    grad_8 = gen_step_gradation(width=grad_width*2, height=grad_height,
                                step_num=257, bit_depth=8,
                                color=(1.0, 1.0, 1.0), direction='h')
    grad_10 = gen_step_gradation(width=grad_width*2, height=grad_height,
                                 step_num=1025, bit_depth=10,
                                 color=(1.0, 1.0, 1.0), direction='h')
    width_st = grad_width // 2
    width_ed = width_st + grad_width
    grad_8 = grad_8[:, width_st:width_ed]
    grad_10 = grad_10[:, width_st:width_ed]
    start_h = width // 2 - grad_width // 2
    start_v = grad_start_v
    img[start_v:start_v+grad_height, start_h:start_h+grad_width] = grad_8
    start_v += grad_height + grad_space_v
    img[start_v:start_v+grad_height, start_h:start_h+grad_width] = grad_10

    # グラデーションの横に端点を示すマーカーを用意
    # ----------------------------------------
    marker = make_marker(marker_width, marker_height, rotate=0)
    vst1 = grad_start_v - mk_space_v
    ved1 = grad_start_v - mk_space_v + marker_height
    hst1 = start_h - (marker_width // 2) - 1
    hed1 = start_h - (marker_width // 2) - 1 + marker_width
    img[vst1:ved1, hst1:hed1] = marker
    vst2 = grad_start_v - mk_space_v + grad_height + grad_space_v
    ved2 = grad_start_v - mk_space_v + marker_height + grad_height\
        + grad_space_v
    hst2 = start_h - (marker_width // 2) - 1
    hed2 = start_h - (marker_width // 2) - 1 + marker_width
    img[vst2:ved2, hst2:hed2] = marker

    vst3 = grad_start_v - mk_space_v
    ved3 = grad_start_v - mk_space_v + marker_height
    hst3 = start_h - (marker_width // 2) - 1 + grad_width
    hed3 = start_h - (marker_width // 2) - 1 + marker_width + grad_width
    img[vst3:ved3, hst3:hed3] = marker
    vst4 = grad_start_v - mk_space_v + grad_height + grad_space_v
    ved4 = grad_start_v - mk_space_v + marker_height + grad_height\
        + grad_space_v
    hst4 = start_h - (marker_width // 2) - 1 + grad_width
    hed4 = start_h - (marker_width // 2) - 1 + marker_width + grad_width
    img[vst4:ved4, hst4:hed4] = marker

    # マーカーの横に説明テキスト付与
    # -------------------------------
    pos = (hst1 + text_offset_h, vst1 + text_offset_v)
    text = "8bit gradation"
    cv2.putText(img, text, pos, font, text_scale, font_color)
    pos = (hst2 + text_offset_h, vst2 + text_offset_v)
    text = "10bit gradation"
    cv2.putText(img, text, pos, font, text_scale, font_color)


def composite_csf_pattern(img, width, height):
    rate = height / 2160
    csf_start_h = width // 2 - int(1024 * rate)
    csf_start_v = int(800 * rate)
    csf_width = int(640 * rate)
    csf_height = int(256 * rate)
    csf_h_space = int(64 * rate)
    csf_h_offset = csf_width + csf_h_space
    bar_num = 24
    text_scale = 0.5 * (rate ** 0.6)
    text_offset_v = int(16 * rate)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_color = (0x8000, 0x8000, 0x0000)

    # csf pattern 作成
    # ----------------------------------
    fg_8bit = [128 * 256 for x in range(3)]
    bg_8bit = [130 * 256 for x in range(3)]
    csf_8bit = gen_csf_pattern(width=csf_width, height=csf_height,
                               bar_num=bar_num, a=fg_8bit, b=bg_8bit,
                               dtype=np.uint16)
    img[csf_start_v:csf_start_v+csf_height,
        csf_start_h:csf_start_h+csf_width] = csf_8bit

    fg_10bit = [512 * 64 for x in range(3)]
    bg_10bit = [514 * 64 for x in range(3)]
    csf_10bit = gen_csf_pattern(width=csf_width, height=csf_height,
                                bar_num=bar_num, a=fg_10bit, b=bg_10bit,
                                dtype=np.uint16)
    h_start = csf_start_h + csf_h_offset * 1
    h_end = csf_start_h + csf_h_offset * 1 + csf_width
    img[csf_start_v:csf_start_v+csf_height,
        h_start:h_end] = csf_10bit

    fg_12bit = [2048 * 16 for x in range(3)]
    bg_12bit = [2050 * 16 for x in range(3)]
    csf_12bit = gen_csf_pattern(width=csf_width, height=csf_height,
                                bar_num=bar_num, a=fg_12bit, b=bg_12bit,
                                dtype=np.uint16)
    h_start = csf_start_h + csf_h_offset * 2
    h_end = csf_start_h + csf_h_offset * 2 + csf_width
    img[csf_start_v:csf_start_v+csf_height,
        h_start:h_end] = csf_12bit

    # 説明用のテキスト
    # -------------------
    pos = (csf_start_h + csf_h_offset * 0, csf_start_v - text_offset_v)
    text = "128 and 129 level"
    cv2.putText(img, text, pos, font, text_scale, font_color)

    pos = (csf_start_h + csf_h_offset * 1, csf_start_v - text_offset_v)
    text = "512 and 513 level"
    cv2.putText(img, text, pos, font, text_scale, font_color)

    pos = (csf_start_h + csf_h_offset * 2, csf_start_v - text_offset_v)
    text = "2048 and 2049 level"
    cv2.putText(img, text, pos, font, text_scale, font_color)


def composite_limited_full_pattern(img, width, height):
    rate = height / 2160
    csf_start_h = width // 2 - int(1024 * rate)
    csf_start_v = int(1150 * rate)
    csf_width = int(640 * rate)
    csf_height = int(256 * rate)
    csf_h_space = int(64 * rate)
    csf_h_offset = csf_width + csf_h_space
    bar_num = 24
    text_scale = 0.5 * (rate ** 0.6)
    text_offset_v = int(16 * rate)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_color = (0x8000, 0x8000, 0x0000)

    # csf pattern 作成
    # ----------------------------------
    fg_8bit = [0 * 256 for x in range(3)]
    bg_8bit = [16 * 256 for x in range(3)]
    csf_8bit = gen_csf_pattern(width=csf_width-2, height=csf_height-2,
                               bar_num=bar_num, a=fg_8bit, b=bg_8bit,
                               dtype=np.uint16)
    img[csf_start_v:csf_start_v+csf_height,
        csf_start_h:csf_start_h+csf_width] = [48*256, 48*256, 48*256]
    img[csf_start_v+1:csf_start_v+csf_height-1,
        csf_start_h+1:csf_start_h+csf_width-1] = csf_8bit

    fg_12bit = [235 * 256 for x in range(3)]
    bg_12bit = [255 * 256 for x in range(3)]
    csf_12bit = gen_csf_pattern(width=csf_width, height=csf_height,
                                bar_num=bar_num, a=fg_12bit, b=bg_12bit,
                                dtype=np.uint16)
    h_start = csf_start_h + csf_h_offset * 2
    h_end = csf_start_h + csf_h_offset * 2 + csf_width
    img[csf_start_v:csf_start_v+csf_height,
        h_start:h_end] = csf_12bit

    # 説明用のテキスト
    # -------------------
    pos = (csf_start_h + csf_h_offset * 0, csf_start_v - text_offset_v)
    text = "0 and 64 level"
    cv2.putText(img, text, pos, font, text_scale, font_color)

    pos = (csf_start_h + csf_h_offset * 2, csf_start_v - text_offset_v)
    text = "940 and 1023 level"
    cv2.putText(img, text, pos, font, text_scale, font_color)


def gen_ST2084_gray_scale(img, width, height):
    rate = height / 2160
    scale_width = int(96 * rate)
    scale_height = height - 2  # "-2" is for pixels of frame.
    scale_step = 65
    bit_depth = 10
    scale_color = (1.0, 1.0, 1.0)
    text_offset_h = int(12 * rate)
    text_offset_v = int(26 * rate)
    text_scale = 0.5 * (rate ** 0.6)
    text_d_offset = int(128 * (rate ** 0.6))

    # グレースケール設置
    # --------------------------
    scale = gen_step_gradation(width=scale_width, height=scale_height,
                               step_num=scale_step, color=scale_color,
                               direction='v', bit_depth=bit_depth)
    img[0+1:height-1, 0+1:scale_width+1] = scale

    # テキスト情報付与
    # --------------------------
    font = cv2.FONT_HERSHEY_DUPLEX
    font_color = (0x8000, 0x8000, 0x0000)

    len_list = common.equal_devision(scale_height, scale_step)
    v_st = 0
    val_list = np.linspace(0, 2**bit_depth, scale_step)
    val_list[-1] -= 1
    luminance = get_bt2100_pq_curve(val_list / ((2**bit_depth)-1))

    for idx, x in enumerate(len_list):
        pos = (scale_width + text_offset_h, text_offset_v + v_st)
        v_st += x
        if luminance[idx] < 999.99999:
            text = "{:>4.0f},{:>6.1f}".format(val_list[idx], luminance[idx])
        else:
            text = "{:>4.0f},{:>5.0f}".format(val_list[idx], luminance[idx])
        cv2.putText(img, text, pos, font, text_scale, font_color)

    # 説明用のマーカー＆テキスト付与
    # -------------------------------
    pos = (scale_width + text_offset_h + text_d_offset, text_offset_v)
    text = "<-- VideoLevel, Brightness for PQ Curve."
    cv2.putText(img, text, pos, font, text_scale, font_color)


def gen_hlg_gray_scale(img, width, height):
    rate = height / 2160
    scale_width = int(96 * rate)
    scale_height = height - 2  # "-2" is for pixels of frame.
    scale_step = 65
    bit_depth = 10
    scale_color = (1.0, 1.0, 1.0)
    text_offset_h = int((118) * (rate ** 0.6))
    text_offset_v = int(26 * rate)
    text_scale = 0.5 * (rate ** 0.6)
    text_d_offset = int(532 * (rate ** 0.7))

    # グレースケール設置
    # --------------------------
    scale = gen_step_gradation(width=scale_width, height=scale_height,
                               step_num=scale_step, color=scale_color,
                               direction='v', bit_depth=bit_depth)
    v_b = 0 + 1
    v_e = height - 1
    h_b = width - scale_width - 1
    h_e = width - scale_width + scale_width - 1
    img[v_b:v_e, h_b:h_e] = scale

    # テキスト情報付与
    # --------------------------
    font = cv2.FONT_HERSHEY_DUPLEX
    font_color = (0x8000, 0x8000, 0x0000)

    len_list = common.equal_devision(scale_height, scale_step)
    v_st = 0
    val_list = np.linspace(0, 2**bit_depth, scale_step)
    val_list[-1] -= 1
    luminance = get_bt2100_hlg_curve(val_list / ((2**bit_depth)-1)) * 1000

    for idx, x in enumerate(len_list):
        pos = (width - scale_width - text_offset_h, text_offset_v + v_st)
        v_st += x
        if luminance[idx] < 999.99999:
            text = "{:>4.0f}, {:>5.1f}".format(val_list[idx], luminance[idx])
        else:
            text = "{:>4.0f}, {:>4.0f}".format(val_list[idx], luminance[idx])
        cv2.putText(img, text, pos, font, text_scale, font_color)

    # 説明用のマーカー＆テキスト付与
    # -------------------------------
    pos = (width - text_d_offset, text_offset_v)
    text = "VideoLevel, Brightness for HLG. -->"
    cv2.putText(img, text, pos, font, text_scale, font_color)


def gen_pq_sdr_color_checker(img, width, height):
    rate = width / 4096
    patch_width = int(112 * rate)
    patch_height = int(112 * rate)
    patch_space = int(24 * rate)
    h_offset = int(292 * rate)
    v_offset = int(128 * rate)
    h_num = 4
    v_num = 6

    text_scale = 0.5 * (rate ** 0.6)
    text_offset_v = int(16 * rate)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_color = (0x8000, 0x8000, 0x0000)

    xyY = np.array(const_color_checker_xyY)
    xyY = xyY.reshape((1, xyY.shape[0], xyY.shape[1]))
    rgb = ccv.xyY_to_RGB(xyY=xyY,
                         gamut=ccv.const_rec2020_xy,
                         white=ccv.const_d65_large_xyz)
    rgb[rgb < 0] = 0

    # scale to 100 nits (rgssb/100).
    rgb = np.uint16(np.round(ccv.linear_to_pq(rgb/100) * 0xFFFF))

    for idx in range(h_num * v_num):
        h_idx = idx // v_num
        v_idx = v_num - (idx % v_num) - 1
        patch = np.ones((patch_height, patch_width, 3), dtype=np.uint16)
        patch[:, :] = rgb[0][idx]
        st_h = h_offset + (patch_width + patch_space) * h_idx
        st_v = v_offset + (patch_height + patch_space) * v_idx
        img[st_v:st_v+patch_height, st_h:st_h+patch_width] = patch

    pos = (h_offset, v_offset - text_offset_v)
    text = "vvv ColorChecker for PQ1000 vvv"
    cv2.putText(img, text, pos, font, text_scale, font_color)


def gen_hlg_sdr_color_checker(img, width, height):
    rate = width / 4096
    patch_width = int(112 * rate)
    patch_height = int(112 * rate)
    patch_space = int(24 * rate)
    h_offset = int((2755 + 512) * rate)
    v_offset = int(128 * rate)
    h_num = 4
    v_num = 6

    text_scale = 0.5 * (rate ** 0.6)
    text_offset_v = int(16 * rate)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_color = (0x8000, 0x8000, 0x0000)

    xyY = np.array(const_color_checker_xyY)
    xyY = xyY.reshape((1, xyY.shape[0], xyY.shape[1]))
    rgb = ccv.xyY_to_RGB(xyY=xyY,
                         gamut=ccv.const_rec2020_xy,
                         white=ccv.const_d65_large_xyz)
    rgb[rgb < 0] = 0
    rgb = np.uint16(np.round(ccv.linear_to_hlg(rgb/10) * 0xFFFF))

    for idx in range(h_num * v_num):
        h_idx = idx // v_num
        v_idx = v_num - (idx % v_num) - 1
        patch = np.ones((patch_height, patch_width, 3), dtype=np.uint16)
        patch[:, :] = rgb[0][idx]
        st_h = h_offset + (patch_width + patch_space) * h_idx
        st_v = v_offset + (patch_height + patch_space) * v_idx
        img[st_v:st_v+patch_height, st_h:st_h+patch_width] = patch

    pos = (int(h_offset), v_offset - text_offset_v)
    text = "vvv ColorChecker for HLG system gamma=1.0 vvv"
    cv2.putText(img, text, pos, font, text_scale, font_color)


def gen_rgbmyc_color_bar(img, width, height):
    # パラメータ設定
    # ----------------------
    rate = height / 2160
    bar_width = int(2048 * rate)
    bar_total_height = int(256 * rate)
    h_st = width // 2 - bar_width // 2
    v_st = height - bar_total_height - 1
    marker_width = int(30 * rate) + 1
    marker_height = int(20 * rate) + 1
    mk_space_v = int(32 * rate)
    scale_step = 65
    color_list = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (1, 0, 1), (1, 1, 0), (0, 1, 1)]

    # color bar 作成
    # ----------------------
    bar_height_list = common.equal_devision(bar_total_height, 6)
    bar_img_list = []
    for color, bar_height in zip(color_list, bar_height_list):
        color_bar = gen_step_gradation(width=bar_width, height=bar_height,
                                       step_num=scale_step, bit_depth=10,
                                       color=color, direction='h')
        bar_img_list.append(color_bar)
    color_bar = np.vstack(bar_img_list)

    img[v_st:v_st+bar_total_height, h_st:h_st+bar_width] = color_bar

    # marker用意
    # ----------------------
    marker = make_marker(marker_width, marker_height, rotate=0)
    vst1 = v_st - mk_space_v
    ved1 = v_st - mk_space_v + marker_height
    hst1 = h_st - (marker_width // 2) - 1
    hed1 = h_st - (marker_width // 2) - 1 + marker_width
    img[vst1:ved1, hst1:hed1] = marker
    vst3 = v_st - mk_space_v
    ved3 = v_st - mk_space_v + marker_height
    hst3 = h_st - (marker_width // 2) - 1 + bar_width
    hed3 = h_st - (marker_width // 2) - 1 + marker_width + bar_width
    img[vst3:ved3, hst3:hed3] = marker


def make_m_and_e_test_pattern(size='uhd'):
    """
    # 概要
    機材の状況が予定通りとなっているか判断できるパターンを作る。
    設計書は $ROOT/signal_generation/test_pattern/doc/
    # 詳細
    引数は 'uhd' or 'dci4k' を指定すること
    """

    # サイズの設定
    # -------------------------
    if size == 'uhd':
        width = 3840
        height = 2160
    elif size == 'dci4k':
        width = 4096
        height = 2160
    elif size == 'fhd':
        width = 1920
        height = 1080
    else:
        print("error. 'size' parameter is invalid.")

    # 黒ベタの背景にグレー枠を付ける
    # ----------------------------
    img = np.ones((height, width, 3), dtype=np.uint16) * 0x8000
    img[1:-1, 1:-1] = [0, 0, 0]

    # 8bit, 10bit のグラデーション表示
    # -------------------------------
    composite_gray_scale(img, width, height)

    # CSFパターンを 8bit/10bit/12git の3種類用意
    # -----------------------------------------
    composite_csf_pattern(img, width, height)

    # CSFパターンを limited/full 確認用に2パターン用意
    # -----------------------------------------
    composite_limited_full_pattern(img, width, height)

    # 左側にST2084確認用のパターンを表示
    # ----------------------------------------
    gen_ST2084_gray_scale(img, width, height)

    # 右側にSTD-B67確認用のパターンを表示
    # ----------------------------------------
    gen_hlg_gray_scale(img, width, height)

    # 左側にSDRレンジのColorChecker(PQ)を表示
    # ----------------------------------------
    gen_pq_sdr_color_checker(img, width, height)

    # 右側にSDRレンジのColorChecker(HLG)を表示
    # ----------------------------------------
    gen_hlg_sdr_color_checker(img, width, height)

    # 画面下にRGBMYCのカラーバーを表示
    # ----------------------------------------
    gen_rgbmyc_color_bar(img, width, height)

    # img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    preview_image(img, 'rgb')
    cv2.imwrite('hoge.tif', img[:, :, ::-1])


def make_dot_mesh(width=1920, height=1080,
                  fg_color=const_white, bg_color=const_black):
    img = np.zeros((height, width, 3))

    # replace fg_color
    # -------------------------
    img[::2, ::2, :] = fg_color
    img[1::2, 1::2, :] = fg_color

    # replace bg_color
    # -------------------------
    img[::2, 1::2, :] = bg_color
    img[1::2, ::2, :] = bg_color

    preview_image(img, 'rgb')
    img = np.uint8(np.round(img * 0xFF))
    cv2.imwrite("dot_mesh.tif", img[:, :, ::-1])


def make_marker(width=101, height=50, rotate=0, preview=False):
    """
    # 概要
    画像を指し示すマーカーを作る
    """
    img = np.zeros((height, width, 3), dtype=np.uint16)
    ptrs = np.array([(0, 0), (width-1, 0), (width//2, height-1)])
    marker_color = (32768, 32768, 32768)
    cv2.fillConvexPoly(img, ptrs, marker_color, cv2.LINE_AA)
    if rotate is not 0:
        img = ndimage.rotate(img, rotate, reshape=True)

    if preview:
        preview_image(img, 'rgb')
        # save

    return img


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # make_m_and_e_test_pattern(size="fhd")
    # _croshatch_fragment(debug=True)
    # make_dot_mesh(fg_color=const_white, bg_color=const_black)
    # make_marker(preview=True)
    # result = get_color_array(order='decrement', color=[0, 1, 0.5],
    #                          min=0, max=255, div_num=8, endpoint=True)
    # print(result)

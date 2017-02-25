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
    font = cv2.FONT_HERSHEY_DUPLEX
    text_format = "bg:({:03d}, {:03d}, {:03d})"
    text = text_format.format(bg_color[0], bg_color[1], bg_color[2])
    cv2.putText(img, text, (15, 20), font, 0.35, fg_color, 1, cv2.LINE_AA)
    text_format = "fg:({:03.0f}, {:03.0f}, {:03.0f})"
    text = text_format.format(fg_color[0], fg_color[1], fg_color[2])
    cv2.putText(img, text, (15, 40), font, 0.35, fg_color, 1, cv2.LINE_AA)

    if debug:
        preview_image(img[:, :, ::-1])

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

    block_width = width // h_block
    block_height = height // v_block

    v_img_list = []
    for v_idx in range(v_block):
        h_img_list = []
        for h_idx in range(h_block):
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
        preview_image(img[:, :, ::-1])

    return img


def _get_center_address(width, height):
    """
    # 概要
    格子の中心座標(整数値)を求める
    """
    h_pos = width // 2
    v_pos = height // 2

    return h_pos, v_pos


def make_circle_pattern(width=1920, height=1080,
                        circle_size=1, linetype=cv2.LINE_AA,
                        fragment_width=64, fragment_height=64,
                        bg_color=const_black, fg_color=const_white,
                        debug=False):

    # convert float to uint8
    # ---------------------------------
    bg_color = np.uint8(np.round(bg_color * np.iinfo(np.uint8).max))
    fg_color = np.round(fg_color * np.iinfo(np.uint8).max)

    img = np.ones((height, width, 3), dtype=np.uint8)
    for idx in range(3):
        img[:, :, idx] *= bg_color[idx]

    fragment_h_num = (width // fragment_width) + 1
    fragment_v_num = (height // fragment_height) + 1
    st_pos_h, st_pos_v = _get_center_address(fragment_width, fragment_height)
    print(st_pos_h, st_pos_v)
    for v_idx in range(fragment_v_num):
        pos_v = st_pos_v + v_idx * fragment_height
        for h_idx in range(fragment_h_num):
            idx = v_idx * fragment_h_num + h_idx
            pos_h = st_pos_h + h_idx * fragment_width
            cv2.circle(img, (pos_h, pos_v), circle_size,
                       fg_color, -1, linetype)

    if debug:
        preview_image(img[:, :, ::-1])


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
                           h_side_len=16, v_side_len=8,
                           angle=45,
                           linetype=cv2.LINE_AA,
                           fragment_width=64, fragment_height=64,
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
    center = (h_side_len / 2.0, v_side_len / 2.0)
    pt1_h = 0 - center[0]
    pt1_v = 0 - center[1]
    pt2_h = pt1_h + h_side_len
    pt2_v = pt1_v
    pt3_h = pt1_h
    pt3_v = pt1_v + v_side_len
    pt4_h = pt1_h + h_side_len
    pt4_v = pt1_v + v_side_len

    pt1_pos = _rotate_coordinate((pt1_h, pt1_v), angle)
    pt2_pos = _rotate_coordinate((pt2_h, pt2_v), angle)
    pt3_pos = _rotate_coordinate((pt3_h, pt3_v), angle)
    pt4_pos = _rotate_coordinate((pt4_h, pt4_v), angle)

    pt1_pos = (pt1_pos[0] + center[0] + st_offset_h,
               pt1_pos[1] + center[1] + st_offset_v)
    pt2_pos = (pt2_pos[0] + center[0] + st_offset_h,
               pt2_pos[1] + center[1] + st_offset_v)
    pt3_pos = (pt3_pos[0] + center[0] + st_offset_h,
               pt3_pos[1] + center[1] + st_offset_v)
    pt4_pos = (pt4_pos[0] + center[0] + st_offset_h,
               pt4_pos[1] + center[1] + st_offset_v)

    fragment_h_num = (width // fragment_width) + 1
    fragment_v_num = (height // fragment_height) + 1

    for v_idx in range(fragment_v_num):
        pt1_pos_v = pt1_pos[1] + v_idx * fragment_width
        pt2_pos_v = pt2_pos[1] + v_idx * fragment_width
        pt3_pos_v = pt3_pos[1] + v_idx * fragment_width
        pt4_pos_v = pt4_pos[1] + v_idx * fragment_width
        for h_idx in range(fragment_h_num):
            idx = v_idx * fragment_h_num + h_idx
            pt1_pos_h = pt1_pos[0] + h_idx * fragment_width
            pt2_pos_h = pt2_pos[0] + h_idx * fragment_width
            pt3_pos_h = pt3_pos[0] + h_idx * fragment_width
            pt4_pos_h = pt4_pos[0] + h_idx * fragment_width
            ptrs = [[pt1_pos_h, pt1_pos_v], [pt2_pos_h, pt2_pos_v],
                    [pt4_pos_h, pt4_pos_v], [pt3_pos_h, pt3_pos_v]]
            ptrs = np.array(ptrs, np.int32)
            print(ptrs)
            cv2.fillConvexPoly(img, ptrs, fg_color)

    if debug:
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
    # img_aa = make_crosshatch(width=1920, height=1080,
    #                          linewidth=1, linetype=cv2.LINE_AA,
    #                          fragment_width=64, fragment_height=64,
    #                          bg_color=const_black, fg_color=const_white,
    #                          angle=30, debug=False)
    # img_na = make_crosshatch(width=1920, height=1080,
    #                          linewidth=1, linetype=cv2.LINE_8,
    #                          fragment_width=64, fragment_height=64,
    #                          bg_color=const_black, fg_color=const_white,
    #                          angle=30)
    # img = cv2.hconcat([img_na, img_aa])
    # preview_image(img[:, :, ::-1])
    # make_multi_crosshatch(debug=True)
    # make_circle_pattern(width=1920, height=1080,
    #                     circle_size=10, linetype=cv2.LINE_AA,
    #                     fragment_width=64, fragment_height=64,
    #                     bg_color=const_black, fg_color=const_white,
    #                     debug=True)
    make_rectangle_pattern(debug=True)

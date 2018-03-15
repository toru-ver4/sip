#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HDR用のテストパターンを作る
"""

import os
import cv2
import numpy as np
import common as cmn
import test_pattern_generator as tpg
import colour
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import color_convert as cc
from scipy import linalg
import imp
imp.reload(tpg)

INTERNAL_PADDING_V = 0.04  # 同一モジュール内でのスペース
INTERNAL_PADDING_H = 0.04  # 同一モジュール内でのスペース

EXTERNAL_PADDING_H = 0.068  # モジュール間でのスペース
EXTERNAL_PADDING_V = 0.068  # モジュール間でのスペース

MARKER_SIZE = 0.011
MARKER_TEXT_SIZE = 0.019
MARKER_TEXT_PADDING_H = 0.01

CSF_PATTERN_WIDTH = 0.12
CSF_PATTERN_HEIGHT = 0.1
CSF_STRIPE_NUM = 6
CSF_H_PADDING = 0.09

CSF_COLOR_PATTERN_WIDTH = 0.12
CSF_COLOR_PATTERN_HEIGHT = 0.1
CSF_COLOR_STRIPE_NUM = 6
CSF_COLOR_H_PADDING = 0.09

LIMITED_PATTERN_WIDTH = 0.1
LIMITED_PATTERN_HEIGHT = 0.1

SIDE_V_GRADATION_WIDTH = 0.026
SIDE_V_GRADATION_TEXT_WIDTH = 0.046
SIDE_V_GRADATION_TEXT_H_OFFFSET = 0.08
SIDE_V_GRADATION_DESC_TEXT_WIDTH = 0.20
SIDE_V_GRADATION_DESC_TEXT_V_OFFSET = 0.005

COLOR_CHECKER_SIZE = 0.05
COLOR_CHECKER_PADDING = 0.005
COLOR_CHECKER_H_START = 0.09
COLOR_CHECKER_V_START = 0.09

EBU_TEST_COLOR_H_START = COLOR_CHECKER_H_START
EBU_TEST_COLOR_V_START = 0.53
EBU_TEST_COLOR_SIZE = 0.0676
EBU_TEST_COLOR_PADDING = 0.006

PQ_CLIP_CHECKER_WIDTH = 0.05
PQ_CLIP_CHECKER_HEIGHT = 0.16
PQ_CLIP_CHECKER_H_OFFSET = 0.07
PQ_CLIP_CHECKER_DESC_TEXT_WIDTH = 0.05

H_GRADATION_HEIGHT = 0.07
H_COLOR_GRADATION_HEIGHT = 0.13

H_FULL_GRADATION_HEIGHT = 0.1

HEAD_V_OFFSET = 0.06

global g_cuurent_pos_v
g_cuurent_pos_v = 0


def _get_dci_primary_on_bt2020():
    """
    DCI色域のPrimaryをBT.2020色域にマッピングし直す。

    Parameters
    ----------
    -

    Returns
    -------
    array_like
        DCI primaries on BT.2020 color space.
    """
    data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    data = data.reshape((1, 3, 3))
    rgb_xyz_mtx = cc.get_rgb_to_xyz_matrix(gamut=cc.const_dci_p3_xy,
                                           white=cc.const_d65_large_xyz)
    xyz_to_rgb_mtx = cc.get_rgb_to_xyz_matrix(gamut=cc.const_rec2020_xy,
                                              white=cc.const_d65_large_xyz)
    xyz_to_rgb_mtx = linalg.inv(xyz_to_rgb_mtx)
    mtx = xyz_to_rgb_mtx.dot(rgb_xyz_mtx)
    data = cc.color_cvt(data, mtx)

    return data.reshape((3, 3))


def _make_csf_image(width=640, height=480, lv1=940, lv2=1023,
                    stripe_num=18):
    """
    長方形を複数個ズラして重ねることでCSFパターンっぽいのを作る。
    入力信号レベルは10bitに限定する。

    Parameters
    ----------
    width : numeric.
        width of the pattern image.
    height : numeric.
        height of the pattern image.
    lv1 : numeric
        video level 1. this value must be 10bit.
    lv2 : numeric
        video level 2. this value must be 10bit.
    stripe_num : numeric
        number of the stripe.

    Returns
    -------
    array_like
        a cms pattern image.
    """
    width_list = cmn.equal_devision(width, stripe_num)
    height_list = cmn.equal_devision(height, stripe_num)
    h_pos_list = cmn.equal_devision(width // 2, stripe_num)
    v_pos_list = cmn.equal_devision(height // 2, stripe_num)
    lv1_16bit = lv1 * (2 ** 6)
    lv2_16bit = lv2 * (2 ** 6)
    img = np.zeros((height, width, 3), dtype=np.uint16)
    
    width_temp = width
    height_temp = height
    h_pos_temp = 0
    v_pos_temp = 0
    for idx in range(stripe_num):
        lv = lv1_16bit if (idx % 2) == 0 else lv2_16bit
        temp_img = np.ones((height_temp, width_temp, 3), dtype=np.uint16)
        temp_img *= lv
        ed_pos_h = h_pos_temp + width_temp
        ed_pos_v = v_pos_temp + height_temp
        img[v_pos_temp:ed_pos_v, h_pos_temp:ed_pos_h] = temp_img
        width_temp -= width_list[stripe_num - 1 - idx]
        height_temp -= height_list[stripe_num - 1 - idx]
        h_pos_temp += h_pos_list[idx]
        v_pos_temp += v_pos_list[idx]

    return img


def _make_csf_color_image(width=640, height=480, lv1=None, lv2=None,
                          stripe_num=18):
    """
    長方形を複数個ズラして重ねることでCSFパターンっぽいのを作る。
    入力信号レベルは10bitに限定する。

    Parameters
    ----------
    width : numeric.
        width of the pattern image.
    height : numeric.
        height of the pattern image.
    lv1 : numeric
        video level 1. this value must be 10bit.
    lv2 : numeric
        video level 2. this value must be 10bit.
    stripe_num : numeric
        number of the stripe.

    Returns
    -------
    array_like
        a cms pattern image.
    """
    width_list = cmn.equal_devision(width, stripe_num)
    height_list = cmn.equal_devision(height, stripe_num)
    h_pos_list = cmn.equal_devision(width // 2, stripe_num)
    v_pos_list = cmn.equal_devision(height // 2, stripe_num)
    lv1_16bit = lv1 * (2 ** 6)
    lv2_16bit = lv2 * (2 ** 6)
    img = np.zeros((height, width, 3), dtype=np.uint16)
    
    width_temp = width
    height_temp = height
    h_pos_temp = 0
    v_pos_temp = 0
    for idx in range(stripe_num):
        lv = lv1_16bit if (idx % 2) == 0 else lv2_16bit
        temp_img = np.ones((height_temp, width_temp, 3), dtype=np.uint16)
        temp_img[:, :, 0] *= lv[0]
        temp_img[:, :, 1] *= lv[1]
        temp_img[:, :, 2] *= lv[2]
        ed_pos_h = h_pos_temp + width_temp
        ed_pos_v = v_pos_temp + height_temp
        img[v_pos_temp:ed_pos_v, h_pos_temp:ed_pos_h] = temp_img
        width_temp -= width_list[stripe_num - 1 - idx]
        height_temp -= height_list[stripe_num - 1 - idx]
        h_pos_temp += h_pos_list[idx]
        v_pos_temp += v_pos_list[idx]

    return img


def composite_bt2020_check_pattern(img):
    """
    BT.2020 でのクリップ具合？を確認するパターン

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    """
    global g_cuurent_pos_v
    img_width = img.shape[1]
    img_height = img.shape[0]

    module_st_h = _get_center_obj_h_start(img)
    module_st_v = g_cuurent_pos_v + int(img_height * EXTERNAL_PADDING_V)

    width = int(img_height * CSF_COLOR_PATTERN_WIDTH)
    height = int(img_height * CSF_COLOR_PATTERN_HEIGHT)
    stripe_num = CSF_COLOR_STRIPE_NUM

    # Primary シマシマのビデオレベル算出
    # ----------------------------------
    rgb_dci = _get_dci_primary_on_bt2020() * 0.01  # 100nits
    rgb_dci[rgb_dci < 0] = 0.0
    rgb_dci[rgb_dci > 1] = 1.0
    rgb_2020 = [[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]]
    rgb_2020 = np.array(rgb_2020) * 0.01  # 100nits

    rgb_2020 = colour.oetf(rgb_2020 * 10000, 'ST 2084')
    rgb_dci = colour.oetf(rgb_dci * 10000, 'ST 2084')
    rgb_2020 = np.uint16(np.round(rgb_2020 * 0x3FF))
    rgb_dci = np.uint16(np.round(rgb_dci * 0x3FF))
    rgb_img = [_make_csf_color_image(width=width, height=height,
                                     lv1=rgb_2020[idx], lv2=rgb_dci[idx],
                                     stripe_num=stripe_num)
               for idx in range(3)]

    # 赤の配置
    # ---------------------------------------
    st_v = module_st_v
    ed_v = st_v + height
    st_h = module_st_h
    ed_h = st_h + width
    img[st_v:ed_v, st_h:ed_h] = rgb_img[0]

    text_pos_h = st_h
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_v - text_height
    text = "BT.2020/DCI-P3"
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.4, 0.4, 0.4))

    # 緑の配置
    # ---------------------------------------
    st_h = (img_width // 2) - (width // 2)
    ed_h = st_h + width
    img[st_v:ed_v, st_h:ed_h] = rgb_img[1]

    text_pos_h = st_h
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_v - text_height
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.4, 0.4, 0.4))

    # 青の配置
    # -------------------------------------------
    st_h = img_width - module_st_h - width
    ed_h = st_h + width
    img[st_v:ed_v, st_h:ed_h] = rgb_img[2]

    text_pos_h = st_h
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_v - text_height
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.6, 0.6, 0.6))

    # 現在のV座標を更新
    g_cuurent_pos_v = ed_v


def composite_limited_csf_pattern(img):
    """
    Limited/Full 確認用のCSFパターン作成＆合成

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    """
    # 基本パラメータ計算
    # ----------------------------------
    global g_cuurent_pos_v
    img_width = img.shape[1]
    img_height = img.shape[0]

    width = int(img_height * CSF_PATTERN_WIDTH)
    height = int(img_height * CSF_PATTERN_HEIGHT)
    stripe_num = CSF_STRIPE_NUM

    module_st_h = _get_center_obj_h_start(img)
    module_st_v = g_cuurent_pos_v + int(img_height * EXTERNAL_PADDING_V)

    limited64 = _make_csf_image(width=width, height=height,
                                lv1=64, lv2=0, stripe_num=stripe_num)
    limited940 = _make_csf_image(width=width, height=height,
                                 lv1=1023, lv2=940, stripe_num=stripe_num)

    bit8_csf = _make_csf_image(width=width, height=height,
                               lv1=520, lv2=512, stripe_num=stripe_num)
    bit10_csf = _make_csf_image(width=width, height=height,
                                lv1=514, lv2=512, stripe_num=stripe_num)

    # 0, 64 のパターン合成
    # -------------------------------------------
    st_v = module_st_v
    ed_v = st_v + height
    st_h = module_st_h
    ed_h = st_h + width
    img[st_v:ed_v, st_h:ed_h] = limited64

    text_pos_h = st_h
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_v - text_height
    text = "▼ 0, 64 Lv"
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.4, 0.4, 0.4))

    # 8bit csf のパターン合成
    # -------------------------------------------
    st_h = ed_h + int(img_width * CSF_H_PADDING)
    ed_h = st_h + width
    img[st_v:ed_v, st_h:ed_h] = bit8_csf

    text_pos_h = st_h
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_v - text_height
    text = "▼ 512, 520 Lv"
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.6, 0.6, 0.6))

    # 940, 1023 のパターン合成
    # -------------------------------------------
    st_h = img_width - module_st_h - width
    ed_h = st_h + width
    img[st_v:ed_v, st_h:ed_h] = limited940

    text_pos_h = st_h
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_v - text_height
    text = "▼ 940, 1023 Lv"
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.6, 0.6, 0.6))

    # 10bit csf のパターン合成
    # -------------------------------------------
    st_h = st_h - int(img_width * CSF_H_PADDING) - width
    ed_h = st_h + width
    img[st_v:ed_v, st_h:ed_h] = bit10_csf

    text_pos_h = st_h
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_v - text_height
    text = "▼ 512, 514 Lv"
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.6, 0.6, 0.6))

    g_cuurent_pos_v = ed_v


def _get_text_height_and_font_size(img_height=1080):
    text_height = img_height * MARKER_TEXT_SIZE
    font_size = int(text_height / 96 * 72)
    text_height = int(text_height)

    return text_height, font_size


def _add_text_info(img, st_pos, font_size=15, font_color=(0.5, 0.5, 0.5),
                   text="I'm a engineer"):
    """
    imgに対して指定座標からテキストを書く

    Parameters
    ----------
    img : ndarray
        image
    pos_st : array of numeric.
        position of the start. order is (H, V).
    font_size : integer
        font size
    font_color : tuple of numeric
        font color. (red, green, blue)
    text : strings
        text for write to the image.

    Returns
    -------
    -

    Examples
    --------
    >>> img = np.zeros((1080, 1920, 3), np.dtype=uint8)
    >>> st_pos = (300, 400)
    >>> _add_text_info(img, st_pos, text="I'm a HERO")
    """
    img_width = img.shape[1]
    img_height = img.shape[0]
    fg_color = tuple([int(x * 0xFF) for x in font_color])

    width = img_width // 3
    height = img_height // 20
    if st_pos[0] + width > img_width:
        print("text box is over(h).")
        width = img_width - st_pos[0]
    if st_pos[1] + height > img_height:
        print("text box is over(v).")
        height = img_height - st_pos[1]

    txt_img = Image.new("RGB", (width, height), (0x00, 0x00, 0x00))
    draw = ImageDraw.Draw(txt_img)
    font = ImageFont.truetype("./fonts/NotoSansMonoCJKjp-Regular.otf",
                              font_size)
    draw.text((0, 0), text, font=font, fill=fg_color)
    text_img = (np.asarray(txt_img) * 0x100).astype(np.uint16)

    # temp_img = img[st_pos[1]:height, st_pos[0]:width]
    # temp_img = text_img
    st_pos_v = st_pos[1]
    ed_pos_v = st_pos[1] + height
    st_pos_h = st_pos[0]
    ed_pos_h = st_pos[0] + width

    # img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = text_img
    text_index = text_img > 0
    temp_img = img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h]
    temp_img[text_index] = text_img[text_index]

    img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = temp_img


def _make_marker(img, vertex_pos, direction="down"):
    """
    始端、終端を示すマーカーを作成する。
    また、適切な設定座標も出力する。

    Parameters
    ----------
    img : ndarray
        image
    vertex_pos : array of numeric.
        position of the vertex. order is (H, V).
    direction : strings
        direction of the vertex.
        you can select "up", "down", "left" or "right".

    Returns
    -------
    -

    Examples
    --------
    >>> img = np.zeros((1080, 1920, 3), np.dtype=uint8)
    >>> vertex_pos = (300, 400)
    >>> _make_marker(img, vertex_pos, direction="down")
    """
    img_width = img.shape[1]
    width = int(img_width * MARKER_SIZE / 2.0) * 2
    height = width // 2

    if direction == "down":
        pt_0 = (vertex_pos[0], vertex_pos[1])
        pt_1 = (vertex_pos[0] - width // 2, vertex_pos[1] - height)
        pt_2 = (vertex_pos[0] + width // 2, vertex_pos[1] - height)
    elif direction == "up":
        pt_0 = (vertex_pos[0], vertex_pos[1])
        pt_1 = (vertex_pos[0] - width // 2, vertex_pos[1] + height)
        pt_2 = (vertex_pos[0] + width // 2, vertex_pos[1] + height)
    elif direction == "left":
        pt_0 = (vertex_pos[0], vertex_pos[1])
        pt_1 = (vertex_pos[0] + height, vertex_pos[1] + width // 2)
        pt_2 = (vertex_pos[0] - height, vertex_pos[1] + width // 2)
    elif direction == "right":
        pt_0 = (vertex_pos[0], vertex_pos[1])
        pt_1 = (vertex_pos[0] + height, vertex_pos[1] - width // 2)
        pt_2 = (vertex_pos[0] - height, vertex_pos[1] - width // 2)
    else:
        print("error. parameter is invalid at _make_marker.")

    ptrs = np.array([pt_0, pt_1, pt_2])
    marker_color = (32768, 32768, 32768)
    cv2.fillConvexPoly(img, ptrs, marker_color, 8)

    return img


def gen_video_level_text_img(width=1024, height=768,
                             font_size=15, font_color=(0.5, 0.5, 0.5),
                             text_info=[["Video Level", "Brightness"]]):
    """
    テキスト情報が付与された画像データを生成する。

    Parameters
    ----------
    width : numeric
        width of the text image.
    height : numeric
        height of the text image.
    font_size : integer
        font size
    font_color : tuple of numeric
        font color. (red, green, blue)
    text_info : array_like
        array of [[video level, brightness], ...].

    Returns
    -------
    ndarray
        text info image.
    """
    fg_color = tuple([int(x * 0xFF) for x in font_color])
    v_offset = height / text_info.shape[0]
    st_pos_h = width * SIDE_V_GRADATION_TEXT_H_OFFFSET
    st_pos_v = 0
    txt_img = Image.new("RGB", (width, height), (0x00, 0x00, 0x00))
    draw = ImageDraw.Draw(txt_img)
    font = ImageFont.truetype("./fonts/NotoSansMonoCJKjp-Regular.otf",
                              font_size)
    for idx, text_pair in enumerate(text_info):
        pos = (st_pos_h, int(st_pos_v + v_offset * idx))
        if text_pair[1] < 999.99999:
            text_data = "{:>4.0f},{:>7.1f}".format(text_pair[0],
                                                   text_pair[1])
        else:
            text_data = "{:>4.0f},{:>6.0f}".format(text_pair[0],
                                                   text_pair[1])
        draw.text(pos, text_data, font=font, fill=fg_color)

    txt_img = (np.asarray(txt_img) * 0x100).astype(np.uint16)

    return txt_img


def composite_pq_vertical_gray_scale(img):
    """
    execute the composition processing for the virtical pq gradation.

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    Returns
    -------
    ndarray
        a image with pq gray scale.

    Notes
    -----
    -

    Examples
    --------
    >>> img = np.zeros((1080, 1920, 3), np.dtype=uint8)
    >>> composite_pq_vertical_gray_scale(img)
    """
    # 基本情報作成
    # ------------------------------------------------------
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    vertual_width = (img_width // 1920) * 1920
    scale_width = int(vertual_width * SIDE_V_GRADATION_WIDTH)
    scale_height = img_height - 2  # "-2" is for pixels of frame.
    text_width = int(vertual_width * SIDE_V_GRADATION_TEXT_WIDTH)
    text_height = img_height - 2  # "-2" is for pixels of frame.
    bit_depth = 10
    video_max = (2 ** bit_depth) - 1
    step_num = 65

    # PQ カーブのグラデーション作成
    # ------------------------------------------------------
    scale = tpg.gen_step_gradation(width=scale_width, height=scale_height,
                                   step_num=step_num, color=(1.0, 1.0, 1.0),
                                   direction='v', bit_depth=bit_depth)
    img[0+1:scale_height+1, 0+1:scale_width+1] = scale

    # ビデオレベルと明るさを表示
    # ------------------------------------------------------
    video_level = [x * (2 ** bit_depth) // (step_num - 1)
                   for x in range(step_num)]
    video_level[-1] -= 1  # 最終データは1多いので引いておく
    video_level_float = np.array(video_level) / video_max
    bright = colour.eotf(video_level_float, 'ITU-R BT.2100 PQ')
    text_info = np.dstack((video_level, bright)).reshape((bright.shape[0], 2))
    font_size = int(text_height / step_num / 96 * 72)
    txt_img = gen_video_level_text_img(width=text_width, height=text_height,
                                       font_size=font_size,
                                       text_info=text_info)
    img[0+1:text_height+1, scale_width:scale_width+text_width, :] = txt_img

    # 説明用テキスト付与
    # ------------------------------------------------------
    text_pos_h = scale_width + text_width
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = int(img_height * SIDE_V_GRADATION_DESC_TEXT_V_OFFSET)
    text = " ◀ PQ's video level(10bit) and luminance(nits)"
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.4, 0.4, 0.4))


def composite_hlg_vertical_gray_scale(img):
    """
    execute the composition processing for the virtical hlg gradation.

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    Returns
    -------
    ndarray
        a image with pq gray scale.

    Notes
    -----
    -

    Examples
    --------
    >>> img = np.zeros((1080, 1920, 3), np.dtype=uint8)
    >>> composite_hlg_vertical_gray_scale(img)
    """
    # 基本情報作成
    # ------------------------------------------------------
    img_width = img.shape[1]
    img_height = img.shape[0]
    vertual_width = (img_width // 1920) * 1920
    scale_width = int(vertual_width * SIDE_V_GRADATION_WIDTH)
    scale_height = img_height - 2  # "-2" is for pixels of frame.
    text_width = int(vertual_width * SIDE_V_GRADATION_TEXT_WIDTH)
    text_height = img_height - 2  # "-2" is for pixels of frame.
    bit_depth = 10
    video_max = (2 ** bit_depth) - 1
    step_num = 65

    # HLGのグラデーション作成
    # ------------------------------------------------------
    scale = tpg.gen_step_gradation(width=scale_width, height=scale_height,
                                   step_num=step_num, color=(1.0, 1.0, 1.0),
                                   direction='v', bit_depth=bit_depth)
    h_st = img_width - 1 - scale_width
    h_ed = -1
    img[0+1:scale_height+1, h_st:h_ed] = scale

    # ビデオレベルと明るさを表示
    # ------------------------------------------------------
    video_level = [x * (2 ** bit_depth) // (step_num - 1)
                   for x in range(step_num)]
    video_level[-1] -= 1  # 最終データは1多いので引いておく
    video_level_float = np.array(video_level) / video_max
    bright = colour.eotf(video_level_float, 'ITU-R BT.2100 HLG',
                         L_W=1000, gamma=1.2)
    text_info = np.dstack((video_level, bright)).reshape((bright.shape[0], 2))
    font_size = int(text_height / step_num / 96 * 72)
    txt_img = gen_video_level_text_img(width=text_width, height=text_height,
                                       font_size=font_size,
                                       text_info=text_info)
    h_st = img_width - (text_width + scale_width)
    h_ed = h_st + text_width
    img[0+1:text_height+1, h_st:h_ed, :] = txt_img

    # 説明用テキスト付与
    # ------------------------------------------------------
    text_pos_h = (h_st - int(img_width * SIDE_V_GRADATION_DESC_TEXT_WIDTH))
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = int(img_height * SIDE_V_GRADATION_DESC_TEXT_V_OFFSET)
    text = "HLG's video level(10bit) and luminance(nits) ▶"
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.4, 0.4, 0.4))


def _get_center_grad_width(img):
    img_width = img.shape[1]

    if img_width <= 2048:
        width = 2048
    else:
        width = 4096

    return width


def _get_center_obj_h_start(img):
    img_width = img.shape[1]
    width = _get_center_grad_width(img)
    st_pos_h = (img_width // 2) - (width // 4)

    return st_pos_h


def composite_8_10bit_middle_gray_scale(img):
    """
    execute the composition processing for the horizontal 8/10bit gradation.

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    Returns
    -------
    ndarray
        a image with 8/10bit horizontal gray scale.

    Notes
    -----
    -

    Examples
    --------
    >>> img = np.zeros((1080, 1920, 3), np.dtype=uint8)
    >>> composite_8_10bit_middle_gray_scale(img)
    """
    global g_cuurent_pos_v
    img_width = img.shape[1]
    img_height = img.shape[0]

    # 解像にに応じた横幅調整。ただし、後でトリミングする
    # ----------------------------------------------
    grad_width = _get_center_grad_width(img)
    grad_height = int(img_height * H_GRADATION_HEIGHT)

    # グラデーション作成。
    # --------------------------------------------------------------------
    grad_8 = tpg.gen_step_gradation(width=grad_width, height=grad_height,
                                    step_num=257, bit_depth=8,
                                    color=(1.0, 1.0, 1.0), direction='h')

    grad_10 = tpg.gen_step_gradation(width=grad_width, height=grad_height,
                                     step_num=1025, bit_depth=10,
                                     color=(1.0, 1.0, 1.0), direction='h')

    # 8bit 合成
    # ------------------------------------------------------------------
    st_pos_h = _get_center_obj_h_start(img)
    # st_pos_h = (img_width // 2) - (grad_width // 4)
    st_pos_v = int(img_height * HEAD_V_OFFSET)
    ed_pos_h = st_pos_h + (grad_width // 2)
    ed_pos_v = st_pos_v + grad_height
    grad_st_h = grad_width // 4
    grad_ed_h = grad_st_h + (grad_width // 2)
    img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = grad_8[:, grad_st_h:grad_ed_h]

    marker_vertex = (st_pos_h, st_pos_v - 1)
    _make_marker(img, marker_vertex, direction='down')
    marker_vertex = (ed_pos_h - 1, st_pos_v - 1)
    _make_marker(img, marker_vertex, direction='down')

    text_pos_h = (st_pos_h + int(img_width * MARKER_TEXT_PADDING_H))
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_pos_v - text_height
    text = "8bit gray scale from 256 to 768 level."
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.4, 0.4, 0.4))

    # 10bit 合成
    # ------------------------------------------------------------------
    pading_v = int(img_height * INTERNAL_PADDING_V)
    st_pos_v = st_pos_v + grad_height + pading_v
    ed_pos_v = st_pos_v + grad_height
    img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = grad_10[:, grad_st_h:grad_ed_h]

    marker_vertex = (st_pos_h, st_pos_v - 1)
    _make_marker(img, marker_vertex, direction='down')
    marker_vertex = (ed_pos_h - 1, st_pos_v - 1)
    _make_marker(img, marker_vertex, direction='down')
    text_pos_h = (st_pos_h + int(img_width * MARKER_TEXT_PADDING_H))
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_pos_v - text_height
    text = "10bit gray scale from 256 to 768 level."
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.4, 0.4, 0.4))

    # 現在のV座標を更新
    g_cuurent_pos_v = ed_pos_v


def composite_10bit_gray_scale(img):
    """
    10bit のグレースケールを 0～1023Lvで表示

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    """
    global g_cuurent_pos_v
    img_width = img.shape[1]
    img_height = img.shape[0]
    module_st_v = g_cuurent_pos_v + int(img_height * EXTERNAL_PADDING_V)

    width = (img_height // 1080) * 1024
    height = int(img_height * H_FULL_GRADATION_HEIGHT)

    grad_10 = tpg.gen_step_gradation(width=width, height=height,
                                     step_num=1025, bit_depth=10,
                                     color=(1.0, 1.0, 1.0), direction='h')

    st_pos_h = _get_center_obj_h_start(img)
    st_pos_v = module_st_v
    ed_pos_h = st_pos_h + width
    ed_pos_v = st_pos_v + height
    img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = grad_10

    marker_vertex = (st_pos_h, st_pos_v - 1)
    _make_marker(img, marker_vertex, direction='down')
    marker_vertex = (ed_pos_h - 1, st_pos_v - 1)
    _make_marker(img, marker_vertex, direction='down')
    text_pos_h = (st_pos_h + int(img_width * MARKER_TEXT_PADDING_H))
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_pos_v - text_height
    text = "10bit gray scale from 0 to 1023 level."
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.4, 0.4, 0.4))

    # 現在のV座標を更新
    g_cuurent_pos_v = ed_pos_v


def composite_rgbmyc_color_bar(img):
    """
    RGBMYCのカラーバーを画面下部に追加

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    """
    img_width = img.shape[1]
    img_height = img.shape[0]

    scale_step = 65
    color_list = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (1, 0, 1), (1, 1, 0), (0, 1, 1)]

    width = _get_center_grad_width(img) // 2
    height = int(img_height * H_COLOR_GRADATION_HEIGHT)

    # color bar 作成
    # ----------------------
    bar_height_list = cmn.equal_devision(height, 6)
    bar_img_list = []
    for color, bar_height in zip(color_list, bar_height_list):
        color_bar = tpg.gen_step_gradation(width=width, height=bar_height,
                                           step_num=scale_step, bit_depth=10,
                                           color=color, direction='h')
        bar_img_list.append(color_bar)
    color_bar = np.vstack(bar_img_list)

    h_st = _get_center_obj_h_start(img)
    h_ed = h_st + width
    v_st = img_height - 1 - height
    v_ed = v_st + height
    img[v_st:v_ed, h_st:h_ed] = color_bar

    # マーカーとテキスト
    # ----------------------------------------
    marker_vertex = (h_st, v_st - 1)
    _make_marker(img, marker_vertex, direction='down')
    marker_vertex = (h_ed - 1, v_st - 1)
    _make_marker(img, marker_vertex, direction='down')
    text_pos_h = (h_st + int(img_width * MARKER_TEXT_PADDING_H))
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = v_st - text_height
    text = "RGBMYC Scale. Video Level ⇒ 0, 16, 32, 48, ..., 992, 1008, 1023"
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.4, 0.4, 0.4))


def composite_pq_color_checker(img):
    """
    SDRレンジのカラーチェッカーをPQ用に作成＆貼り付け

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    """
    # 基本情報
    # --------------------------------------
    img_width = img.shape[1]
    img_height = img.shape[0]
    h_num = 4
    v_num = 6
    patch_width = int(img_height * COLOR_CHECKER_SIZE)
    patch_height = patch_width
    patch_space = int(img_height * COLOR_CHECKER_PADDING)
    patch_st_h = int(img_width * COLOR_CHECKER_H_START)
    patch_st_v = int(img_height * COLOR_CHECKER_V_START)

    xyY = np.array(tpg.const_color_checker_xyY)
    xyY = xyY.reshape((1, xyY.shape[0], xyY.shape[1]))
    rgb = cc.xyY_to_RGB(xyY=xyY,
                        gamut=cc.const_rec2020_xy,
                        white=cc.const_d65_large_xyz)
    rgb[rgb < 0] = 0

    # scale to 100 nits (rgssb/100).
    # rgb = np.uint16(np.round(cc.linear_to_pq(rgb/100) * 0xFFFF))
    rgb = colour.oetf(rgb * 100, function="ST 2084") * 0xFFC0
    rgb = np.uint16(np.round(rgb))

    for idx in range(h_num * v_num):
        h_idx = idx // v_num
        v_idx = v_num - (idx % v_num) - 1
        patch = np.ones((patch_height, patch_width, 3), dtype=np.uint16)
        patch[:, :] = rgb[0][idx]
        st_h = patch_st_h + (patch_width + patch_space) * h_idx
        st_v = patch_st_v + (patch_height + patch_space) * v_idx
        img[st_v:st_v+patch_height, st_h:st_h+patch_width] = patch

    text_pos_h = patch_st_h
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_v - text_height
    text = "▼ ColorChecker for ST2084"
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.3, 0.3, 0.3))


def _get_ebu_color_rgb_from_XYZ():
    xyz_file = "./doc/ebu_test_colour_value2.csv"
    large_xyz = np.loadtxt(xyz_file, delimiter=",",
                           skiprows=1, usecols=(1, 2, 3)) / 100.0
    rgb_val = large_xyz_to_rgb(large_xyz, 'ITU-R BT.2020')
    rgb_val = rgb_val.reshape(1, 15, 3)

    return rgb_val


def large_xyz_to_rgb(large_xyz, gamut_str='ITU-R BT.709'):
    illuminant_XYZ = colour.RGB_COLOURSPACES[gamut_str].whitepoint
    illuminant_RGB = colour.RGB_COLOURSPACES[gamut_str].whitepoint
    chromatic_adaptation_transform = 'Bradford'
    xyz_to_rgb_mtx = colour.RGB_COLOURSPACES[gamut_str].XYZ_to_RGB_matrix
    rgb_val = colour.XYZ_to_RGB(large_xyz, illuminant_XYZ, illuminant_RGB,
                                xyz_to_rgb_mtx, chromatic_adaptation_transform)

    return rgb_val


def composite_pq_ebu_test_colour(img):
    """
    SDRレンジのEBU TEST COLOR をPQ用に作成＆貼り付け

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    """
    # 基本情報
    # --------------------------------------
    img_width = img.shape[1]
    img_height = img.shape[0]
    h_num = 3
    v_num = 5
    patch_width = int(img_height * EBU_TEST_COLOR_SIZE)
    patch_height = patch_width
    patch_space = int(img_height * EBU_TEST_COLOR_PADDING)
    patch_st_h = int(img_width * EBU_TEST_COLOR_H_START)
    patch_st_v = int(img_height * EBU_TEST_COLOR_V_START)

    rgb = _get_ebu_color_rgb_from_XYZ()
    rgb = colour.oetf(rgb * 100, function="ST 2084") * 0xFFC0
    rgb = np.uint16(np.round(rgb))

    for idx in range(h_num * v_num):
        h_idx = idx // v_num
        v_idx = v_num - (idx % v_num) - 1
        patch = np.ones((patch_height, patch_width, 3), dtype=np.uint16)
        patch[:, :] = rgb[0][idx]
        st_h = patch_st_h + (patch_width + patch_space) * h_idx
        st_v = patch_st_v + (patch_height + patch_space) * v_idx
        img[st_v:st_v+patch_height, st_h:st_h+patch_width] = patch

    text_pos_h = patch_st_h
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_v - text_height
    text = "▼ EBU TEST COLOUR for PQ"
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.3, 0.3, 0.3))


def composite_hlg_color_checker(img):
    """
    SDRレンジのカラーチェッカーをHLG用に作成＆貼り付け

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    """
    # 基本情報
    # --------------------------------------
    img_width = img.shape[1]
    img_height = img.shape[0]
    h_num = 4
    v_num = 6
    patch_width = int(img_height * COLOR_CHECKER_SIZE)
    patch_height = patch_width
    patch_space = int(img_height * COLOR_CHECKER_PADDING)
    patch_st_h = int(img_width * COLOR_CHECKER_H_START)
    patch_st_h = img_width - patch_st_h - (patch_width + patch_space) * h_num
    patch_st_v = int(img_height * COLOR_CHECKER_V_START)

    xyY = np.array(tpg.const_color_checker_xyY)
    xyY = xyY.reshape((1, xyY.shape[0], xyY.shape[1]))
    rgb = cc.xyY_to_RGB(xyY=xyY,
                        gamut=cc.const_rec2020_xy,
                        white=cc.const_d65_large_xyz)
    rgb[rgb < 0] = 0

    rgb = colour.eotf_reverse(rgb*100, 'ITU-R BT.2100 HLG', gamma=1.2)
    rgb = np.uint16(np.round(rgb * 0xFFC0))

    for idx in range(h_num * v_num):
        h_idx = idx // v_num
        v_idx = v_num - (idx % v_num) - 1
        patch = np.ones((patch_height, patch_width, 3), dtype=np.uint16)
        patch[:, :] = rgb[0][idx]
        st_h = patch_st_h + (patch_width + patch_space) * h_idx
        st_v = patch_st_v + (patch_height + patch_space) * v_idx
        img[st_v:st_v+patch_height, st_h:st_h+patch_width] = patch

    text_pos_h = patch_st_h
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_v - text_height
    text = "▼ ColorChecker for HLG SG=1.2"
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.3, 0.3, 0.3))


def composite_hlg_ebu_test_colour(img):
    """
    SDRレンジのカラーチェッカーをHLG用に作成＆貼り付け

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    """
    # 基本情報
    # --------------------------------------
    img_width = img.shape[1]
    img_height = img.shape[0]
    h_num = 3
    v_num = 5
    patch_width = int(img_height * EBU_TEST_COLOR_SIZE)
    patch_height = patch_width
    patch_space = int(img_height * EBU_TEST_COLOR_PADDING)
    patch_st_h = int(img_width * EBU_TEST_COLOR_H_START)
    patch_st_h = img_width - patch_st_h - (patch_width + patch_space) * h_num
    patch_st_v = int(img_height * EBU_TEST_COLOR_V_START)

    rgb = _get_ebu_color_rgb_from_XYZ()
    rgb = colour.eotf_reverse(rgb*100, 'ITU-R BT.2100 HLG', gamma=1.2)
    rgb = np.uint16(np.round(rgb * 0xFFC0))

    for idx in range(h_num * v_num):
        h_idx = idx // v_num
        v_idx = v_num - (idx % v_num) - 1
        patch = np.ones((patch_height, patch_width, 3), dtype=np.uint16)
        patch[:, :] = rgb[0][idx]
        st_h = patch_st_h + (patch_width + patch_space) * h_idx
        st_v = patch_st_v + (patch_height + patch_space) * v_idx
        img[st_v:st_v+patch_height, st_h:st_h+patch_width] = patch

    text_pos_h = patch_st_h
    text_height, font_size = _get_text_height_and_font_size(img_height)
    text_pos_v = st_v - text_height
    text = "▼ EBU TEST COLOUR for HLG SG=1.2"
    _add_text_info(img, st_pos=(text_pos_h, text_pos_v), font_size=font_size,
                   text=text, font_color=(0.3, 0.3, 0.3))


def _get_pq_video_levels_for_clip_check(bright=1000, level_num=4,
                                        level_step=4):
    """
    bright で指定した階調を中心とした計 level_num 個 のビデオレベルを
    計算で求める。

    Parameters
    ----------
    bright : numeric.
        target brightness.1
    level_num : numeric.
        a number to calculate.
    level_step : a step of the video level.

    Returns
    -------
    array_like
        video levels

    Examples
    --------
    >>> _get_pq_video_levels_for_clip_check(1000, 4, 4)
    [769-16, 769-12, 769-8, 769-4, 769, 769+4, 769+8, 769+12, 769+16]
    """
    center = colour.oetf(bright, function="ST 2084") * 1023
    center = np.uint16(np.round(center))
    num = level_num * 2 + 1
    ret_val = [center + level_step * (x - level_num)
               for x in range(num)]

    ret_val = _reshape_for_2d(ret_val)
    return ret_val


def _make_block(width=640, height=480,
                level=[[900, 0, 0], [0, 800, 0]]):
    """
    level で指定した色で塗りつぶした長方形を
    V方向に積んだ画像を生成する。

    Parameters
    ----------
    width : numeric.
        total width
    height : numeric.
        total height
    level : array of the numeric or array_like
        values of red, green, blue.
        the bit depth must be 10bit.

    Returns
    -------
    array_like
        block image.

    """
    block_num = len(level)
    height_p = cmn.equal_devision(height, block_num)

    img_list = []
    for idx in range(len(level)):
        img = np.ones((height_p[idx], width, 3), dtype=np.uint16)
        img[:, :, 0] = level[idx][0]
        img[:, :, 1] = level[idx][1]
        img[:, :, 2] = level[idx][2]
        img_list.append(img)
    img = cv2.vconcat(img_list) * (2 ** 6)

    return img


def _reshape_for_2d(data):
    """
    `data` で記述したビデオレベルをR,G,Bに展開する。
    また、shape を (N, 3) に整形する。
    """
    data = np.array(data)
    data = np.dstack((data, data, data))
    data = data.reshape((data.shape[1], data.shape[2]))

    return data


def composite_pq_clip_checker(img):
    """
    PQカーブのクリップ位置を確認するためのテストパターンを作る。
    300nits, 500nits, 1000nits, 4000nits の 4パターン作成

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    """
    global g_cuurent_pos_v
    img_width = img.shape[1]
    img_height = img.shape[0]
    vertual_width = (img_width // 1920) * 1920
    center_bright_list = [300, 500, 1000, 4000]
    level_num = 4
    level_step = 8
    width = int(img_width * PQ_CLIP_CHECKER_WIDTH)
    height = int(img_height * PQ_CLIP_CHECKER_HEIGHT)
    text_width = int(vertual_width * PQ_CLIP_CHECKER_DESC_TEXT_WIDTH)

    module_st_h = _get_center_obj_h_start(img)
    module_st_v = g_cuurent_pos_v + int(img_height * EXTERNAL_PADDING_V)

    h_offset = img_width - (2 * module_st_h) - width - text_width
    h_offset = cmn.equal_devision(h_offset, len(center_bright_list) - 1)

    font_size = int(height / (level_num * 2 + 1) / 96 * 72)

    img_list = []
    text_img_list = []
    for bright in center_bright_list:
        level_temp\
            = _get_pq_video_levels_for_clip_check(bright=bright,
                                                  level_num=level_num,
                                                  level_step=level_step)
        img_temp = _make_block(width, height, level_temp)
        bright_temp = colour.eotf(level_temp / 1024, 'ITU-R BT.2100 PQ')
        level_temp = level_temp[:, 0]
        bright_temp = bright_temp[:, 0]
        text_info = np.dstack((level_temp, bright_temp))
        text_info = text_info.reshape((text_info.shape[1], 2))
        img_list.append(img_temp)
        text_temp = gen_video_level_text_img(width=text_width,
                                             height=height,
                                             font_size=font_size,
                                             text_info=text_info)
        text_img_list.append(text_temp)

    # 配置
    # -------------------------------------
    for idx in range(len(img_list)):
        sum_h = 0
        for o_idx in range(idx):
            sum_h += h_offset[o_idx]
        st_h = module_st_h + sum_h
        st_v = module_st_v
        ed_h = st_h + width
        ed_v = st_v + height
        img[st_v:ed_v, st_h:ed_h] = img_list[idx]

        text_st_h = st_h + width
        text_ed_h = text_st_h + text_width
        text_st_v = st_v
        text_ed_v = ed_v
        img[text_st_v:text_ed_v, text_st_h:text_ed_h] = text_img_list[idx]

        text_pos_h = st_h
        text_height, font_size = _get_text_height_and_font_size(img_height)
        text_pos_v = st_v - text_height
        text = "▼ {}nits clip check".format(center_bright_list[idx])
        _add_text_info(img, st_pos=(text_pos_h, text_pos_v),
                       font_size=font_size,
                       text=text, font_color=(0.3, 0.3, 0.3))

    # 現在のV座標を更新
    # -------------------------------------------
    g_cuurent_pos_v = ed_v


def m_and_e_tp_rev5(width=1920, height=1080):

    # ベースの背景画像を作成
    # img = np.zeros((height, width, 3), dtype=np.uint16)
    img = np.ones((height, width, 3), dtype=np.uint16) * 0x2000

    # 外枠のフレーム作成
    tpg.draw_rectangle(img, (0, 0), (width-1, height-1), (0.5, 0.5, 0.5))

    # 左端にPQグレースケール
    composite_pq_vertical_gray_scale(img)

    # 右端にHLGグレースケール
    composite_hlg_vertical_gray_scale(img)

    # 真ん中上部に8btグレースケール(64～192)
    composite_8_10bit_middle_gray_scale(img)

    # Limited - Full 確認用パターン
    composite_limited_csf_pattern(img)

    # BT.2020 クリップ確認用パターン
    # composite_bt2020_check_pattern(img)

    # PQ クリップ位置チェッカー！
    composite_pq_clip_checker(img)

    # 10bit Gray Scale
    composite_10bit_gray_scale(img)

    # RGBCMY カラーバー
    composite_rgbmyc_color_bar(img)

    # 画面左側にPQ用カラーチェッカー
    composite_pq_color_checker(img)

    # 画面右側にHLG用カラーチェッカー
    composite_hlg_color_checker(img)

    # 画面左型にPQ用 EBU TEST Colour
    composite_pq_ebu_test_colour(img)

    # 画面右側にHLG用 EBU TEST Colour
    composite_hlg_ebu_test_colour(img)

    # preview
    tpg.preview_image(img, 'rgb')

    # write to the file
    cv2.imwrite("test.tiff", img[:, :, ::-1])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    m_and_e_tp_rev5(1920, 1080)
    # m_and_e_tp_rev5(4096, 2160)

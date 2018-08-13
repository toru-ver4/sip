#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ITU-R BT.xxxx に書いてある放送業界向けのテストパターンを作る
"""

import os
import cv2
import numpy as np
import common as cmn
import test_pattern_generator2 as tpg
import colour
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import color_convert as cc
from scipy import linalg
import imp
imp.reload(tpg)


# Define
# -------------------------
PLUGE_HIGHER_LEVEL = {'sdr': 940, 'hdr': 399}
PLUGE_BLACK_LEVEL = 64
PLUGE_SLIGHTLY_LIGHTER_LEVEL = 80
PLUGE_SLIGHTLY_DARKER_LEVEL = 48
PLUGE_HORIZONTAL_STRIPE_TOAL_NUM = 39
PLUGE_HORIZONTAL_STRIPE_EACH_NUM = 10

Sa = {'1920x1080': 0, '3840x2160': 0, '7680x4320': 0}
Sb = {'1920x1080': 312, '3840x2160': 624, '7680x4320': 1248}
Sc = {'1920x1080': 599, '3840x2160': 1199, '7680x4320': 2399}
Sd = {'1920x1080': 888, '3840x2160': 1776, '7680x4320': 3552}
Se = {'1920x1080': 1031, '3840x2160': 2063, '7680x4320': 4127}
Sf = {'1920x1080': 1320, '3840x2160': 2640, '7680x4320': 5280}
Sg = {'1920x1080': 1607, '3840x2160': 3215, '7680x4320': 6431}
Sh = {'1920x1080': 1919, '3840x2160': 3839, '7680x4320': 7679}

La = {'1920x1080': 42 - 42, '3840x2160': 0, '7680x4320': 0}
Lb = {'1920x1080': 366 - 42, '3840x2160': 648, '7680x4320': 1296}
Lc = {'1920x1080': 387 - 42, '3840x2160': 690, '7680x4320': 1380}
Ld = {'1920x1080': 509 - 42, '3840x2160': 935, '7680x4320': 1871}
Le = {'1920x1080': 510 - 42, '3840x2160': 936, '7680x4320': 1872}
Lf = {'1920x1080': 653 - 42, '3840x2160': 1223, '7680x4320': 2447}
Lg = {'1920x1080': 654 - 42, '3840x2160': 1224, '7680x4320': 2448}
Lh = {'1920x1080': 776 - 42, '3840x2160': 1469, '7680x4320': 2939}
Li = {'1920x1080': 797 - 42, '3840x2160': 1511, '7680x4320': 3023}
Lj = {'1920x1080': 1121 - 42, '3840x2160': 2159, '7680x4320': 4319}


def bit_shift_10_to_16(data):
    return data * (2 ** 6)


def _get_pixel_rate(resolution):
    if resolution == '1920x1080':
        return 1
    elif resolution == '3840x2160':
        return 2
    else:
        raise ValueError('Invalid "resolution" is entered.')


def _get_widht_height_param(resolution):
    if resolution == '1920x1080':
        return 1920, 1080
    elif resolution == '3840x2160':
        return 3840, 2160
    elif resolution == '7680x4320':
        return 7680, 4320
    else:
        raise ValueError('Invalid "resolution" is entered.')


def color_bar_bt2111(resolution='1920x1080'):
    """
    ITU-R BT.2111-0 のテストパターンを作成する。

    ref: https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2111-0-201712-I!!PDF-E.pdf
    """
    None


def composite_pluge_higher_level(img, resolution, d_range):
    h_st_pos = Sd[resolution]
    h_ed_pos = Se[resolution] + 1
    v_st_pos = Le[resolution]
    v_ed_pos = Lf[resolution] + 1
    h_len = h_ed_pos - h_st_pos
    v_len = v_ed_pos - v_st_pos

    higher_img = np.ones((v_len, h_len, 3), dtype=np.uint16)
    higher_img *= bit_shift_10_to_16(PLUGE_HIGHER_LEVEL[d_range])
    img[v_st_pos:v_ed_pos, h_st_pos:h_ed_pos] = higher_img


def composite_horizontal_stripes(img, resolution):
    height_all = Lh[resolution] - Lc[resolution] + 1
    height = height_all // PLUGE_HORIZONTAL_STRIPE_TOAL_NUM

    # 本来は端数が出ないはず。万が一出た場合はエラーを出しておく
    if height * PLUGE_HORIZONTAL_STRIPE_TOAL_NUM != height_all:
        raise ValueError('"Lh" or "Lc" parameter is invalid.')

    h_st_pos = Sb[resolution]
    h_ed_pos = Sc[resolution] + 1
    v_st_pos = Lc[resolution]
    v_ed_pos = v_st_pos + height
    h_len = h_ed_pos - h_st_pos
    v_len = v_ed_pos - v_st_pos

    # まずは明るい方のループを実施
    # -----------------------------
    for idx in range(PLUGE_HORIZONTAL_STRIPE_EACH_NUM):
        stripe_img = np.ones((v_len, h_len, 3), dtype=np.uint16)
        stripe_img *= bit_shift_10_to_16(PLUGE_SLIGHTLY_LIGHTER_LEVEL)
        img[v_st_pos:v_ed_pos, h_st_pos:h_ed_pos] = stripe_img

        # update parameters
        v_st_pos = v_ed_pos + height
        v_ed_pos = v_st_pos + height

    # 次に暗い方のループを実施
    # -----------------------
    for idx in range(PLUGE_HORIZONTAL_STRIPE_EACH_NUM):
        stripe_img = np.ones((v_len, h_len, 3), dtype=np.uint16)
        stripe_img *= bit_shift_10_to_16(PLUGE_SLIGHTLY_DARKER_LEVEL)
        img[v_st_pos:v_ed_pos, h_st_pos:h_ed_pos] = stripe_img

        # update parameters
        v_st_pos = v_ed_pos + height
        v_ed_pos = v_st_pos + height


def composite_rectangular_patch(img, resolution):
    # まずは明るい方を実施
    # -----------------------------
    h_st_pos = Sf[resolution]
    h_ed_pos = Sg[resolution] + 1
    v_st_pos = Lb[resolution]
    v_ed_pos = Ld[resolution] + 1
    h_len = h_ed_pos - h_st_pos
    v_len = v_ed_pos - v_st_pos
    rectangle_img = np.ones((v_len, h_len, 3), dtype=np.uint16)
    rectangle_img *= bit_shift_10_to_16(PLUGE_SLIGHTLY_LIGHTER_LEVEL)
    img[v_st_pos:v_ed_pos, h_st_pos:h_ed_pos] = rectangle_img

    # 次に暗い方を実施
    # -----------------------------
    h_st_pos = Sf[resolution]
    h_ed_pos = Sg[resolution] + 1
    v_st_pos = Lg[resolution]
    v_ed_pos = Li[resolution] + 1
    h_len = h_ed_pos - h_st_pos
    v_len = v_ed_pos - v_st_pos
    rectangle_img = np.ones((v_len, h_len, 3), dtype=np.uint16)
    rectangle_img *= bit_shift_10_to_16(PLUGE_SLIGHTLY_DARKER_LEVEL)
    img[v_st_pos:v_ed_pos, h_st_pos:h_ed_pos] = rectangle_img


def write_dpx(sample_file, out_file, img):
    """参考：https://github.com/guerilla-di/depix"""

    with open(sample_file, 'rb') as f:
        header = f.read(8192)

    with open(out_file, 'wb') as f:
        f.write(header)
        img = np.uint32(img) >> 6
        raw = ((img[:, :, 0] & 0x000003FF) << 22) | ((img[:, :, 1] & 0x000003FF) << 12) | ((img[:, :, 2] & 0x000003FF) << 2)
        raw = raw.byteswap()
        raw.tofile(f, sep="")


def save_test_pattern(img, resolution, prefix='pluge_sdr'):
    file_str = "./img/{:s}_{:s}.{:s}"
    file_name_tiff = file_str.format(prefix, resolution, "tif   ")
    cv2.imwrite(file_name_tiff, img)

    # DPX
    file_name_dpx = file_str.format(prefix, resolution, "dpx")
    file_name_sample = "HDR_TEST_PATTEN_{}_bg_0.20nits.dpx".format(resolution)
    file_name_sample = os.path.join("sample_dpx", file_name_sample)
    write_dpx(file_name_sample, file_name_dpx, img)


def pluge_pattern(resolution='1920x1080', d_range='sdr'):
    """
    ITU-R BT.814-4 の PLUGE パターンを作成する。

    ref: https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.814-4-201807-I!!PDF-E.pdf
    """
    width, height = _get_widht_height_param(resolution)
    img = np.ones((height, width, 3), dtype=np.uint16)
    img *= bit_shift_10_to_16(PLUGE_BLACK_LEVEL)

    composite_pluge_higher_level(img, resolution, d_range)
    composite_horizontal_stripes(img, resolution)
    composite_rectangular_patch(img, resolution)

    tpg.preview_image(img)
    save_test_pattern(img, resolution, prefix='pluge_' + d_range)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    color_bar_bt2111(resolution='1920x1080')
    pluge_pattern(resolution='1920x1080', d_range='sdr')
    pluge_pattern(resolution='3840x2160', d_range='sdr')

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

W100 = {'hlg': [940, 940, 940], 'pq': [940, 940, 940], 'pq_full': [1023, 1023, 1023]}
Y100 = {'hlg': [940, 940, 64], 'pq': [940, 940, 64], 'pq_full': [1023, 1023, 0]}
C100 = {'hlg': [64, 940, 940], 'pq': [64, 940, 940], 'pq_full': [0, 1023, 1023]}
G100 = {'hlg': [64, 940, 64], 'pq': [64, 940, 64], 'pq_full': [0, 1023, 0]}
M100 = {'hlg': [940, 64, 940], 'pq': [940, 64, 940], 'pq_full': [1023, 0, 1023]}
R100 = {'hlg': [940, 64, 64], 'pq': [940, 64, 64], 'pq_full': [1023, 0, 0]}
B100 = {'hlg': [64, 64, 940], 'pq': [64, 64, 940], 'pq_full': [0, 0, 1023]}
W_Mid = {'hlg': [721, 721, 721], 'pq': [572, 572, 572], 'pq_full': [593, 593, 593]}
Y_Mid = {'hlg': [721, 721, 64], 'pq': [572, 572, 64], 'pq_full': [593, 593, 0]}
C_Mid = {'hlg': [64, 721, 721], 'pq': [64, 572, 572], 'pq_full': [0, 593, 593]}
G_Mid = {'hlg': [64, 721, 64], 'pq': [64, 572, 64], 'pq_full': [0, 593, 0]}
M_Mid = {'hlg': [721, 64, 721], 'pq': [572, 64, 572], 'pq_full': [593, 0, 593]}
R_Mid = {'hlg': [721, 64, 64], 'pq': [572, 64, 64], 'pq_full': [593, 0, 0]}
B_Mid = {'hlg': [64, 64, 721], 'pq': [64, 64, 572], 'pq_full': [0, 0, 593]}
W40 = {'hlg': [414, 414, 414], 'pq': [414, 414, 414], 'pq_full': [409, 409, 409]}
Step7 = {'hlg': [4, 4, 4], 'pq': [4, 4, 4], 'pq_full': [0, 0, 0]}
Step0 = {'hlg': [64, 64, 64], 'pq': [64, 64, 64], 'pq_full': [0, 0, 0]}
Step10 = {'hlg': [152, 152, 152], 'pq': [152, 152, 152], 'pq_full': [102, 102, 102]}
Step20 = {'hlg': [239, 239, 239], 'pq': [239, 239, 239], 'pq_full': [205, 205, 205]}
Step30 = {'hlg': [327, 327, 327], 'pq': [327, 327, 327], 'pq_full': [307, 307, 307]}
Step40 = {'hlg': [414, 414, 414], 'pq': [414, 414, 414], 'pq_full': [409, 409, 409]}
Step50 = {'hlg': [502, 502, 502], 'pq': [502, 502, 502], 'pq_full': [512, 512, 512]}
Step60 = {'hlg': [590, 590, 590], 'pq': [590, 590, 590], 'pq_full': [614, 614, 614]}
Step70 = {'hlg': [677, 677, 677], 'pq': [677, 677, 677], 'pq_full': [716, 716, 716]}
Step80 = {'hlg': [765, 765, 765], 'pq': [765, 765, 765], 'pq_full': [818, 818, 818]}
Step90 = {'hlg': [852, 852, 852], 'pq': [852, 852, 852], 'pq_full': [921, 921, 921]}
Step100 = {'hlg': [940, 940, 940], 'pq': [940, 940, 940], 'pq_full': [1023, 1023, 1023]}
Step109 = {'hlg': [1019, 1019, 1019], 'pq': [1019, 1019, 1019], 'pq_full': [1023, 1023, 1023]}
Y75 = {'hlg': [713, 719, 316], 'pq': [568, 571, 381], 'pq_full': [589, 592, 370]}
C75 = {'hlg': [538, 709, 718], 'pq': [484, 566, 571], 'pq_full': [491, 586, 592]}
G75 = {'hlg': [512, 706, 296], 'pq': [474, 564, 368], 'pq_full': [478, 584, 355]}
M75 = {'hlg': [651, 286, 705], 'pq': [536, 361, 564], 'pq_full': [551, 347, 584]}
R75 = {'hlg': [639, 269, 164], 'pq': [530, 350, 256], 'pq_full': [544, 334, 225]}
B75 = {'hlg': [227, 147, 702], 'pq': [317, 236, 562], 'pq_full': [296, 201, 582]}
K0 = {'hlg': [64, 64, 64], 'pq': [64, 64, 64], 'pq_full': [0, 0, 0]}
Ku2 = {'hlg': [48, 48, 48], 'pq': [48, 48, 48], 'pq_full': [0, 0, 0]}
K2 = {'hlg': [80, 80, 80], 'pq': [80, 80, 80], 'pq_full': [20, 20, 20]}
K4 = {'hlg': [99, 99, 99], 'pq': [99, 99, 99], 'pq_full': [41, 41, 41]}

Aa = {'1920x1080': 1920, '3840x2160': 3840, '7680x4320': 7680}
Bb = {'1920x1080': 1080, '3840x2160': 2160, '7680x4320': 4320}
Cc = {'1920x1080': 240, '3840x2160': 480, '7680x4320': 960}
Dd = {'1920x1080': 206, '3840x2160': 412, '7680x4320': 824}
Ee = {'1920x1080': 204, '3840x2160': 408, '7680x4320': 816}
Ff = {'1920x1080': 136, '3840x2160': 272, '7680x4320': 544}
Gg = {'1920x1080': 70, '3840x2160': 140, '7680x4320': 280}
Hh = {'1920x1080': 68, '3840x2160': 136, '7680x4320': 272}
Ii = {'1920x1080': 238, '3840x2160': 476, '7680x4320': 952}
Jj = {'1920x1080': 438, '3840x2160': 876, '7680x4320': 1752}
Kk = {'1920x1080': 282, '3840x2160': 564, '7680x4320': 1128}

AA = {'hlg': {'1920x1080': 1680, '3840x2160': 3360, '7680x4320': 6720}, 'pq': {'1920x1080': 1680, '3840x2160': 3360, '7680x4320': 6720}, 'pq_full': {'1920x1080': 1680, '3840x2160': 3360, '7680x4320': 6720}}
BB = {'hlg': {'1920x1080': 559, '3840x2160': 1118, '7680x4320': 2236}, 'pq': {'1920x1080': 559, '3840x2160': 1118, '7680x4320': 2236}, 'pq_full': {'1920x1080': 551, '3840x2160': 1102, '7680x4320': 2204}}
CC = {'hlg': {'1920x1080': 1015, '3840x2160': 2030, '7680x4320': 4060}, 'pq': {'1920x1080': 1015, '3840x2160': 2030, '7680x4320': 4060}, 'pq_full': {'1920x1080': 1023, '3840x2160': 2046, '7680x4320': 4092}}
DD = {'hlg': {'1920x1080': 106, '3840x2160': 212, '7680x4320': 424}, 'pq': {'1920x1080': 106, '3840x2160': 212, '7680x4320': 424}, 'pq_full': {'1920x1080': 106, '3840x2160': 212, '7680x4320': 424}}

COLOR_BAR_EOTF_SET = set(['hlg', 'pq', 'pq_full'])


def bit_shift_10_to_16(data):
    return data * (2 ** 6)


def _get_pixel_rate(resolution):
    if resolution == '1920x1080':
        return 1
    elif resolution == '3840x2160':
        return 2
    elif resolution == '7680x4320':
        return 4
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


def _update_param_horizontal_cb(h_st, h_ed, h_diff):
    h_st_temp = h_ed
    h_ed_temp = h_st_temp + h_diff

    return h_st_temp, h_ed_temp


def _composite_color_img(img, h_st, h_ed, v_st, v_ed, rgb):
    h_len = h_ed - h_st
    v_len = v_ed - v_st

    img_temp = np.ones((v_len, h_len, 3), dtype=np.uint16)
    img_temp[..., 0] *= bit_shift_10_to_16(rgb[0])
    img_temp[..., 1] *= bit_shift_10_to_16(rgb[1])
    img_temp[..., 2] *= bit_shift_10_to_16(rgb[2])
    img[v_st:v_ed, h_st:h_ed] = img_temp


def composite_color_bar_rgbmyc(img, resolution, eotf):
    """
    カラーバー上部の中間階調のベタ色を作成
    """
    # 左端Gray
    h_st_pos = 0
    h_ed_pos = h_st_pos + Cc[resolution]
    v_st_pos = 0
    v_ed_pos = v_st_pos + Bb[resolution]//2 + Bb[resolution]//12
    _composite_color_img(img, h_st_pos, h_ed_pos, v_st_pos, v_ed_pos,
                         Step40[eotf])

    d = Dd[resolution]
    e = Ee[resolution]
    c = Cc[resolution]
    h_list = [d, d, d, e, d, d, d, c]
    rgb_list = [W_Mid, Y_Mid, C_Mid, G_Mid, M_Mid, R_Mid, B_Mid, W40]

    for h, rgb in zip(h_list, rgb_list):
        h_st_pos, h_ed_pos =\
            _update_param_horizontal_cb(h_st_pos, h_ed_pos, h)
        _composite_color_img(img, h_st_pos, h_ed_pos, v_st_pos, v_ed_pos,
                             rgb[eotf])


def composite_color_bar_top_max_level(img, resolution, eotf):
    """
    カラーバー上部の最大輝度のところを作る
    """
    # 左端100%White
    h_st_pos = Cc[resolution]
    h_ed_pos = h_st_pos + Dd[resolution]
    v_st_pos = 0
    v_ed_pos = v_st_pos + Bb[resolution]//12
    _composite_color_img(img, h_st_pos, h_ed_pos, v_st_pos, v_ed_pos,
                         W100[eotf])

    d = Dd[resolution]
    e = Ee[resolution]
    h_list = [d, d, e, d, d, d]
    rgb_list = [Y100, C100, G100, M100, R100, B100]

    for h, rgb in zip(h_list, rgb_list):
        h_st_pos, h_ed_pos =\
            _update_param_horizontal_cb(h_st_pos, h_ed_pos, h)
        _composite_color_img(img, h_st_pos, h_ed_pos, v_st_pos, v_ed_pos,
                             rgb[eotf])


def composite_color_bar_step_level(img, resolution, eotf):
    """
    カラーバー中段のグレースケール（Step）を作る
    """

    # 左端100%White
    h_st_pos = 0
    h_ed_pos = h_st_pos + Cc[resolution]
    v_st_pos = Bb[resolution]//12 + Bb[resolution]//2
    v_ed_pos = v_st_pos + Bb[resolution]//12
    _composite_color_img(img, h_st_pos, h_ed_pos, v_st_pos, v_ed_pos,
                         W_Mid[eotf])

    d = Dd[resolution]
    d2 = Dd[resolution] // 2
    e2 = Ee[resolution] // 2
    c = Cc[resolution]
    h_list = [d, d2, d2, d2, d2, e2, e2, d2, d2, d2, d2, d2, d2, c]
    rgb_list = [Step7, Step0, Step10, Step20, Step30, Step40, Step50,
                Step60, Step70, Step80, Step90, Step100, Step109, W_Mid]
    for h, rgb in zip(h_list, rgb_list):
        h_st_pos, h_ed_pos =\
            _update_param_horizontal_cb(h_st_pos, h_ed_pos, h)
        _composite_color_img(img, h_st_pos, h_ed_pos, v_st_pos, v_ed_pos,
                             rgb[eotf])


def color_bar_bt2111(resolution='1920x1080', eotf='hlg'):
    """
    ITU-R BT.2111-0 のテストパターンを作成する。

    ref: https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2111-0-201712-I!!PDF-E.pdf
    """

    width, height = _get_widht_height_param(resolution)
    if eotf not in COLOR_BAR_EOTF_SET:
        raise ValueError('"eotf" parameter is invalid')

    img = np.zeros((height, width, 3), dtype=np.uint16)

    composite_color_bar_rgbmyc(img, resolution, eotf)
    composite_color_bar_top_max_level(img, resolution, eotf)
    composite_color_bar_step_level(img, resolution, eotf)

    tpg.preview_image(img)
    # save_test_pattern(img, resolution, prefix='color_bar_' + eotf)


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
    color_bar_bt2111(resolution='1920x1080', eotf='hlg')
    # color_bar_bt2111(resolution='1920x1080', eotf='pq')
    # color_bar_bt2111(resolution='1920x1080', eotf='pq_full')
    # pluge_pattern(resolution='1920x1080', d_range='sdr')
    # pluge_pattern(resolution='3840x2160', d_range='sdr')

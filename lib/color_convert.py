#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 概要
評価用のテストパターン作成ツール集

# 使い方

"""

import sys
import numpy as np
from scipy import linalg
import common
import plot_utility as pu
import matplotlib.pyplot as plt


const_lab_delta = 6.0/29.0
const_lab_xn_d65 = 95.047
const_lab_yn_d65 = 100.0
const_lab_zn_d65 = 108.883

""" src : http://www.easyrgb.com/en/math.php """
const_d50_large_xyz = [96.422, 100.000, 82.521]
const_d65_large_xyz = [95.047, 100.000, 108.883]

const_sRGB_xy = [[0.64, 0.33],
                 [0.30, 0.60],
                 [0.15, 0.06]]

const_AdobeRGB_xy = [[0.6400, 0.3300],
                     [0.2100, 0.7100],
                     [0.1500, 0.0600]]

const_ntsc_xy = [[0.67, 0.33],
                 [0.21, 0.71],
                 [0.14, 0.08],
                 [0.310, 0.316]]

const_rec601_xy = const_ntsc_xy

const_rec709_xy = const_sRGB_xy

const_rec2020_xy = [[0.708, 0.292],
                    [0.170, 0.797],
                    [0.131, 0.046]]

const_sRGB_xyz = [[0.64, 0.33, 0.03],
                  [0.30, 0.60, 0.10],
                  [0.15, 0.06, 0.79]]

const_xyz_to_lms = [[0.8951000, 0.2664000, -0.1614000],
                    [-0.7502000, 1.7135000, 0.0367000],
                    [0.0389000, -0.0685000, 1.0296000]]

const_rgb_to_large_xyz = [[2.7689, 1.7517, 1.1302],
                          [1.0000, 4.5907, 0.0601],
                          [0.0000, 0.0565, 5.5943]]

const_d65_xy = [0.31271, 0.32902]
const_d50_xy = [0.34567, 0.35850]

const_rec601_y_coef = [0.2990, 0.5870, 0.1140]
const_rec709_y_coef = [0.2126, 0.7152, 0.0722]

const_srgb_eotf_threshold = 0.04045
const_srgb_oetf_threshold = 0.0031308


def get_white_point_conv_matrix(src=const_d65_large_xyz,
                                dst=const_d50_large_xyz):
    """
    # brief
    execute bradford transform.
    # reference
    http://w3.kcua.ac.jp/~fujiwara/infosci/colorspace/bradford.html
    """

    src = np.array(src)
    dst = np.array(dst)

    # LMS値を求めよう
    # --------------------------------------
    ma = np.array(const_xyz_to_lms)
    ma_inv = linalg.inv(ma)

    src_LMS = ma.dot(src)
    dst_LMS = ma.dot(dst)

    # M行列を求めよう
    # --------------------------------------
    mtx = [[dst_LMS[0]/src_LMS[0], 0.0, 0.0],
           [0.0, dst_LMS[1]/src_LMS[1], 0.0],
           [0.0, 0.0, dst_LMS[2]/src_LMS[2]]]

    m_mtx = ma_inv.dot(mtx).dot(ma)

    return m_mtx


def xy_to_xyz_internal(xy):
    rz = 1 - (xy[0][0] + xy[0][1])
    gz = 1 - (xy[1][0] + xy[1][1])
    bz = 1 - (xy[2][0] + xy[2][1])

    xyz = [[xy[0][0], xy[0][1], rz],
           [xy[1][0], xy[1][1], gz],
           [xy[2][0], xy[2][1], bz]]

    return xyz


def get_rgb_to_xyz_matrix(gamut=const_sRGB_xy, white=const_d65_large_xyz):

    # まずは xyz 座標を準備
    # ------------------------------------------------
    if np.array(gamut).shape == (3, 2):
        gamut = xy_to_xyz_internal(gamut)
    elif np.array(gamut).shape == (3, 3):
        pass
    else:
        print("============ Fatal Error ============")
        print("invalid xy gamut parameter.")
        print("=====================================")
        sys.exit(1)

    gamut_mtx = np.array(gamut)

    # 白色点の XYZ を算出。Y=1 となるように調整
    # ------------------------------------------------
    large_xyz = [white[0] / white[1], white[1] / white[1], white[2] / white[1]]
    large_xyz = np.array(large_xyz)

    # Sr, Sg, Sb を算出
    # ------------------------------------------------
    s = linalg.inv(gamut_mtx[0:3]).T.dot(large_xyz)

    # RGB2XYZ 行列を算出
    # ------------------------------------------------
    s_matrix = [[s[0], 0.0,  0.0],
                [0.0,  s[1], 0.0],
                [0.0,  0.0,  s[2]]]
    s_matrix = np.array(s_matrix)
    rgb2xyz_mtx = gamut_mtx.T.dot(s_matrix)

    return rgb2xyz_mtx


def color_cvt(img, mtx):
    """
    # 概要
    img に対して mtx を適用する。
    # 注意事項
    例によって、RGBの並びを考えている。BGRの並びの場合は
    img[:, :, ::-1] してから関数をコールすること。
    """
    try:
        img_max = np.iinfo(img.dtype).max
        img_min = np.iinfo(img.dtype).min
    except ValueError:
        img_max = np.finfo(img.dtype).max
        img_min = np.finfo(img.dtype).min

    r, g, b = np.dsplit(img, 3)
    ro = r * mtx[0][0] + g * mtx[0][1] + b * mtx[0][2]
    go = r * mtx[1][0] + g * mtx[1][1] + b * mtx[1][2]
    bo = r * mtx[2][0] + g * mtx[2][1] + b * mtx[2][2]

    out_img = np.dstack((ro, go, bo))

    out_img[out_img < img_min] = img_min
    out_img[out_img > img_max] = img_max

    return out_img


def xyY_to_XYZ(xyY):
    """
    # 概要
    xyYからXYZを計算する

    # 入力データ
    numpy形式。shape = (1, N, 3)
    """
    small_x, small_y, large_y = np.dsplit(xyY, 3)
    small_z = 1 - small_x - small_y
    large_x = large_y / small_y * small_x
    large_z = large_y / small_y * small_z

    return np.dstack((large_x, large_y, large_z))


def srgb_to_linear(img):
    """
    # 概要
    sRGB の 画像をリニアに戻す

    # 注意事項
    img は float型であること
    """

    float_list = [np.float, np.float16, np.float32]

    if img.dtype not in float_list:
        raise TypeError('img must be float type!')

    if np.sum(img > 1) > 0:
        raise ValueError('img must be normalized to 0 .. 1')

    lower_img = img / 12.92
    upper_img = ((img + 0.055) / 1.055) ** 2.4

    out_img = lower_img * (img <= const_srgb_eotf_threshold)\
        + upper_img * (img >= const_srgb_eotf_threshold)

    return out_img


def linear_to_srgb(img):
    """
    # 概要
    リニアの画像をsRGBに変換する

    # 注意事項
    img は float型であること
    """

    float_list = [np.float, np.float16, np.float32]

    if img.dtype not in float_list:
        raise TypeError('img must be float type!')

    if np.sum(img > 1) > 0:
        raise ValueError('img must be normalized to 0 .. 1')

    lower_img = img * 12.92
    upper_img = 1.055 * (img ** (1/2.4)) - 0.055

    out_img = lower_img * (img <= const_srgb_oetf_threshold)\
        + upper_img * (img >= const_srgb_oetf_threshold)

    return out_img


def xyY_to_RGB(xyY, gamut=const_sRGB_xy, white=const_d65_large_xyz):
    """
    # 概要
    xyY から RGB値を算出する
    # 入力データ
    numpy形式。shape = (N, M, 3)
    """
    if not common.is_img_shape(xyY):
        raise TypeError('xyY shape must be (N, M, 3)')

    large_xyz = xyY_to_XYZ(xyY)
    rgb = large_xyz_to_rgb(large_xyz=large_xyz, gamut=gamut, white=white)
    return rgb


def large_xyz_to_rgb(large_xyz, gamut=const_sRGB_xy,
                     white=const_d65_large_xyz):
    """
    # 概要
    XYZ から RGB値を算出する
    # 入力データ
    numpy形式。shape = (N, M, 3)
    """
    if not common.is_img_shape(large_xyz):
        raise TypeError('XYZ shape must be (N, M, 3)')
    cvt_mtx = get_rgb_to_xyz_matrix(gamut=gamut, white=white)
    print(cvt_mtx)
    cvt_mtx = linalg.inv(cvt_mtx)
    rgb = color_cvt(large_xyz, cvt_mtx)
    return rgb


def _func_t(t):
    upper = (t > const_lab_delta) * (t ** 3)
    lower = (t <= const_lab_delta) * 3 * (const_lab_delta ** 2) * (t - 4/29)
    return upper + lower


def lab_to_large_xyz(lab, white=const_d50_large_xyz):
    """
    # 概要
    L*a*b* から XYZ値を算出する
    # 入力データ
    numpy形式。shape = (N, M, 3)
    """
    if not common.is_img_shape(lab):
        raise TypeError('lab shape must be (N, M, 3)')

    l, a, b = np.dsplit(lab, 3)
    large_x = white[0] * _func_t((l + 16)/116 + a/500)
    large_y = white[1] * _func_t((l + 16)/116)
    large_z = white[2] * _func_t((l + 16)/116 - b/200)

    return np.dstack((large_x, large_y, large_z))


if __name__ == '__main__':
    # lab = np.ones((1, 1, 3))
    # lab[0][0][0] = 42.101
    # lab[0][0][1] = 53.378
    # lab[0][0][2] = 28.19
    # # lab[0][0][0] = 96.539
    # # lab[0][0][1] = -0.425
    # # lab[0][0][2] = 1.186
    # large_xyz = lab_to_large_xyz(lab, white=const_d50_large_xyz)
    # print(large_xyz)
    # d50_to_d65_mtx = get_white_point_conv_matrix(src=const_d50_large_xyz,
    #                                              dst=const_d65_large_xyz)
    # large_xyz = color_cvt(img=large_xyz, mtx=d50_to_d65_mtx)
    # print(large_xyz)
    # rgb = large_xyz_to_rgb(large_xyz, const_sRGB_xy)
    # print(rgb)
    # print(linear_to_srgb(rgb/100) * 255)
    # print(get_rgb_to_xyz_matrix(gamut=const_sRGB_xy, white=const_d65_large_xyz))
    test_color_cvt()

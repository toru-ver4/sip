#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RGB --> YCbCr --> RGB 変換で係数間違えを犯した場合の
情報欠落について調査する。
"""

import os
import numpy as np
import cv2
from colour import RGB_to_YCbCr, YCbCr_to_RGB, RGB_to_XYZ, XYZ_to_xyY
from colour import RGB_COLOURSPACES
from colour.utilities import CaseInsensitiveMapping
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import plot_utility as pu
import test_pattern_generator2 as tpg
import rgb_yuv_rgb_transformation as ryr


BT601 = 'ITU-R BT.601'
BT709 = 'ITU-R BT.709'
BT2020 = 'ITU-R BT.2020'
BASE_SRC_16BIT_PATTERN = "./img/src_16bit.tiff"
# BASE_SRC_8BIT_PATTERN = "./img/src_8bit.tiff"
BASE_SRC_8BIT_PATTERN = "./img/src_8bit_trim2.tif"

YCBCR_WEIGHTS = CaseInsensitiveMapping({
    BT601: np.array([0.2990, 0.1140]),
    BT709: np.array([0.2126, 0.0722]),
    BT2020: np.array([0.2627, 0.0593])
})

BT2020_Y_PARAM = np.array([0.2627, 0.6780, 0.0593])

# カラーユニバーサルデザイン推奨配色セット制作委員会資料より抜粋
R_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(255, 75, 0)
G_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(3, 175, 122)
B_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(0, 90, 255)
K_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(132, 145, 158)

# R_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(255, 202, 191)
# G_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(216, 242, 85)
# B_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(191, 228, 255)
# K_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(200, 200, 203)


def calc_yuv_transform_matrix(y_param=BT2020_Y_PARAM):
    """
    RGB to YUV 変換のMatrixを算出する。
    """
    r = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0])
    diff_r = r - y_param
    coef_r = np.sum(np.absolute(diff_r))
    diff_b = b - y_param
    coef_b = np.sum(np.absolute(diff_b))
    mtx = np.array([y_param, diff_b/coef_b, diff_r/coef_r])
    print(mtx)
    return mtx


def img_read(filename):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    """
    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    return img[:, :, ::-1]


def img_write(filename, img):
    """
    OpenCV の BGR 配列が怖いので並べ替えるwrapperを用意。
    """
    cv2.imwrite(filename, img[:, :, ::-1])


def convert_16bit_tiff_to_8bit_tiff():
    """
    16bitのテストパターンを8bitに変換する。
    8bitの場合は1023じゃなくて1020で正規化するのがポイント
    """
    in_name = BASE_SRC_16BIT_PATTERN
    out_name = BASE_SRC_8BIT_PATTERN
    img = img_read(in_name)
    img = (img / 0xFFFF) * 1023 / 4
    img[img > 255] = 255
    img = np.uint8(np.round(img))
    img_write(out_name, img)


def make_wrong_ycbcr_conv_image_all_pattern():
    rgb_to_ycbcr_coef_list = [BT601, BT709, BT2020]
    ycbcr_to_rgb_coef_list = [BT601, BT709, BT2020]

    for src_coef in rgb_to_ycbcr_coef_list:
        for dst_coef in ycbcr_to_rgb_coef_list:
            make_wrong_ycbcr_conv_image(src_coef, dst_coef)


def make_wrong_ycbcr_conv_image(src_coef=BT709, dst_coef=BT2020):
    src_img = img_read(BASE_SRC_8BIT_PATTERN)
    ycbcr_img = ryr.convert_to_ycbcr(src_img, src_coef, bit_depth=8,
                                     limited_range=True)
    dst_img = ryr.convert_to_rgb(ycbcr_img, dst_coef, bit_depth=8,
                                 limited_range=True).astype(np.uint8)
    file_name = "./img/{}_{}.tiff".format(src_coef, dst_coef)
    img_write(file_name, dst_img)


def concatenate_all_images():
    """
    各係数の画像を1枚にまとめてみる。
    """
    h_list = [BT601, BT709, BT2020]
    v_list = [BT601, BT709, BT2020]
    v_buf = []
    for v_val in v_list:
        h_buf = []
        for h_val in h_list:
            fname = "./img/{}_{}.tiff".format(v_val, h_val)
            print(fname)
            h_buf.append(img_read(fname))
        v_buf.append(np.hstack(h_buf))
    img = np.vstack(v_buf)

    img_write("./img/all.tiff", img)


def linear_rgb_to_cielab(rgb):
    """
    LinearなRGB値をRGB --> XYZ --> L*a*b* に変換する。
    rgb は [0:1] に正規化済みの前提ね。
    """
    print(rgb)


def test_func():
    x = np.linspace(0, 1, 1024)
    rgb = np.dstack((x, x, x))
    linear_rgb_to_cielab(rgb)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_func()
    # calc_yuv_transform_matrix()
    # convert_16bit_tiff_to_8bit_tiff()
    # make_wrong_ycbcr_conv_image_all_pattern()
    # concatenate_all_images()

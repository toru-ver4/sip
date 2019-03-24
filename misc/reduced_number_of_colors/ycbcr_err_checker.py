#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YCbCrの係数誤りを確認するテストパターンの作成
"""

import os
import numpy as np
import cv2
from colour import RGB_to_XYZ, XYZ_to_Lab
from colour import delta_E
from colour.utilities import CaseInsensitiveMapping
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# original libraries
import plot_utility as pu
import test_pattern_generator2 as tpg
import rgb_yuv_rgb_transformation as ryr
import ycbcr_wrong_transformation as ywt


BT601 = 'ITU-R BT.601'
BT709 = 'ITU-R BT.709'
BT2020 = 'ITU-R BT.2020'
# BASE_SRC_8BIT_PATTERN = "./img/src_8bit_128_clip.tiff"
BASE_SRC_8BIT_PATTERN = "./img/pattern.png"

YCBCR_WEIGHTS = CaseInsensitiveMapping({
    BT601: np.array([0.2990, 0.1140]),
    BT709: np.array([0.2126, 0.0722]),
    BT2020: np.array([0.2627, 0.0593])
})

# カラーユニバーサルデザイン推奨配色セット制作委員会資料より抜粋
R_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(255, 75, 0)
G_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(3, 175, 122)
B_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(0, 90, 255)
K_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(132, 145, 158)

# R_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(255, 202, 191)
# G_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(216, 242, 85)
# B_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(191, 228, 255)
# K_BAR_COLOR = "#{:02x}{:02x}{:02x}".format(200, 200, 203)


def make_wrong_ycbcr_conv_image_all_pattern():
    rgb_to_ycbcr_coef_list = [BT601, BT709, BT2020]
    ycbcr_to_rgb_coef_list = [BT601, BT709, BT2020]

    for src_coef in rgb_to_ycbcr_coef_list:
        for dst_coef in ycbcr_to_rgb_coef_list:
            make_wrong_ycbcr_conv_image(src_coef, dst_coef)


def make_wrong_ycbcr_conv_image(src_coef=BT709, dst_coef=BT2020):
    src_img = ywt.img_read(BASE_SRC_8BIT_PATTERN)
    dst_img = ywt.convert_rgb_to_ycbcr_to_rgb(src_img, src_coef, dst_coef)
    file_name = "./img/clip_{}_{}.png".format(src_coef, dst_coef)
    ywt.img_write(file_name, dst_img)


def make_wrong_pattern_img():
    """
    クリップした画像をベースに書く係数での変換を試す。
    """
    make_wrong_ycbcr_conv_image_all_pattern()


def clip_over_512_level():
    """
    512階調から上を強制クリップする。
    YCbCrの誤変換でのクリップ位置ズレ確認のため
    """
    img = ywt.img_read("./img/src_8bit.tiff")
    img[img > 128] = 128
    ywt.img_write("./img/src_8bit_128_clip.tiff", img)


def analyze_cyan_level(cv=192):
    """
    cyan = (0, 192, 192) の 色の変動を確認
    """
    cyan = [0, cv, cv]
    cyan = np.array(cyan, dtype=np.uint8).reshape((1, 1, 3))

    rgb_to_ycbcr_coef_list = [BT601, BT709, BT2020]
    ycbcr_to_rgb_coef_list = [BT601, BT709, BT2020]

    for src_coef in rgb_to_ycbcr_coef_list:
        for dst_coef in ycbcr_to_rgb_coef_list:
            dst_img = ywt.convert_rgb_to_ycbcr_to_rgb(cyan, src_coef, dst_coef)
            print(dst_img)


def make_mono_clip_pattern(color_mask=[1, 1, 1]):
    cv_max = 255
    color_mask = np.array(color_mask, dtype=np.uint8)
    step = 8
    total = 33
    width_base = 480
    height = ((width_base * 2) // total)
    width = height

    max_img = np.ones([height, width, 3], dtype=np.uint8) * cv_max

    img_h_buf = []

    for idx in range(total - 1):
        img_v_buf = []
        code_value = 192 + step * idx
        if code_value > cv_max:
            code_value = cv_max
        temp_img = np.ones_like(max_img) * code_value * color_mask
        img_v_buf.append(temp_img)
        img_v_buf.append(max_img)
        img_h_buf.append(np.vstack(img_v_buf))

    img_v_buf = []
    temp_img = max_img.copy() * color_mask
    img_v_buf.append(temp_img)
    img_v_buf.append(max_img)
    img_h_buf.append(np.vstack(img_v_buf))

    img = np.hstack(img_h_buf)

    tpg.preview_image(img)

    return img


def make_test_test_pattern():
    img_buf = []
    img_buf.append(make_mono_clip_pattern([1, 1, 1]))
    img_buf.append(make_mono_clip_pattern([0, 1, 1]))
    img = np.vstack(img_buf)

    # tpg.preview_image(img)
    ywt.img_write("./img/pattern.png", img)


def test_func():
    # clip_over_512_level()
    # analyze_cyan_level(cv=128)
    # make_mono_clip_pattern()
    make_test_test_pattern()
    make_wrong_pattern_img()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_func()

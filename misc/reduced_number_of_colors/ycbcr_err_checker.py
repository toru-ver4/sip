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


def get_clip_test_img_name(src_coef=BT709, dst_coef=BT2020):
    return "./img/clip_{}_{}.png".format(src_coef, dst_coef)


def make_wrong_ycbcr_conv_image(src_coef=BT709, dst_coef=BT2020):
    src_img = ywt.img_read(BASE_SRC_8BIT_PATTERN)
    dst_img = ywt.convert_rgb_to_ycbcr_to_rgb(src_img, src_coef, dst_coef)
    file_name = get_clip_test_img_name(src_coef, dst_coef)
    ywt.img_write(file_name, dst_img)


def concatenate_all_images(get_fname_func=get_clip_test_img_name):
    """
    各係数の画像を1枚にまとめてみる。
    """
    h_list = [BT601, BT709, BT2020]
    v_list = [BT601, BT709, BT2020]
    v_buf = []
    for v_val in v_list:
        h_buf = []
        for h_val in h_list:
            fname = get_fname_func(h_val, v_val)
            print(fname)
            h_buf.append(ywt.img_read(fname))
        v_buf.append(np.hstack(h_buf))
    img = np.vstack(v_buf)

    ywt.img_write("./img/all.png", img)


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


def analyze_video_level(rgb=[0, 192, 192]):
    """
    任意のビデオレベルの変動を確認
    """
    rgb = np.array(rgb, dtype=np.uint8).reshape((1, 1, 3))

    rgb_to_ycbcr_coef_list = [BT601, BT709, BT2020]
    ycbcr_to_rgb_coef_list = [BT601, BT709, BT2020]

    result_buf = []

    for src_coef in rgb_to_ycbcr_coef_list:
        for dst_coef in ycbcr_to_rgb_coef_list:
            dst_img = ywt.convert_rgb_to_ycbcr_to_rgb(rgb, src_coef, dst_coef)
            result_buf.append(dst_img)
            print(dst_img)

    return np.hstack(result_buf)


def make_mono_clip_pattern(color_mask=[1, 1, 1]):
    cv_max = 255
    color_mask = np.array(color_mask, dtype=np.uint8)
    step = 8
    block_num = 4
    total = 6
    width_base = 480
    height = (width_base // (total * (block_num)))
    width = height

    max_img = np.ones([height, width, 3], dtype=np.uint8) * cv_max * color_mask

    img_h_buf = []

    for idx in range(total):
        code_value = 224 + step * idx
        if code_value > cv_max:
            code_value = cv_max
        cur_img = np.ones_like(max_img) * code_value * color_mask

        block_img = np.hstack([cur_img, max_img])
        line_img = np.hstack([block_img for x in range(block_num // 2)])
        line_img_inv = line_img.copy()[:, ::-1, :]
        temp_img = np.vstack([line_img if x % 2 == 0 else line_img_inv
                              for x in range(block_num)])

        img_h_buf.append(temp_img)

    img = np.hstack(img_h_buf)

    tpg.preview_image(img)

    return img


def make_test_test_pattern():
    img_buf = []
    img_buf.append(make_mono_clip_pattern([1, 1, 1]))
    img_buf.append(make_mono_clip_pattern([1, 0, 1]))
    img_buf.append(make_mono_clip_pattern([0, 1, 1]))
    img = np.vstack(img_buf)

    out_img = np.zeros((img.shape[0] + 10, img.shape[1] + 10, img.shape[2]))
    out_img[0:img.shape[0], 0:img.shape[1], :] = img

    # tpg.preview_image(img)
    ywt.img_write("./img/pattern.png", out_img)


def plot_rg_before_after_in_ycbcr_conversion(rgb1=[192, 192, 0],
                                             rgb2=[0, 192, 192]):
    rgb1_2 = analyze_video_level(rgb1)
    rgb2_2 = analyze_video_level(rgb2)

    g1 = rgb1_2[0, :, 0]
    g2 = rgb2_2[0, :, 1]

    print(g1)
    print(g2)

    label_1 = "ref of the rgb={}".format(rgb1)
    label_2 = "green of the rgb={}".format(rgb2)
    ref_x = np.arange(len(g1))
    ref_y = np.ones_like(ref_x) * (rgb1[0] + rgb2[1]) / 2

    ax1 = pu.plot_1_graph()
    ax1.plot(ref_x, g1, '-o', label=label_1)
    ax1.plot(ref_x, g2, '-o', label=label_2)
    ax1.plot(ref_x, ref_y, label="ref_value")
    plt.legend(loc="upper left")
    plt.show()


def test_func():
    # clip_over_512_level()
    # analyze_video_level(rgb=[192, 192, 0])
    # make_mono_clip_pattern()
    make_test_test_pattern()
    make_wrong_pattern_img()
    concatenate_all_images()
    # plot_rg_before_after_in_ycbcr_conversion(rgb1=[192, 0, 0],
    #                                          rgb2=[0, 192, 192])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_func()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RGB --> YCbCr --> RGB 変換で係数間違えを犯した場合の
情報欠落について調査する。
"""

import os
import numpy as np
import cv2
from colour import RGB_to_XYZ, XYZ_to_Lab
from colour import delta_E
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
BASE_SRC_8BIT_PATTERN = "./img/src_8bit.tiff"
# BASE_SRC_8BIT_PATTERN = "./img/src_8bit_trim2.tif"

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


def convert_rgb_to_ycbcr_to_rgb(src_img, src_coef, dst_coef):
    """
    RGB --> YCbCr --> RGB 変換。
    RGB, YCbCr は整数型。なので量子化誤差＋αは確実に発生。
    """
    ycbcr_img = ryr.convert_to_ycbcr(src_img, src_coef, bit_depth=8,
                                     limited_range=True)
    dst_img = ryr.convert_to_rgb(ycbcr_img, dst_coef, bit_depth=8,
                                 limited_range=True).astype(np.uint8)
    return dst_img


def make_wrong_ycbcr_conv_image(src_coef=BT709, dst_coef=BT2020):
    src_img = img_read(BASE_SRC_8BIT_PATTERN)
    dst_img = convert_rgb_to_ycbcr_to_rgb(src_img, src_coef, dst_coef)
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


def linear_rgb_to_cielab(rgb, gamut):
    """
    LinearなRGB値をRGB --> XYZ --> L*a*b* に変換する。
    rgb は [0:1] に正規化済みの前提ね。
    """
    illuminant_XYZ = tpg.D65_WHITE
    illuminant_RGB = tpg.D65_WHITE
    chromatic_adaptation_transform = 'CAT02'
    rgb_to_xyz_matrix = tpg.get_rgb_to_xyz_matrix(gamut)
    large_xyz = RGB_to_XYZ(rgb, illuminant_RGB, illuminant_XYZ,
                           rgb_to_xyz_matrix,
                           chromatic_adaptation_transform)

    lab = XYZ_to_Lab(large_xyz, illuminant_XYZ)

    return lab


def calc_delta_e(src_rgb, dst_rgb, method='cie2000'):
    """
    RGB値からdelta_eを計算。
    rgb値はガンマ補正がかかった8bit整数型の値とする。
    """
    src_linear = (src_rgb / 0xFF) ** 2.4
    dst_linear = (dst_rgb / 0xFF) ** 2.4
    src_lab = linear_rgb_to_cielab(src_linear, BT709)
    dst_lab = linear_rgb_to_cielab(dst_linear, BT709)
    delta = delta_E(src_lab, dst_lab, method)

    return delta


def plot_single_histgram(data, title=None, method='cie2000',
                         plot_range=[0, 20]):
    """
    単色のヒストグラムを作成する
    """
    width = 0.7
    y = ryr.make_histogram_data(data, plot_range)
    range_k = np.arange(plot_range[0], plot_range[1] + 1)
    xtick = [x for x in range(plot_range[0], plot_range[1] + 1)]
    ax1 = pu.plot_1_graph(graph_title=title,
                          graph_title_size=22,
                          xlabel="Color Difference",
                          ylabel="Frequency",
                          xtick=xtick,
                          grid=False)
    label = "delta E. method={}".format(method)
    ax1.bar(range_k, y[0], color=K_BAR_COLOR, label=label,
            width=width)
    ax1.set_yscale("log", nonposy="clip")
    plt.legend(loc='upper right')
    fname = "figures/" + title + "_" + method + ".png"
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    # plt.show()


def make_delta_e_histogram_thread_wrapper(args):
    """
    make_delta_e_histogram をスレッド化するためのラッパー。
    """
    return make_delta_e_histogram(*args)


def make_delta_e_histogram(src_coef=BT709, dst_coef=BT2020, method='cie2000',
                           plot_range=[0, 20]):
    """
    YCbCr変換係数ミス時の色差に関するヒストグラムを作成する。
    """
    x = np.arange(0, 255, 1)
    src_rgb = ryr.make_3d_grid(x)
    dst_rgb = convert_rgb_to_ycbcr_to_rgb(src_rgb, src_coef, dst_coef)
    delta = calc_delta_e(src_rgb, dst_rgb, method)
    title = "src_coef={}, dst_coef={}".format(src_coef, dst_coef)
    plot_single_histgram(delta, title=title, method=method,
                         plot_range=plot_range)


def make_delta_e_histogram_all_pattern(method='cie2000'):
    rgb_to_ycbcr_coef_list = [BT601, BT709, BT2020]
    ycbcr_to_rgb_coef_list = [BT601, BT709, BT2020]

    args_list = []
    for src_coef in rgb_to_ycbcr_coef_list:
        for dst_coef in ycbcr_to_rgb_coef_list:
            args_list.append([src_coef, dst_coef, method])
            make_delta_e_histogram(src_coef, dst_coef, method='cie2000',
                                   plot_range=[0, 15])


def test_func():
    # x = (np.linspace(0, 1, 1024) * 0.5) ** (1/1)
    # print(x)
    # x2 = np.zeros(1024)
    # rgb = np.dstack((x, x, x2))
    # lab = linear_rgb_to_cielab(rgb, BT709)
    # print(lab)
    # delta = delta_E(lab[:, 800:900, :], lab[:, 900:1000, :], 'cie2000')
    # print(lab[:, 800:900, :], lab[:, 900:1000, :])
    # print(delta)

    # x = np.arange(0, 255, 4)
    # x = np.append(x, 255)
    # y = np.arange(1, 255, 4)c
    # y = np.append(y, 255)
    # print(y)
    # rgb = ryr.make_3d_grid(x)
    # rgb2 = ryr.make_3d_grid(y)
    # delta = calc_delta_e(rgb, rgb2)
    # print(delta)

    make_delta_e_histogram(src_coef=BT709, dst_coef=BT601)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # test_func()
    # calc_yuv_transform_matrix()
    # convert_16bit_tiff_to_8bit_tiff()
    # make_wrong_ycbcr_conv_image_all_pattern()
    # concatenate_all_images()
    make_delta_e_histogram_all_pattern()

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
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# original libraries
import plot_utility as pu
import test_pattern_generator2 as tpg
import rgb_yuv_rgb_transformation as ryr


BT601 = 'ITU-R BT.601'
BT709 = 'ITU-R BT.709'
BT2020 = 'ITU-R BT.2020'
BASE_SRC_16BIT_PATTERN = "./img/src_16bit.tiff"
# BASE_SRC_8BIT_PATTERN = "./img/src_8bit.tiff"
BASE_SRC_8BIT_PATTERN = "./img/src_8bit_trim.png"

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

    if img is not None:
        return img[:, :, ::-1]
    else:
        return img


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
    caption = "src={}, dst={}".format(src_coef, dst_coef)
    dst_img = add_caption_to_color_checker(dst_img, caption)
    file_name = "./img/{}_{}.png".format(src_coef, dst_coef)
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


def make_delta_e_base_chroma_diagram_all_pattern():
    """
    make_delta_e_base_chroma_diagram_each_value()を全パターン呼ぶ
    """
    rgb_to_ycbcr_coef_list = [BT601, BT709, BT2020]
    ycbcr_to_rgb_coef_list = [BT601, BT709, BT2020]

    for src_coef in rgb_to_ycbcr_coef_list:
        for dst_coef in ycbcr_to_rgb_coef_list:
            if src_coef == dst_coef:
                pass
            else:
                make_delta_e_base_chroma_diagram_each_value(src_coef, dst_coef,
                                                            method='cie2000')
                concatenate_xy_chtomaticity_iamge(src_coef, dst_coef)


def make_delta_e_base_chroma_diagram_each_value(src_coef, dst_coef,
                                                method='cie2000'):
    """
    delta E の値が大きかった色をxy色度図上にプロットしてみる。
    0～最大値まで1刻みでプロットしてみる。
    """
    x = np.arange(0, 255, 2)
    src_rgb = ryr.make_3d_grid(x)
    dst_rgb = convert_rgb_to_ycbcr_to_rgb(src_rgb, src_coef, dst_coef)
    delta = calc_delta_e(src_rgb, dst_rgb, method)

    range_end = int(np.ceil(np.max(delta)))

    for delta_e_value in range(range_end):
        target_idx = calc_target_idx(delta, delta_e_value, delta_e_value+1)
        normalized_src_rgb = src_rgb / 255
        plot_rgb = normalized_src_rgb[:, target_idx[0], :]
        # title = "src_coef_{}_dst_coef_{}_delta_E_{}-{}"
        title = "xy_png_{:02d}".format(delta_e_value)
        title = title.format(src_coef, dst_coef, delta_e_value,
                             delta_e_value + 1)
        plot_chromaticity_diagram(gamut=BT709, data=plot_rgb, title=title)


def calc_target_idx(data, low_threshold, high_threshold):
    """
    ndarrayの特定領域のidxを計算する。
    """
    return (data >= low_threshold) & (data < high_threshold)


def plot_chromaticity_diagram(gamut, data, title=None):
    xyY = ryr.rgb_to_xyY(data, gamut)
    gamut_xy, _ = tpg.get_primaries(gamut)
    cmf_xy = tpg._get_cmfs_xy()

    rate = 1.0
    ax1 = pu.plot_1_graph(fontsize=20 * rate,
                          figsize=(8 * rate, 9 * rate),
                          graph_title=title,
                          graph_title_size=16,
                          xlabel=None, ylabel=None,
                          axis_label_size=None,
                          legend_size=18 * rate,
                          xlim=(0, 0.8),
                          ylim=(0, 0.9),
                          xtick=[x * 0.1 for x in range(9)],
                          ytick=[x * 0.1 for x in range(10)],
                          xtick_size=17 * rate,
                          ytick_size=17 * rate,
                          linewidth=4 * rate,
                          minor_xtick_num=2,
                          minor_ytick_num=2)
    color = data.reshape((data.shape[0] * data.shape[1],
                          data.shape[2]))
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=2.5*rate, label=None)
    ax1.patch.set_facecolor("#F2F2F2")
    ax1.plot(gamut_xy[..., 0], gamut_xy[..., 1], c=K_BAR_COLOR,
             label="BT.709", lw=3*rate)
    ax1.scatter(xyY[..., 0], xyY[..., 1], s=2*rate, marker='o',
                c=color, edgecolors=None, linewidth=1*rate, zorder=100)
    ax1.scatter(np.array([0.3127]), np.array([0.3290]), s=150*rate, marker='x',
                c="#000000", edgecolors=None, linewidth=2.5*rate,
                zorder=101, label="D65")
    plt.legend(loc='upper right')
    file_name = './figures/xy_chromaticity_{}.png'.format(title)
    plt.savefig(file_name, bbox_inches='tight')
    # plt.show()


def concatenate_xy_chtomaticity_iamge(src_coef=BT709, dst_coef=BT601):
    """
    """
    v_buf = []
    zero_img = None
    for v_idx in range(3):
        h_buf = []
        for h_idx in range(3):
            idx = v_idx * 3 + h_idx
            title = "src_coef_{}_dst_coef_{}_delta_E_{}-{}"
            title = title.format(src_coef, dst_coef, idx, idx + 1)
            file_name = './figures/xy_chromaticity_{}.png'.format(title)
            img = img_read(file_name)
            if idx == 0:
                zero_img = np.zeros_like(img)
            if img is not None:
                h_buf.append(img)
            else:
                h_buf.append(zero_img)

        v_buf.append(np.hstack(h_buf))
    img = np.vstack(v_buf)
    file_name = "./img/all_xy_{}_{}.png".format(src_coef, dst_coef)
    img_write(file_name, img)


def plot_delta_e_shift(roop_num, ave, sigma, title=None):
    """
    YCbCr係数の誤りを何度も繰り返すと delta_E がどう変化するかプロット
    """
    x = np.arange(1, roop_num + 1)
    ax1 = pu.plot_1_graph(graph_title=title,
                          graph_title_size=None,
                          xlabel="Repeat Count",
                          ylabel="Color Difference (CIE DE2000)",
                          xtick=[x + 1 for x in range(roop_num)],
                          linewidth=2)
    ax1.errorbar(x, ave, yerr=sigma, fmt='-o', capsize=7.5, capthick=2,
                 color=B_BAR_COLOR, ecolor=K_BAR_COLOR)
    # ax1.plot(x, ave, '-o')
    plt.show()


def make_delta_e_histogram_repeatedly(src_coef=BT709, dst_coef=BT601,
                                      method='cie2000'):
    """
    誤った係数での変換を繰り返し行うと、どの程度まで劣化が進むかを実験する。
    ここでは、delta_E の推移を見守る。
    """
    delta_e_list = []
    title = "src_coef={}, dst_coef={}".format(src_coef, dst_coef)
    roop_num = 10
    x = np.arange(0, 255, 8)
    src_rgb_org = ryr.make_3d_grid(x)
    src_rgb = src_rgb_org.copy()
    for roop_idx in range(roop_num):
        dst_rgb = convert_rgb_to_ycbcr_to_rgb(src_rgb, src_coef, dst_coef)
        delta = calc_delta_e(src_rgb_org, dst_rgb, method)
        delta_e_list.append(delta.flatten())
        src_rgb = dst_rgb.copy()
    delta_e = np.array(delta_e_list)
    ave = np.average(delta_e, axis=-1)
    sigma = np.var(delta_e, axis=-1) ** 0.5
    plot_delta_e_shift(roop_num, ave, sigma, title=title)


def merge_text(img, txt_img, pos):
    """
    テキストを合成する作業の最後の部分。
    pos は テキストの (st_pos_h, st_pos_v) 。
    ## 個人的実装メモ
    今回はちゃんとアルファチャンネルを使った合成をしたかったが、
    PILは8bit, それ以外は 10～16bit により BG_COLOR に差が出るので断念。
    """
    st_pos_v = pos[1]
    ed_pos_v = pos[1] + txt_img.shape[0]
    st_pos_h = pos[0]
    ed_pos_h = pos[0] + txt_img.shape[1]

    # かなり汚い実装。0x00 で無いピクセルのインデックスを抽出し、
    # そのピクセルのみを元の画像に上書きするという処理をしている。
    text_index = txt_img > 0
    temp_img = img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h]
    temp_img[text_index] = txt_img[text_index]
    img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = temp_img

    return img


def add_caption_to_color_checker(img, text="ITU-R BT.601, ITU-R BT.2020"):
    """
    各パーツの説明テキストを合成。
    pos は テキストの (st_pos_h, st_pos_v) 。
    text_img_size = (size_h, size_v)
    """
    pos = (230, 10)
    text_img_size = (501, 39)
    font_size = 25
    # テキストイメージ作成
    text_width = text_img_size[0]
    text_height = text_img_size[1]
    fg_color = (0xFF, 0xFF, 0xFF)
    bg_coor = (0x00, 0x00, 0x00)
    txt_img = Image.new("RGB", (text_width, text_height), bg_coor)
    draw = ImageDraw.Draw(txt_img)
    font = ImageFont.truetype("./fonts/NotoSansMonoCJKjp-Regular.otf",
                              font_size)
    draw.text((0, 0), text, font=font, fill=fg_color)
    txt_img = np.uint8(np.asarray(txt_img))
    img = merge_text(img, txt_img, pos)

    # tpg.preview_image(img)

    return img


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

    # make_delta_e_histogram(src_coef=BT709, dst_coef=BT601)
    # x = np.arange(10)
    # calc_target_idx(x, 0, 2)
    # make_delta_e_base_chroma_diagram_each_value(src_coef=BT709,
    #                                             dst_coef=BT601)
    # concatenate_xy_chtomaticity_iamge(src_coef=BT709, dst_coef=BT601)
    # make_delta_e_base_chroma_diagram_all_pattern()

    # make_delta_e_histogram_repeatedly(BT709, BT2020)
    text = "ORIGINAL IMAGE"
    img = add_caption_to_color_checker(img_read("./img/src_8bit_trim.png"), text)
    img_write("./img/src_caption.png", img)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_func()
    # calc_yuv_transform_matrix()
    # convert_16bit_tiff_to_8bit_tiff()
    # make_wrong_ycbcr_conv_image_all_pattern()
    # concatenate_all_images()
    # make_delta_e_histogram_all_pattern()

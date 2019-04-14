#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RGB --> YCbCr --> RGB 変換により欠落する情報をまとめる
"""

import os
import numpy as np
from colour import RGB_to_YCbCr, YCbCr_to_RGB, RGB_to_XYZ, XYZ_to_xyY
from colour import RGB_COLOURSPACES
from colour.utilities import CaseInsensitiveMapping
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import plot_utility as pu
import test_pattern_generator2 as tpg


YCBCR_WEIGHTS = CaseInsensitiveMapping({
    'ITU-R BT.601': np.array([0.2990, 0.1140]),
    'ITU-R BT.709': np.array([0.2126, 0.0722]),
    'ITU-R BT.2020': np.array([0.2627, 0.0593])
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


def make_src_rgb_with_blue_index(blue_idx=0, bit_depth=8):
    """
    YCbCr変換前のRGB値を生成する。
    一度に全ての組み合わせを作るとPCのメモリが足りなくなるので、
    BlueのIndexに紐づくRGB値を作る。

    # example
    >>> make_src_rgb_with_blue_index(blue_idx=0, bit_depth=2)
    [[[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0],
      [1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0],
      [2, 0, 0], [2, 1, 0], [2, 2, 0], [2, 3, 0],
      [3, 0, 0], [3, 1, 0], [3, 2, 0], [3, 3, 0]]]
    """
    one_color_num = 2 ** bit_depth
    one_color_increment = np.arange(one_color_num)
    r, g = np.meshgrid(one_color_increment, one_color_increment, indexing='ij')
    b = np.ones((one_color_num, one_color_num), dtype=np.int16) * blue_idx
    rgb = np.dstack((r.flatten(), g.flatten(), b.flatten()))

    return rgb.reshape((1, rgb.shape[0] * rgb.shape[1], rgb.shape[2]))


def make_3d_grid(axis_data=np.arange(3)):
    """
    R, G, B の Grid を作る。

    # example
    >>> make_3d_grid(axis_data=np.array([0, 4, 8], dtype=np.int16))
    [[[0, 0, 0], [0, 0, 4], [0, 0, 8],
      [0, 4, 0], [0, 4, 4], [0, 4, 8],
      [0, 8, 0], [0, 8, 4], [0, 8, 8],
      [4, 0, 0], [4, 0, 4], [4, 0, 8],
      [4, 4, 0], [4, 4, 4], [4, 4, 8],
      [4, 8, 0], [4, 8, 4], [4, 8, 8],
      [8, 0, 0], [8, 0, 4], [8, 0, 8],
      [8, 4, 0], [8, 4, 4], [8, 4, 8],
      [8, 8, 0], [8, 8, 4], [8, 8, 8]]]
    """
    r, g, b = np.meshgrid(axis_data, axis_data, axis_data, indexing='ij')
    rgb = np.dstack((r.flatten(), g.flatten(), b.flatten()))

    return rgb.reshape((1, rgb.shape[0] * rgb.shape[1], rgb.shape[2]))


def convert_to_ycbcr(rgb, gamut='ITU-R BT.709', bit_depth=8,
                     limited_range=False):
    """
    RGB信号をYCbCr信号に変換する
    """
    ycbcr = RGB_to_YCbCr(rgb, K=YCBCR_WEIGHTS[gamut],
                         in_bits=bit_depth, in_int=True,
                         out_bits=bit_depth, out_legal=limited_range,
                         out_int=True)

    # 念の為オーバーフロー＆アンダーフローチェック
    ycbcr = crip_illegal_ycbcr_value(ycbcr, bit_depth=bit_depth,
                                     limited_range=limited_range)

    return ycbcr


def convert_to_rgb(ycbcr, gamut='ITU-R BT.709', bit_depth=8,
                   limited_range=False):
    """
    YCbCr信号をRGB信号に変換する
    """
    rgb = YCbCr_to_RGB(ycbcr, K=YCBCR_WEIGHTS[gamut],
                       in_bits=bit_depth, in_int=True,
                       in_legal=limited_range, out_bits=bit_depth,
                       out_int=True)

    # 念の為オーバーフロー＆アンダーフローチェック
    rgb = crip_illegal_rgb_value(rgb, bit_depth=bit_depth)

    return rgb


def crip_illegal_ycbcr_value(ycbcr, bit_depth=8, limited_range=False):
    """
    chroma成分のオーバーフロー＆アンダーフローをクリップする
    """
    min_chroma = 0
    min_luminance = 0
    if limited_range:
        max_chroma = (224 + 16) * (2 ** (bit_depth - 8))
        max_luminance = (219 + 16) * (2 ** (bit_depth - 8))
    else:
        max_chroma = (2 ** bit_depth) - 1
        max_luminance = (2 ** bit_depth) - 1

    ycbcr[..., 0] = np.clip(ycbcr[..., 0], min_luminance, max_luminance)
    ycbcr[..., 1] = np.clip(ycbcr[..., 1], min_chroma, max_chroma)
    ycbcr[..., 2] = np.clip(ycbcr[..., 2], min_chroma, max_chroma)

    return ycbcr


def crip_illegal_rgb_value(rgb, bit_depth=8):
    """
    chroma成分のオーバーフロー＆アンダーフローをクリップする
    """
    min_luminance = 0
    max_luminance = (2 ** bit_depth) - 1
    rgb = np.clip(rgb, min_luminance, max_luminance)

    return rgb


def count_equal_pairs(rgb_src, rgb_dst):
    """
    src と dst とで値が一致するペアをカウントする。
    """
    equal = np.all(rgb_src == rgb_dst, axis=-1)

    return np.sum(equal)


def test_func():
    # limited_range = False
    # bit_depth = 10
    # test_data = [[0, 0, 0], [512, 512, 512], [1023, 1023, 1023],
    #              [1023, 0, 0], [0, 1023, 0], [0, 0, 1023]]
    # test_data = [[0, 0, 0], [128, 128, 128], [255, 255, 255],
    #              [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    # test_data = np.array(test_data, dtype=np.int16)
    # ycbcr = convert_to_ycbcr(test_data, bit_depth=bit_depth,
    #                          limited_range=limited_range)
    # print(ycbcr)
    # rgb_dst = convert_to_rgb(ycbcr, bit_depth=bit_depth,
    #                          limited_range=limited_range)
    # print(rgb_dst)
    # count_equal_pairs(test_data, rgb_dst)
    sample = np.array([[-1, -2, -3], [-1, 0, 1], [-1, -2, 3]], dtype=np.int)
    diffs = calc_diff_rgb_and_abssum(sample)
    print(sample)
    print(diffs[0], diffs[1], diffs[2], diffs[3])
    print(diffs[3] > 2)


def calc_invertible_rate(gamut='ITU-R BT.709', bit_depth=8,
                         limited_range=False):
    """
    RGB --> YCbCr --> RGB が可逆変換となっている組の比率を計算する。
    スレッドを使わない版。リレファレンスコードとして利用。
    """
    each_ch_sample_num = 2 ** bit_depth
    ok_count = np.zeros(each_ch_sample_num, dtype=np.int32)
    for blue_idx in range(each_ch_sample_num):
        src_rgb = make_src_rgb_with_blue_index(blue_idx, bit_depth)
        ycbcr = convert_to_ycbcr(src_rgb, gamut=gamut, bit_depth=bit_depth,
                                 limited_range=limited_range)
        dst_rgb = convert_to_rgb(ycbcr, gamut=gamut, bit_depth=bit_depth,
                                 limited_range=limited_range)
        ok_count[blue_idx] = count_equal_pairs(src_rgb, dst_rgb)
        # print(blue_idx, ok_count[blue_idx])

    ok_sum = np.sum(ok_count)
    invertible_rate = ok_sum / (2 ** (3 * bit_depth))
    print(invertible_rate)
    return invertible_rate


def calc_invertible_rate_thread(gamut, limited_range, bit_depth,
                                blue_idx):
    """
    Threadで処理するやつ。とある blue_idx に対して、R, G を全通り調べる。
    """
    src_rgb = make_src_rgb_with_blue_index(blue_idx, bit_depth)
    ycbcr = convert_to_ycbcr(src_rgb, gamut=gamut, bit_depth=bit_depth,
                             limited_range=limited_range)
    dst_rgb = convert_to_rgb(ycbcr, gamut=gamut, bit_depth=bit_depth,
                             limited_range=limited_range)
    ok_count = count_equal_pairs(src_rgb, dst_rgb)
    return blue_idx, ok_count


def thread_wrapper(args):
    return calc_invertible_rate_thread(*args)


def calc_invertible_rate_high_speed(gamut='ITU-R BT.709', bit_depth=8,
                                    limited_range=False):
    """
    RGB --> YCbCr --> RGB が可逆変換となっている組の比率を計算する。
    10bitの計算があまりにも遅かったので多少の高速化を実施する。
    """
    each_ch_sample_num = 2 ** bit_depth
    callback = None
    with Pool(cpu_count()//2) as pool:
        args = [(gamut, limited_range, bit_depth, blue_idx)
                for blue_idx in range(each_ch_sample_num)]
        callback = pool.map(thread_wrapper, args)

    callback = np.array(callback, dtype=np.int32)
    ok_count = callback[..., 1]
    ok_sum = np.sum(ok_count)
    invertible_rate = ok_sum / (2 ** (3 * bit_depth))

    out_str = "| {gamut} | {bit_depth}bit | {signal_range} | {ok_rate} |"
    signal_range = "Narrow" if limited_range else "Full"
    print(out_str.format(gamut=gamut, bit_depth=bit_depth,
                         signal_range=signal_range, ok_rate=invertible_rate))


def make_diff_rgb(gamut, bit_depth, limited_range):
    """
    RGB --> YCbCr --> RGB 変換を行った後、src - dst をして返す。
    """
    axis_data = np.arange(2 ** bit_depth)
    src_rgb = make_3d_grid(axis_data)
    ycbcr = convert_to_ycbcr(src_rgb, gamut=gamut, bit_depth=bit_depth,
                             limited_range=limited_range)
    dst_rgb = convert_to_rgb(ycbcr, gamut=gamut, bit_depth=bit_depth,
                             limited_range=limited_range)
    diff_rgb = src_rgb - dst_rgb

    return diff_rgb


def calc_diff_rgb_and_abssum(diff_rgb):
    """
    R, G, B 単体(signed)およびRGB合計(unsigned)の誤差を計算して返す。
    """
    diff_r = diff_rgb[..., 0]
    diff_g = diff_rgb[..., 1]
    diff_b = diff_rgb[..., 2]
    diff_abs_rgb_sum = np.sum(np.absolute(diff_rgb), axis=-1)
    return diff_r, diff_g, diff_b, diff_abs_rgb_sum


def make_histogram_data(data, range):
    """

    """
    bin_min = range[0] - 0.5
    bin_max = range[1] + 0.5
    bins = int(bin_max - bin_min)
    y = np.histogram(data, range=(bin_min, bin_max), bins=bins)

    # range外にデータが存在していた場合は警告を出す
    if (np.min(data) < bin_min) or (np.max(data) > bin_max):
        print("=" * 80)
        print("Warning: Data also exists outside of the histogram.")
        print("=" * 80)

    return y


def plot_single_histgram(data, title=None):
    """
    単色のヒストグラムを作成する
    """
    plot_range = [0, 5]
    width = 0.7
    y = make_histogram_data(data, plot_range)
    range_k = np.arange(plot_range[0], plot_range[1] + 1)
    xtick = [x for x in range(plot_range[0], plot_range[1] + 1)]
    ax1 = pu.plot_1_graph(graph_title=title,
                          graph_title_size=22,
                          xlabel="Difference",
                          ylabel="Frequency",
                          xtick=xtick,
                          grid=False)
    label = "The sum of the absolute value differences of each color"
    ax1.bar(range_k, y[0], color=K_BAR_COLOR, label=label,
            width=width)
    ax1.set_yscale("log", nonposy="clip")
    plt.legend(loc='upper right', fontsize=12)
    fname = "figures/" + title + "sum.png"
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_rgb_histgram(r_data, g_data, b_data, title=None):
    """
    RGB3色のヒストグラムを作成する
    """
    plot_range = [-3, 3]
    three_color_width = 0.7
    each_width = three_color_width / 3

    # データ生成
    r = make_histogram_data(r_data, plot_range)
    g = make_histogram_data(g_data, plot_range)
    b = make_histogram_data(b_data, plot_range)

    # plot
    xtick = [x for x in range(plot_range[0], plot_range[1] + 1)]
    ax1 = pu.plot_1_graph(graph_title=title,
                          graph_title_size=22,
                          xlabel="Difference",
                          ylabel="Frequency",
                          xtick=xtick,
                          grid=False)
    range_r = np.arange(plot_range[0], plot_range[1] + 1) - each_width
    range_g = np.arange(plot_range[0], plot_range[1] + 1)
    range_b = np.arange(plot_range[0], plot_range[1] + 1) + each_width
    ax1.bar(range_r, r[0], color=R_BAR_COLOR, label="Red", width=each_width)
    ax1.bar(range_g, g[0], color=G_BAR_COLOR, label="Green", width=each_width)
    ax1.bar(range_b, b[0], color=B_BAR_COLOR, label="Blue", width=each_width)
    ax1.set_yscale("log", nonposy="clip")
    plt.legend(loc='upper left')
    fname = "figures/" + title + "three.png"
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def make_four_diff_data(gamut, bit_depth, limited_range):
    diff_rgb = make_diff_rgb(gamut, bit_depth, limited_range)
    diff_r, diff_g, diff_b, diff_rgb = calc_diff_rgb_and_abssum(diff_rgb)
    return diff_r, diff_g, diff_b, diff_rgb


def make_four_diff_histgram(diff_r, diff_g, diff_b, diff_rgb, title_str):
    plot_rgb_histgram(diff_r.flatten(), diff_g.flatten(), diff_b.flatten(),
                      title_str)
    plot_single_histgram(diff_rgb.flatten(), title_str)


def make_graph_for_consideration(gamut, bit_depth, limited_range):
    """
    考察用のデータを作る。
    """
    title_str = "Gamut={}, Bit Depth={}, Narrow Range={}"
    title_str = title_str.format(gamut, bit_depth, limited_range)
    diff_r, diff_g, diff_b, diff_rgb = make_four_diff_data(gamut, bit_depth,
                                                           limited_range)
    make_four_diff_histgram(diff_r, diff_g, diff_b, diff_rgb, title_str)
    make_chromaticity_diagram_with_bad_data(gamut, bit_depth, diff_rgb)


def rgb_to_xyY(rgb, gamut):
    linear_rgb = rgb ** 2.4
    illuminant_RGB = RGB_COLOURSPACES[gamut].whitepoint
    illuminant_XYZ = illuminant_RGB
    chromatic_adaptation_transform = 'Bradford'
    rgb_to_xyz_mtx = RGB_COLOURSPACES[gamut].RGB_to_XYZ_matrix
    large_xyz = RGB_to_XYZ(linear_rgb, illuminant_RGB, illuminant_XYZ,
                           rgb_to_xyz_mtx,
                           chromatic_adaptation_transform)
    xyY = XYZ_to_xyY(large_xyz, illuminant_XYZ)
    return xyY


def make_chromaticity_diagram_with_bad_data(gamut, bit_depth, diff_data):
    """
    誤差の大きかったデータをxy色度図にプロットしちゃうよっ！
    """
    # 3以上の誤差をはじき出した悪いIndexを抽出
    bad_idx = (diff_data.flatten() > 2)
    axis_data = np.arange(2 ** bit_depth)
    rgb_normalized = make_3d_grid(axis_data) / ((2 ** bit_depth) - 1)
    bad_rgb = rgb_normalized[:, bad_idx, :]
    plot_chromaticity_diagram(gamut, bad_rgb)


def plot_chromaticity_diagram(gamut, data):
    xyY = rgb_to_xyY(data, gamut)
    gamut_xy, _ = tpg.get_primaries(gamut)
    cmf_xy = tpg._get_cmfs_xy()

    rate = 1.0
    ax1 = pu.plot_1_graph(fontsize=20 * rate,
                          figsize=(8 * rate, 9 * rate),
                          graph_title="CIE1931 Chromaticity Diagram",
                          graph_title_size=None,
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
    ax1.scatter(xyY[..., 0], xyY[..., 1], s=7*rate, marker='o',
                c=color, edgecolors=None, linewidth=1*rate, zorder=100)
    ax1.scatter(np.array([0.3127]), np.array([0.3290]), s=200*rate, marker='x',
                c="#000000", edgecolors=None, linewidth=3*rate,
                zorder=101, label="D65")
    plt.legend(loc='upper right')
    plt.savefig('./figures/xy_chromaticity.png', bbox_inches='tight')
    plt.show()


def calc_invertible_rate_with_various_combinations():
    """
    YCbCr変換による色数の減少を幾つかのパラメータで調査。
    """
    calc_invertible_rate_high_speed(gamut="ITU-R BT.709", bit_depth=8,
                                    limited_range=False)
    calc_invertible_rate_high_speed(gamut="ITU-R BT.709", bit_depth=8,
                                    limited_range=True)
    calc_invertible_rate_high_speed(gamut="ITU-R BT.2020", bit_depth=8,
                                    limited_range=False)
    calc_invertible_rate_high_speed(gamut="ITU-R BT.2020", bit_depth=8,
                                    limited_range=True)
    calc_invertible_rate_high_speed(gamut="ITU-R BT.709", bit_depth=10,
                                    limited_range=False)
    calc_invertible_rate_high_speed(gamut="ITU-R BT.709", bit_depth=10,
                                    limited_range=True)
    calc_invertible_rate_high_speed(gamut="ITU-R BT.2020", bit_depth=10,
                                    limited_range=False)
    calc_invertible_rate_high_speed(gamut="ITU-R BT.2020", bit_depth=10,
                                    limited_range=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # test_func()
    calc_invertible_rate_with_various_combinations()
    make_graph_for_consideration(gamut="ITU-R BT.709", bit_depth=8,
                                 limited_range=False)

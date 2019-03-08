#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RGB --> YCbCr --> RGB 変換により欠落する情報をまとめる
"""

import os
import numpy as np
from colour import RGB_to_YCbCr, YCbCr_to_RGB
from colour.utilities import CaseInsensitiveMapping
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import plot_utility as pu


YCBCR_WEIGHTS = CaseInsensitiveMapping({
    'ITU-R BT.601': np.array([0.2990, 0.1140]),
    'ITU-R BT.709': np.array([0.2126, 0.0722]),
    'ITU-R BT.2020': np.array([0.2627, 0.0593])
})


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
    limited_range = False
    bit_depth = 10
    test_data = [[0, 0, 0], [512, 512, 512], [1023, 1023, 1023],
                 [1023, 0, 0], [0, 1023, 0], [0, 0, 1023]]
    # test_data = [[0, 0, 0], [128, 128, 128], [255, 255, 255],
    #              [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    test_data = np.array(test_data, dtype=np.int16)
    ycbcr = convert_to_ycbcr(test_data, bit_depth=bit_depth,
                             limited_range=limited_range)
    print(ycbcr)
    rgb_dst = convert_to_rgb(ycbcr, bit_depth=bit_depth,
                             limited_range=limited_range)
    print(rgb_dst)
    count_equal_pairs(test_data, rgb_dst)


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


def plot_single_histgram(data, title="r"):
    """
    ababababa
    """
    plot_range = [-5, 5]
    y = make_histogram_data(data, plot_range)
    xtick = [x for x in range(plot_range[0], plot_range[1] + 1)]
    ax1 = pu.plot_1_graph(grid=False, xtick=xtick)
    ax1.bar(np.arange(plot_range[0], plot_range[1] + 1), y[0])
    plt.show()


def make_four_diff_histgram(gamut, bit_depth, limited_range):
    diff_rgb = make_diff_rgb(gamut, bit_depth, limited_range)
    diff_r, diff_g, diff_b, diff_rgb = calc_diff_rgb_and_abssum(diff_rgb)
    plot_single_histgram(diff_r.flatten(), title="Red")
    plot_single_histgram(diff_g.flatten(), title="Green")
    plot_single_histgram(diff_b.flatten(), title="Blue")
    plot_single_histgram(diff_rgb.flatten(), title="SUM")


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
    # calc_invertible_rate_high_speed(gamut="ITU-R BT.709", bit_depth=10,
    #                                 limited_range=False)
    # calc_invertible_rate_high_speed(gamut="ITU-R BT.709", bit_depth=10,
    #                                 limited_range=True)
    # calc_invertible_rate_high_speed(gamut="ITU-R BT.2020", bit_depth=10,
    #                                 limited_range=False)
    # calc_invertible_rate_high_speed(gamut="ITU-R BT.2020", bit_depth=10,
    #                                 limited_range=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # test_func()
    # calc_invertible_rate_with_various_combinations()
    make_four_diff_histgram(gamut="ITU-R BT.709", bit_depth=8,
                            limited_range=True)
    # max_value = 10
    # x = np.array([0, 0, 1, 1, 1, 5, 5, 5, 5, 1])
    # print(x)
    # plot_single_histgram(x)

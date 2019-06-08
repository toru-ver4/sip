#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NUKE と Resolve の RRT+ODT の解析をする
"""

import os
import color_space as cs
import numpy as np
from colour import RGB_to_RGB, RGB_COLOURSPACES
import test_pattern_generator2 as tpg
import plot_utility as pu
import matplotlib.pyplot as plt


def make_primary_value_on_ap0():
    """
    各種カラースペースの RGB Primary値が
    AP0 ではどの値になるかを計算する。
    """
    cs_name_list = [cs.BT709, cs.P3_D65, cs.BT2020, cs.ACES_AP1, cs.ACES_AP0]
    dst_cs = RGB_COLOURSPACES[cs.ACES_AP0]
    src_primaries = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    src_primaries = np.array(src_primaries)

    dst_primaries = {}
    for src_cs_name in cs_name_list:
        src_cs = RGB_COLOURSPACES[src_cs_name]
        temp = RGB_to_RGB(src_primaries, src_cs, dst_cs)
        dst_primaries[src_cs_name] = temp

    return dst_primaries


def _to_10bit(x):
    """
    Examples
    --------
    >>> RGB = np.array([1.0, 1.00150067, -4.85710352e-04])
    >>> _to_10bit(RGB)
    [1023 1025 -6]
    """
    return np.int16(np.round(x * 1023))


def _np_split_with_comma(x):
    """
    Examples
    --------
    >>> RGB = np.array([1023, 0, 0], dtype=np.uint16)
    >>> print(RGB)
    [1023 0 0]
    >>> print(_np_split_with_comma(RGB))
    [1023, 0, 0]
    """
    core_str = list(map(str, x.tolist()))
    space_str = r"&nbsp;"
    space_num = [5 - len(value) for value in core_str]
    space_num = [x + 1 if x != 1 else x for x in space_num]
    return "[{:>4}, {:>4}, {:>4}]".format(
        space_str * space_num[0] + core_str[0],
        space_str * space_num[1] + core_str[1],
        space_str * space_num[2] + core_str[2] + space_str)


def _make_dst_val_rate(x):
    """
    Examples
    --------
    >>> RGB = np.array([800, 100, 40], dtype=np.uint16)
    >>> _make_dst_val_rate(RGB)
    [100.0% , 12.5%, 5.0%]
    """
    x[x < 0] = 0
    x = x / np.max(x) * 100
    return "{:.1f}%, {:.1f}%, {:.1f}%".format(
        x[0], x[1], x[2]
    )


def print_table_ap0_rgb_value(data):
    """
    make_primary_value_on_ap0() の結果を
    ブログに貼り付ける形で吐き出す
    """
    src_val = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    src_val = _to_10bit(src_val)

    print("|Color Space|src|dst|rate|")
    print("|:----|:----|:---|:---|")
    str_fmt = "|{:>14} | {} | {} | {} |"

    for cs_name in list(data.keys()):
        dst_val = _to_10bit(data[cs_name])
        for idx in range(3):
            print(str_fmt.format(
                cs_name,
                _np_split_with_comma(src_val[idx]),
                _np_split_with_comma(dst_val[idx]),
                _make_dst_val_rate(data[cs_name][idx].copy())))


def plot_ap0_ap1():
    ap0, _ = tpg.get_primaries(cs.ACES_AP0)
    ap1, _ = tpg.get_primaries(cs.ACES_AP1)

    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(9, 11),
                          graph_title="CIE1931 Chromaticity Diagram",
                          graph_title_size=None,
                          xlabel=None, ylabel=None,
                          axis_label_size=None,
                          legend_size=18,
                          xlim=(-0.1, 0.8),
                          ylim=(-0.1, 1.1),
                          xtick=[x * 0.1 - 0.1 for x in range(10)],
                          ytick=[x * 0.1 - 0.1 for x in range(12)],
                          xtick_size=17,
                          ytick_size=17,
                          linewidth=2,
                          minor_xtick_num=2,
                          minor_ytick_num=2)
    ax1.plot(ap0[..., 0], ap0[..., 1], label="AP0")
    ax1.plot(ap1[..., 0], ap1[..., 1], label="AP1")
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # data = make_primary_value_on_ap0()
    # print_table_ap0_rgb_value(data)
    # RGB = np.array([1023, 100, 0], dtype=np.uint16)
    # print(RGB)
    # print(_np_split_with_comma(RGB))
    plot_ap0_ap1()

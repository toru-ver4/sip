#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NUKE と Resolve の RRT+ODT の解析をする
"""

import os
import color_space as cs
import numpy as np
from colour import RGB_to_RGB, XYZ_to_RGB, xy_to_XYZ, RGB_COLOURSPACES
from colour.colorimetry import ILLUMINANTS
from sympy import Symbol
from subprocess import run
from subprocess import TimeoutExpired
import OpenImageIO as oiio

# original libraty
import transfer_functions as tf
import plot_utility as pu
import matplotlib.pyplot as plt
import test_pattern_generator2 as tpg
import TyImageIO as tyio


RGB_COLOUR_LIST = ["#FF4800", "#03AF7A", "#005AFF"]
CMFS_NAME = 'CIE 1931 2 Degree Standard Observer'
D65_WHITE = ILLUMINANTS[CMFS_NAME]['D65']


def make_primary_value_on_ap0(oetf=tf.GAMMA24):
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
        chromatic_acaptation = "XYZ Scaling"
        temp = RGB_to_RGB(src_primaries, src_cs, dst_cs, chromatic_acaptation)
        if src_cs_name == cs.ACES_AP0:
            cs_name = "ACES_AP0"
        elif src_cs_name == cs.ACES_AP1:
            cs_name = "ACES_AP1"
        else:
            cs_name = src_cs_name
        temp = np.clip(temp, 0.0, 1.0)
        dst_primaries[cs_name] = tf.oetf(temp, oetf)

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
    # space_str = r"&nbsp;"
    space_str = " "
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


def plot_rgb_stacked_bar_graph(data):
    """
    Examples
    --------
    >>> data = {"ACES AP0": np.array([[100, 0, 0], [0, 100, 0], [0, 0, 100]]),
                "BT.709": np.array([[70, 20, 10], [10, 70, 20], [20, 10, 70]]),
                "BT.2020": np.array([[85, 10, 5], [5, 85, 10], [10, 5, 85]])}
    >>> plot_rgb_stacked_bar_graph(data)
    """
    x_val = np.arange(len(data)) + 1
    x_offset = [-0.25, 0.0, 0.25]
    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(12, 9),
                          graph_title="Color Space Conversion to ACES AP0",
                          xlabel="Source Gamut",
                          ylabel="Percentage of RGB values [%]",
                          ylim=(0, 105))
    for gg, gamut in enumerate(data):  # 色域のループ
        value = data[gamut]
        value[value < 0] = 0
        value[value > 1] = 1
        normalize_val = np.sum(value, axis=-1) / 100
        value = value / normalize_val.reshape(value.shape[0], 1)
        for ii in range(3):  # 原色が3種類ある、のループ
            x = x_val[gg] + x_offset[ii]
            bottom = 0
            for jj in range(3):  # R, G, B の各要素のループ
                y = value[ii][jj]
                color = RGB_COLOUR_LIST[jj]
                ax1.bar(x, y, bottom=bottom, width=0.2, color=color)
                bottom += y
    plt.xticks(x_val, list(data.keys()))
    plt.savefig('stacked_ap0.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def def_line(pos1=(2, 4), pos2=(10, 10)):
    """
    Examples
    --------
    >>> x, y = def_line(pos1=(2, 4), pos2=(10, 10))
    >>> y.subs({x: 2}).evalf()
    4.00000000000000
    """
    x = Symbol('x')
    y = (pos2[1] - pos1[1]) / (pos2[0] - pos1[0]) * (x - pos1[0]) + pos1[1]

    return x, y


def calc_xy_from_white_to_primary_n_step(name=cs.BT709, step=5, color='green'):
    """
    Examples
    --------
    >>> calc_xy_from_white_to_primary_n_step(name=cs.BT709,
                                             step=5, color='green')
    [[0.31269999999999998, 0.329000000000000],
     [0.30952499999999999, 0.396750000000000],
     [0.30635000000000001, 0.464499999999999],
     [0.30317499999999997, 0.532250000000000],
     [0.29999999999999999, 0.600000000000000]]
    """
    pos1_temp = D65_WHITE
    if color == 'red':
        pos2_temp = tpg.get_primaries(name)[0][0]
    elif color == 'green':
        pos2_temp = tpg.get_primaries(name)[0][1]
    else:
        pos2_temp = tpg.get_primaries(name)[0][2]
    if pos1_temp[1] < pos2_temp[1]:
        pos1 = pos1_temp
        pos2 = pos2_temp
    else:
        pos1 = pos2_temp
        pos2 = pos1_temp
    x, y = def_line(pos1, pos2)
    z = np.linspace(pos1[0], pos2[0], step)

    val = [[w, y.subs({x: w}).evalf()] for w in z]

    return np.array(val)


def get_normalize_rgb_value_from_small_xy(small_xy, name=cs.BT709):
    """
    Examples
    --------
    >>> xy = calc_xy_from_white_to_primary_n_step(name=cs.BT709,
                                                  step=5, color='green')
    >>> print(xy)
    [[0.3127, 0.3290], [0.3095, 0.3968], [0.3064, 0.4645],
     [0.3032, 0.5323], [0.3000, 0.6000]]
    >>> get_normalize_rgb_value_from_small_xy(xy)
    [[  1.00000000e+00   1.00000000e+00   1.00000000e+00]
     [  5.40536717e-01   1.00000000e+00   5.40536717e-01]
     [  2.81687026e-01   1.00000000e+00   2.81687026e-01]
     [  1.15605362e-01   1.00000000e+00   1.15605362e-01]
     [  1.58799347e-16   1.00000000e+00   4.73916800e-16]]
    """
    large_xyz = xy_to_XYZ(small_xy)
    xyz_to_rgb_mtx = RGB_COLOURSPACES[name].XYZ_to_RGB_matrix
    rgb_linear = XYZ_to_RGB(large_xyz, D65_WHITE, D65_WHITE, xyz_to_rgb_mtx)
    normalize_val = np.max(rgb_linear, axis=-1)
    rgb_linear /= normalize_val.reshape((rgb_linear.shape[0], 1))

    return rgb_linear


def get_from_white_to_primary_rgb_value(primary_color='green', step=5,
                                        name=cs.BT709, oetf_name=tf.GAMMA24):
    """
    >>> get_from_white_to_primary_rgb_value(
    >>>     'green', step=5, name=cs.BT709, oetf_name=tf.GAMMA24)
    [[1023 1023 1023]
     [ 792 1023  792]
     [ 603 1023  603]
     [ 416 1023  416]
     [   0 1023    0]]
    """
    xy = calc_xy_from_white_to_primary_n_step(name=name, step=step,
                                              color=primary_color)
    linear_rgb = get_normalize_rgb_value_from_small_xy(xy)
    rgb = np.int16(np.round(tf.oetf(linear_rgb, oetf_name) * 1023))

    return rgb


def plot_from_white_to_primary_rgb_value(primary_color='green', step=5,
                                         name=cs.BT709, oetf_name=tf.GAMMA24):
    xy = calc_xy_from_white_to_primary_n_step(name=name, step=step,
                                              color=primary_color)
    linear_rgb = get_normalize_rgb_value_from_small_xy(xy, cs.ACES_AP1)
    rgb = tf.oetf(linear_rgb, oetf_name)

    plot_chromaticity_diagram(rate=480/755.0*2,
                              xmin=-0.1, xmax=0.8, ymin=-0.1, ymax=1.05,
                              test_scatter=(xy, rgb),
                              white_point=D65_WHITE)


def plot_chromaticity_diagram(
        rate=480/755.0*2, xmin=0.0, xmax=0.8, ymin=0.0, ymax=0.9, **kwargs):
    """
    >>> xy = calc_xy_from_white_to_primary_n_step(name=name, step=step,
    >>>                                           color=primary_color)
    >>> linear_rgb = get_normalize_rgb_value_from_small_xy(xy, cs.ACES_AP1)
    >>> rgb = tf.oetf(linear_rgb, oetf_name)
    >>> plot_chromaticity_diagram(rate=480/755.0*2,
    >>>                           xmin=-0.1, xmax=0.8, ymin=-0.1, ymax=1.05,
    >>>                           test_scatter=(xy, rgb),
    >>>                           white_point=D65_WHITE)
    """
    # キーワード引数の初期値設定
    # ------------------------------------
    test_scatter = kwargs.get('test_scatter', None)
    white_point = kwargs.get('white_point', None)

    # プロット用データ準備
    # ---------------------------------
    xy_image = tpg.get_chromaticity_image(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    cmf_xy = tpg._get_cmfs_xy()

    bt709_gamut, _ = tpg.get_primaries(name=cs.BT709)
    bt2020_gamut, _ = tpg.get_primaries(name=cs.BT2020)
    dci_p3_gamut, _ = tpg.get_primaries(name=cs.P3_D65)
    ap0_gamut, _ = tpg.get_primaries(name=cs.ACES_AP0)
    ap1_gamut, _ = tpg.get_primaries(name=cs.ACES_AP1)
    xlim = (min(0, xmin), max(0.8, xmax))
    ylim = (min(0, ymin), max(0.9, ymax))

    ax1 = pu.plot_1_graph(fontsize=20 * rate,
                          figsize=((xmax - xmin) * 10 * rate,
                                   (ymax - ymin) * 10 * rate),
                          graph_title="CIE1931 Chromaticity Diagram",
                          graph_title_size=None,
                          xlabel=None, ylabel=None,
                          axis_label_size=None,
                          legend_size=18 * rate,
                          xlim=xlim, ylim=ylim,
                          xtick=[x * 0.1 + xmin for x in
                                 range(int((xlim[1] - xlim[0])/0.1) + 1)],
                          ytick=[x * 0.1 + ymin for x in
                                 range(int((ylim[1] - ylim[0])/0.1) + 1)],
                          xtick_size=17 * rate,
                          ytick_size=17 * rate,
                          linewidth=4 * rate,
                          minor_xtick_num=2,
                          minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3.5*rate, label=None)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1],
             c=tpg.UNIVERSAL_COLOR_LIST[0], label="BT.709", lw=2.75*rate)
    ax1.plot(bt2020_gamut[:, 0], bt2020_gamut[:, 1],
             c=tpg.UNIVERSAL_COLOR_LIST[1], label="BT.2020", lw=2.75*rate)
    ax1.plot(dci_p3_gamut[:, 0], dci_p3_gamut[:, 1],
             c=tpg.UNIVERSAL_COLOR_LIST[2], label="DCI-P3", lw=2.75*rate)
    ax1.plot(ap1_gamut[:, 0], ap1_gamut[:, 1],
             c=tpg.UNIVERSAL_COLOR_LIST[3], label="ACES AP1", lw=2.75*rate)
    ax1.plot(ap0_gamut[:, 0], ap0_gamut[:, 1],
             c=tpg.UNIVERSAL_COLOR_LIST[4], label="ACES AP0", lw=2.75*rate)
    if test_scatter is not None:
        xy, rgb = test_scatter
        ax1.scatter(xy[..., 0], xy[..., 1], s=300*rate, marker='s', c=rgb,
                    edgecolors='#404040', linewidth=2*rate)
    if white_point is not None:
        ax1.plot(white_point[0], white_point[1], 'kx', label="D65",
                 markersize=10*rate, markeredgewidth=2.0*rate)
    for idx in range(test_scatter[0].shape[0]):
        text = "No.{}".format(idx)
        xy = (test_scatter[0][idx][0]+0.01, test_scatter[0][idx][1])
        xy_text = (xy[0] + 0.35, xy[1] + 0.1)
        ax1.annotate(text, xy=xy, xycoords='data',
                     xytext=xy_text, textcoords='data',
                     ha='left', va='bottom',
                     arrowprops=dict(facecolor='#333333', shrink=0.0))

    ax1.imshow(xy_image, extent=(xmin, xmax, ymin, ymax))
    plt.legend(loc='upper right')
    plt.savefig('temp_fig.png', bbox_inches='tight')
    plt.show()


def plot_from_white_to_primary_rgb_value_with_bar(
        primary_color='green', step=5, name=cs.BT709, oetf_name=tf.GAMMA24):
    rgb = get_from_white_to_primary_rgb_value(
        'green', step=5, name=cs.BT709, oetf_name=tf.GAMMA24)
    normalize_coef = np.sum(rgb, axis=-1).reshape((rgb.shape[0], 1))
    rgb_normalized = rgb / normalize_coef * 100

    x_caption = ["No.{}".format(x) for x in range(rgb.shape[0])]
    x_val = np.arange(rgb.shape[0])

    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(9, 9),
                          graph_title="色とRGBの割合の関係",
                          xlabel=None,
                          ylabel="Percentage of RGB values [%]",
                          ylim=(0, 105))
    for no_idx in range(rgb.shape[0]):
        bottom = 0
        for c_idx in range(3):
            x = no_idx
            y = rgb_normalized[no_idx][c_idx]
            color = RGB_COLOUR_LIST[c_idx]
            ax1.bar(x, y, bottom=bottom, width=0.7, color=color)
            bottom += y
    plt.xticks(x_val, x_caption)
    plt.savefig('stacked_graph.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_converted_primaries_with_bar():
    """
    積み上げ棒グラフでAP0に色域変換したRGB値を表示する。
    """
    dst_primaries = make_primary_value_on_ap0()
    plot_rgb_stacked_bar_graph(dst_primaries)


def make_dst_name(src_name, suffix_list):
    """
    Examples
    --------
    >>> src_name = "./src_709_gamut.exr"
    >>> suffix_list = ["./ctl/rrt/RRT.ctl",
                       "./ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl"]
    >>> make_dst_name(src_name, suffix_list)
    ./src_709_gamut_RRT_ODT.Academy.sRGB_100nits_dim.exr
    """
    src_root, src_ext = os.path.splitext(src_name)
    dst_ext = ".tiff"
    suffix_bare_list = [os.path.basename(os.path.splitext(x)[0])
                        for x in suffix_list]
    out_name = src_root + "_" + "_".join(suffix_bare_list) + dst_ext

    return out_name


def apply_ctl_to_exr_image(img_list, ctl_list):
    """
    Examples
    --------
    >>> img_list = ["./src_709_gamut.exr", "./src_2020_gamut.exr",
                    "./src_ap1.exr", "./src_ap0.exr"]
    >>> ctl_list = ["./ctl/rrt/RRT.ctl",
                    "./ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl"]
    >>> out_img_name_list = apply_ctl_to_exr_image(img_list, ctl_list)
    >>> print(out_img_name_list)
    ['./src_709_gamut_RRT_ODT.Academy.sRGB_100nits_dim.tiff',
     './src_2020_gamut_RRT_ODT.Academy.sRGB_100nits_dim.tiff',
     './src_ap1_RRT_ODT.Academy.sRGB_100nits_dim.tiff',
     './src_ap0_RRT_ODT.Academy.sRGB_100nits_dim.tiff']
    """
    cmd_base = "ctlrender "
    ctl_ops = ["-ctl {}".format(x) for x in ctl_list]
    format_ops = "-format tiff16"
    cmd_base += " ".join(ctl_ops) + " " + format_ops
    cmd_list = ["{} {} {}".format(cmd_base, src, make_dst_name(src, ctl_list))
                for src in img_list]
    for cmd in cmd_list:
        print(cmd)
        # run(cmd.split(" "))

    return [make_dst_name(src, ctl_list) for src in img_list]


def exr_file_read(fname):
    reader = tyio.TyReader(fname)
    return reader.read()


def exr_file_write(img, fname):
    writer = tyio.TyWriter(img, fname)
    writer.write(out_img_type_desc=oiio.FLOAT)


def gamut_convert_linear_data(src_img, src_cs_name, dst_cs_name):
    """
    Examples
    --------
    >>> src_img = exr_file_read("src_bt709.exr")
    >>> dst_img = gamut_convert_linear_data(src_img,
                                            'ITU-R BT.709', 'ACES2065-1')
    >>> exr_file_write(dst_img, "src_bt709_to_ap0.exr")
    """
    chromatic_adaptation = 'XYZ Scaling'
    src_cs = RGB_COLOURSPACES[src_cs_name]
    dst_cs = RGB_COLOURSPACES[dst_cs_name]
    dst_img = RGB_to_RGB(src_img, src_cs, dst_cs, chromatic_adaptation)
    return dst_img


def file_list_cs_convert(
        src_file_list, dst_file_list, src_cs_list, dst_cs_list):
    """
    Examples
    --------
    >>> src_file_list = ["src_bt709.exr", "src_bt2020.exr"]
    >>> dst_file_list = ["src_bt709_to_ap0.exr", "src_bt2020_to_ap0.exr"]
    >>> src_cs_list = ['ITU-R BT.709', 'ITU-R BT.2020']
    >>> dst_cs_list = ['ACES2065-1', 'ACES2065-1']
    >>> file_list_cs_convert(src_file_list, dst_file_list,
                             src_cs_list, dst_cs_list)
    """
    for idx, src_file in enumerate(src_file_list):
        src_img = exr_file_read(src_file)
        src_cs = src_cs_list[idx]
        dst_cs = dst_cs_list[idx]
        dst_file = dst_file_list[idx]
        dst_img = gamut_convert_linear_data(src_img, src_cs, dst_cs)
        exr_file_write(dst_img, dst_file)


def make_to_ap0_file_name(src_file_list):
    """
    Examples
    --------
    >>> src_file_list = ["src_bt709.exr", "src_bt2020.exr"]
    >>> make_to_ap0_file_name(src_file_list)
    ["src_bt709_to_ap0.exr", "src_bt2020_to_ap0.exr"]
    """
    suffix = "_to_ap0.exr"
    return [os.path.splitext(name)[0] + suffix for name in src_file_list]


def make_rrt_src_exr_files():
    src_file_list = ["src_bt709.exr", "src_p3.exr",
                     "src_bt2020.exr", "src_ap0.exr"]
    dst_file_list = make_to_ap0_file_name(src_file_list)
    src_cs_list = [cs.BT709, cs.P3_D65, cs.BT2020, cs.ACES_AP0]
    dst_cs_list = [cs.ACES_AP0 for x in range(len(src_file_list))]
    file_list_cs_convert(
        src_file_list, dst_file_list, src_cs_list, dst_cs_list)


def experiment_func():
    # data = make_primary_value_on_ap0()
    # print_table_ap0_rgb_value(data)
    # RGB = np.array([1023, 100, 0], dtype=np.uint16)
    # print(RGB)
    # print(_np_split_with_comma(RGB))
    # plot_ap0_ap1()
    # tpg.plot_chromaticity_diagram(xmin=-0.1, xmax=0.8, ymin=-0.1, ymax=1.05)
    # data = {"ACES AP0": np.array([[100, 0, 0], [0, 100, 0], [0, 0, 100]]),
    #         "BT.709": np.array([[70, 20, 10], [10, 70, 20], [20, 10, 70]]),
    #         "BT.2020": np.array([[85, 10, 5], [5, 85, 10], [10, 5, 85]])}
    # plot_rgb_stacked_bar_graph(data)
    # rgb = get_from_white_to_primary_rgb_value(
    #     'green', step=5, name=cs.BT709, oetf_name=tf.GAMMA24)
    # print(rgb)
    # plot_from_white_to_primary_rgb_value(
    #     'green', step=5, name=cs.BT709, oetf_name=tf.GAMMA24)
    # plot_from_white_to_primary_rgb_value_with_bar(
    #     primary_color='green', step=5, name=cs.BT709, oetf_name=tf.GAMMA24)
    # plot_converted_primaries_with_bar()
    # src_name = "./src_709_gamut.exr"
    # suffix_list = ["./ctl/rrt/RRT.ctl",
    #                "./ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl"]
    # make_dst_name(src_name, suffix_list)
    # img_list = ["./src_709_gamut.exr", "./src_2020_gamut.exr",
    #             "./src_ap1.exr", "./src_ap0.exr"]
    # ctl_list = ["./ctl/rrt/RRT.ctl",
    #             "./ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl"]
    # out_img_name_list = apply_ctl_to_exr_image(img_list, ctl_list)
    # print(out_img_name_list)
    make_rrt_src_exr_files()


def main_func():
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experiment_func()

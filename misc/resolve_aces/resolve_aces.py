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
import OpenImageIO as oiio
import transfer_functions as tf

# original libraty
import transfer_functions as tf
import plot_utility as pu
import matplotlib.pyplot as plt
import test_pattern_generator2 as tpg
import TyImageIO as tyio
import lut


RGB_COLOUR_LIST = ["#FF4800", "#03AF7A", "#005AFF",
                   "#FF0000", "#00FF00", "#0000FF",
                   "#FF00FF", "#E0FF00", "#00FFFF"]
CMFS_NAME = 'CIE 1931 2 Degree Standard Observer'
D65_WHITE = ILLUMINANTS[CMFS_NAME]['D65']
OUTPUT_TRANS_P3D65_108NITS_CTL =\
    "./ctl/outputTransforms/RRTODT.Academy.P3D65_108nits_7.2nits_ST2084.ctl"
OUTPUT_TRANS_BT2020_1000NITS_CTL =\
    "./ctl/outputTransforms/RRTODT.Academy.Rec2020_1000nits_15nits_ST2084.ctl"
OUTPUT_TRANS_BT2020_4000NITS_CTL =\
    "./ctl/outputTransforms/RRTODT.Academy.Rec2020_4000nits_15nits_ST2084.ctl"


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


def make_dst_name(src_name, suffix_list, dst_ext=".tiff"):
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
    suffix_bare_list = [os.path.basename(os.path.splitext(x)[0])
                        for x in suffix_list]
    out_name = src_root + "_" + "_".join(suffix_bare_list) + dst_ext

    return out_name


def apply_ctl_to_exr_image(img_list, ctl_list, out_ext=".tiff"):
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
    cmd_base = "ctlrender -force "
    if len(ctl_list) < 2:
        ctl_ops = ["-ctl {}".format(ctl_list[0])]
    else:
        ctl_ops = ["-ctl {}".format(x) for x in ctl_list]
    if out_ext == ".tiff":
        format_ops = "-format tiff16"
    else:
        format_ops = "-format exr32"
    cmd_base += " ".join(ctl_ops) + " " + format_ops
    cmd_list = ["{} {} {}".format(cmd_base,
                                  src,
                                  make_dst_name(src, ctl_list, out_ext))
                for src in img_list]
    for cmd in cmd_list:
        print(cmd)
        os.environ['CTL_MODULE_PATH'] = "/work/src/misc/resolve_aces/ctl/lib"
        run(cmd.split(" "))

    return [make_dst_name(src, ctl_list, out_ext) for src in img_list]


def exr_file_read(fname):
    reader = tyio.TyReader(fname)
    return reader.read()


def exr_file_write(img, fname):
    """
    Examples
    --------
    >>> x = np.linspace(0, 1, 1920)
    >>> line = np.dstack((x, x, x))
    >>> img = np.vstack([line for x in range(1080)])
    >>> exr_file_write(img, "gray_ramp.exr")
    """
    writer = tyio.TyWriter(img, fname)
    writer.write(out_img_type_desc=oiio.FLOAT)


def gamut_convert_linear_data(
        src_img, src_cs_name, dst_cs_name, ca='XYZ Scaling'):
    """
    Examples
    --------
    >>> src_img = exr_file_read("src_bt709.exr")
    >>> dst_img = gamut_convert_linear_data(src_img,
                                            'ITU-R BT.709', 'ACES2065-1')
    >>> exr_file_write(dst_img, "src_bt709_to_ap0.exr")
    """
    chromatic_adaptation = ca
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
    src_file_list = ["./src_img/src_bt709.exr", "./src_img/src_p3.exr",
                     "./src_img/src_bt2020.exr", "./src_img/src_ap0.exr"]
    dst_file_list = make_to_ap0_file_name(src_file_list)
    src_cs_list = [cs.BT709, cs.P3_D65, cs.BT2020, cs.ACES_AP0]
    dst_cs_list = [cs.ACES_AP0 for x in range(len(src_file_list))]
    file_list_cs_convert(
        src_file_list, dst_file_list, src_cs_list, dst_cs_list)

    return dst_file_list


def plot_shaper_func(mid_gray=0.18, min_exposure=-6.0, max_exposure=6.0):
    ex_exposure = 1.0
    x = tpg.get_log2_x_scale(sample_num=1024, ref_val=0.18,
                             min_exposure=min_exposure-ex_exposure,
                             max_exposure=max_exposure+ex_exposure)
    y_lg2 = np.log2(x / mid_gray)
    y_logNorm = tpg.shaper_func_linear_to_log2(
        x, mid_gray=mid_gray,
        min_exposure=min_exposure, max_exposure=max_exposure)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Linear to Log2",
        graph_title_size=None,
        xlabel="Linear ",
        ylabel="lg2",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=None, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.plot(x, y_lg2, '-', label="lg2")
    plt.legend(loc='upper left')
    plt.savefig("lg2_with_linear_x.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Linear to Log2",
        graph_title_size=None,
        xlabel="Linear Value (Log Scale)",
        ylabel="lg2",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=13, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.set_xscale('log', basex=2.0)
    ax1.plot(x, y_lg2, '-', label="lg2")
    x_val = [mid_gray * (2 ** (x - 6)) for x in range(13) if x % 2 == 0]
    # x_caption = [str(x - 6) for x in range(13) if x % 2 == 0]
    x_caption = [r"$0.18 \times 2^{{{}}}$".format(x - 6)
                 for x in range(13) if x % 2 == 0]
    plt.xticks(x_val, x_caption)
    plt.legend(loc='upper left')
    plt.savefig("log2_with_log_x.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Linear to Log2",
        graph_title_size=None,
        xlabel="Linear Value (Log scale)",
        ylabel="Linear to Log2 Value",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=[0.1 * x - 0.1 for x in range(13)],
        xtick_size=13, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.set_xscale('log', basex=2.0)
    y2 = y_logNorm.copy()
    y2[y2 < 0.0] = 0.0
    ax1.plot(x, y2, '-', label="logNorm")
    x_caption = [r"$0.18 \times 2^{{{}}}$".format(x - 6)
                 for x in range(13) if x % 2 == 0]
    plt.xticks(x_val, x_caption)
    plt.legend(loc='upper left')
    plt.savefig("linear_to_log.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="Linear to Log2",
        graph_title_size=None,
        xlabel="Linear Value (Log scale)",
        ylabel="logNorm",
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=[0.1 * x - 0.1 for x in range(13)],
        xtick_size=13, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.set_xscale('log', basex=2.0)
    ax1.plot(x, y_logNorm, '-', label="logNorm")
    x_caption = [r"$0.18 \times 2^{{{}}}$".format(x - 6)
                 for x in range(13) if x % 2 == 0]
    plt.xticks(x_val, x_caption)
    plt.legend(loc='upper left')
    plt.savefig("logNorm.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()


def make_rrt_odt_3dlut(
        lut_grid_num=65,
        mid_gray=0.18, min_expoure=-10.0, max_exposure=10.0,
        ctl_list=["./ctl/rrt/RRT.ctl",
                  "./ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl"]):
    """
    Log2スケールのx軸データを作る。

    Examples
    --------
    >>> make_rrt_odt_3dlut(
    ...     mid_gray=0.18, min_expoure=-10.0, max_exposure=1.0,
    ...     ctl_list=["./ctl/rrt/RRT.ctl",
    ...               "./ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl"])
    """
    temp_exr_name = "./temp_3dlut.exr"

    # Log2 to Linear
    log_x = tpg.get_3d_grid_cube_format(lut_grid_num)
    lin_x = tpg.shaper_func_log2_to_linear(
        log_x, mid_gray=mid_gray,
        min_exposure=min_expoure, max_exposure=max_exposure)

    # save data in OpenEXR format
    exr_file_write(lin_x, temp_exr_name)

    # exec rrt+odt using ctlrender
    out_img_name_list = apply_ctl_to_exr_image(
        img_list=[temp_exr_name], ctl_list=ctl_list, out_ext=".exr")

    # load data from OpenEXR file.
    file_name = out_img_name_list[0]
    rrt_odt_img = exr_file_read(file_name)[0, :, :3]

    # save as the 3dlut file.
    fmt_str = "rrt_{}_3dlut_midg_{}_minexp_{}_maxexp_{}.spi3d"
    odt_name = os.path.basename(os.path.splitext(ctl_list[-1])[0])
    file_name = fmt_str.format(odt_name, mid_gray, min_expoure, max_exposure)
    lut.save_3dlut(rrt_odt_img, lut_grid_num, file_name)


def make_shaper_1dlut(
        sample_num=4096,
        mid_gray=0.18, min_expoure=-10.0, max_exposure=10.0):
    """
    Note
    ----
    本関数で生成するのは、Log2 to Linear の 1DLUT である。
    OCIO 上で Linear to Log2 をする場合は、Inverse オプションを
    有効化すること。

    Examples
    --------
    >>> make_shaper_1dlut(
    ...     sample_num=4096,
    ...     mid_gray=0.18, min_expoure=-10.0, max_exposure=10.0)
    """
    x = np.linspace(0, 1, sample_num)
    y = tpg.shaper_func_log2_to_linear(
        x, mid_gray=mid_gray,
        min_exposure=min_expoure, max_exposure=max_exposure)
    fmt_str = "shaper_log2_lin_1dlut_midg_{}_minexp_{}_maxexp_{}.spi1d"
    file_name = fmt_str.format(mid_gray, min_expoure, max_exposure)
    lut.save_1dlut(y, file_name)


def make_rrt_odt_1dlut_and_3dlut(
        lut_1d_sample_num=4096,
        lut_3d_grid_num=65,
        mid_gray=0.18, min_expoure=-10.0, max_exposure=10.0,
        ctl_list=["./ctl/rrt/RRT.ctl",
                  "./ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl"]):
    make_shaper_1dlut(
        sample_num=lut_1d_sample_num,
        mid_gray=mid_gray, min_expoure=min_expoure, max_exposure=max_exposure)
    make_rrt_odt_3dlut(
        lut_grid_num=lut_3d_grid_num,
        mid_gray=mid_gray, min_expoure=min_expoure, max_exposure=max_exposure,
        ctl_list=ctl_list)


def make_wrgbmyc_ramp(
        sample_num=1920, ref_val=1.0, min_exposure=-8.0, max_exposure=8.0):
    """
    3DLUT と ctlrender の結果比較用に Rampパターンを作る。
    ramp image を返す。

    Examples
    --------
    >>> img = make_wrgbmyc_ramp()
    """
    x = tpg.get_log2_x_scale(
        sample_num=sample_num, ref_val=ref_val,
        min_exposure=min_exposure, max_exposure=max_exposure)
    z = np.zeros_like(x)
    w = np.dstack([x, x, x])
    r = np.dstack([x, z, z])
    g = np.dstack([z, x, z])
    b = np.dstack([z, z, x])
    m = np.dstack([x, z, x])
    y = np.dstack([x, x, z])
    c = np.dstack([z, x, x])
    img = np.vstack([w, r, g, b, m, y, c])

    return x, img


def _plot_rdt_odt_blue_data(x, ctl_b, nuke_b, org):
    """
    3DLUT と ctlrender の比較プロット
    """
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="ctlrender vs 3DLUT",
        graph_title_size=None,
        xlabel="Linear",
        ylabel='After RRT and ODT(init)',
        axis_label_size=None,
        legend_size=17,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=14, ytick_size=None,
        linewidth=3,    
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.set_xscale('log', basex=2.0)
    ax1.set_yscale('log', basey=10.0)
    x_val = [1.0 * (2 ** (x - 8)) for x in range(17) if x % 2 == 0]
    x_caption = [r"$1.0 \times 2^{{{}}}$".format(x - 8)
                 for x in range(17) if x % 2 == 0]
    ax1.plot(x, ctl_b[..., 0], '-', color=RGB_COLOUR_LIST[0], label="CTL R")
    ax1.plot(x, ctl_b[..., 1], '-', color=RGB_COLOUR_LIST[1], label="CTL G")
    ax1.plot(x, ctl_b[..., 2], '-', color=RGB_COLOUR_LIST[2], label="CTL B")
    ax1.plot(x, nuke_b[..., 0], '--', color=RGB_COLOUR_LIST[3], label="3DLUT R")
    ax1.plot(x, nuke_b[..., 1], '--', color=RGB_COLOUR_LIST[4], label="3DLUT G")
    ax1.plot(x, nuke_b[..., 2], '--', color=RGB_COLOUR_LIST[5], label="3DLUT B")
    ax1.plot(x, org[..., 0], '--', color=RGB_COLOUR_LIST[6], label="ORG R")
    ax1.plot(x, org[..., 1], '--', color=RGB_COLOUR_LIST[7], label="ORG G")
    ax1.plot(x, org[..., 2], '--', color=RGB_COLOUR_LIST[8], label="ORG B")
    plt.xticks(x_val, x_caption)
    plt.legend(loc='upper left')
    plt.savefig("ctlrender_vs_3dlut_init.png", bbox_inches='tight',
                pad_inches=0.1)
    plt.show()


def plot_ctl_and_3dlut_result(
        ctl_list=["./ctl/rrt/RRT.ctl",
                  "./ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl"]):
    """
    ctlrender と 3DLUT の結果の差異の分析のために、
    データをプロットしてじっくり比較してみる。
    """
    # 画像データ作成。1920x7。WRGBMYCのRamp
    x, img = make_wrgbmyc_ramp(
        sample_num=1920, ref_val=1.0, min_exposure=-8.0, max_exposure=8.0)

    # BT.2020 --> AP0 への変換
    ap0_img = gamut_convert_linear_data(img, cs.BT2020, cs.ACES_AP0)

    # ファイルの保存
    ramp_fname = "./wrgbmyc_ramp.exr"
    exr_file_write(ap0_img, ramp_fname)

    # ctlrender で RRT+ODT をかける
    ctlrrender_ramp_name =\
        apply_ctl_to_exr_image([ramp_fname], ctl_list, out_ext=".exr")[0]

    # NUKEで 3DLUT の変換をやっておく
    nuke_ramp_name = "./wrgbmyc_ramp_nuke_3dlut_ap0_to_ap1.exr"
    org_name = "./wrgbmyc_ramp_nuke_3dlut_init_odt.exr"

    # ctlrender の結果を 3DLUT の結ふファイルを Read
    ctl_img = exr_file_read(ctlrrender_ramp_name)[3, :, :3]
    nuke_img = exr_file_read(nuke_ramp_name)[3, :, :3]
    # org_img = exr_file_read(org_name)[3, :, :3]
    org_img = ap0_img[3, :, :3]

    # Blue のデータをそれぞれプロット
    _plot_rdt_odt_blue_data(x, ctl_b=ctl_img, nuke_b=nuke_img, org=org_img)


def get_after_ctl_image(
        src_img_name,
        ctl_list=["./ctl/rrt/RRT.ctl",
                  "./ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl"]):
    """
    CTL実行後にファイルを開いて返す。

    Examples
    --------
    >>> ap0_img = gamut_convert_linear_data(
    ...     img, cs.P3_D65, cs.ACES_AP0, ca='CAT02')
    >>> ramp_fname = "./wrgbmyc_ramp_for_dolby_cinema.exr"
    >>> exr_file_write(ap0_img, ramp_fname)
    >>> img = get_after_ctl_image(ramp_fname, OUTPUT_TRANS_P3D65_108NITS)
    """
    ctlrrender_after_name =\
        apply_ctl_to_exr_image([src_img_name], ctl_list, out_ext=".exr")[0]
    img = exr_file_read(ctlrrender_after_name)

    return img


def _plot_comparison_between_1000nits_and_108nits(
        x, ot_108_img, ot_1000_img,
        x_min_exposure, x_max_exposure):
    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(16, 10),
        graph_title="OutputTrasform comparison",
        graph_title_size=None,
        xlabel="Linear (center is 18% gray)",
        ylabel='Luminance [nits]',
        axis_label_size=None,
        legend_size=20,
        xlim=None,
        ylim=None,
        xtick=None,
        ytick=None,
        xtick_size=16, ytick_size=None,
        linewidth=3,
        minor_xtick_num=None,
        minor_ytick_num=None)
    ax1.set_xscale('log', basex=2.0)
    ax1.set_yscale('log', basey=10.0)
    x_val_num = x_max_exposure - x_min_exposure + 1
    x_val = [0.18 * (2 ** (x + x_min_exposure))
             for x in range(x_val_num) if x % 2 == 0]
    x_caption = [r"$0.18 \times 2^{{{}}}$".format(x + x_min_exposure)
                 for x in range(x_val_num) if x % 2 == 0]
    ax1.plot(x, ot_108_img[..., 1], '-', color=RGB_COLOUR_LIST[0],
             label="Output Transform P3D65 108nits")
    ax1.plot(x, ot_1000_img[..., 1], '-', color=RGB_COLOUR_LIST[1],
             label="Output Transform BT2020 1000nits")
    plt.xticks(x_val, x_caption)
    plt.legend(loc='upper left')
    plt.savefig("comparison_108_vs_1000.png", bbox_inches='tight',
                pad_inches=0.1)
    plt.show()


def comparison_between_1000nits_and_108nits(
        min_exposure=-10.0, max_exposure=10.0):
    """
    Output Transforms の 1000nits と 108nits の結果を比較してみる
    """
    x, img = make_wrgbmyc_ramp(
        sample_num=1920, ref_val=0.18,
        min_exposure=min_exposure, max_exposure=max_exposure)

    # P3-D65 --> AP0 への変換
    ap0_img = gamut_convert_linear_data(img, cs.P3_D65, cs.ACES_AP0, ca='CAT02')
    ramp_fname = "./wrgbmyc_ramp_for_dolby_cinema.exr"
    exr_file_write(ap0_img, ramp_fname)

    # ctlrender で 108nits 変換を実行
    ctl_list = [OUTPUT_TRANS_P3D65_108NITS_CTL]
    ot_108_img = get_after_ctl_image(ramp_fname, ctl_list)[0, :, :3]

    # ctlrender で 1000nits 変換を実行
    ctl_list = [OUTPUT_TRANS_BT2020_1000NITS_CTL]
    ot_1000_img = get_after_ctl_image(ramp_fname, ctl_list)[0, :, :3]

    ot_108_img = tf.eotf_to_luminance(ot_108_img, tf.ST2084)
    ot_1000_img = tf.eotf_to_luminance(ot_1000_img, tf.ST2084)
    print(ot_108_img[960, 1])
    print(ot_1000_img[960, 1])
    _plot_comparison_between_1000nits_and_108nits(
        x, ot_108_img=ot_108_img, ot_1000_img=ot_1000_img,
        x_min_exposure=int(min_exposure), x_max_exposure=int(max_exposure))


def experiment_func():
    # print(shaper_func_linear_to_log2(x=(0.18*(2**4)), mid_gray=0.18,
    #                                  min_exposure=-6.5, max_exposure=4.0))
    # plot_shaper_func(mid_gray=0.18, min_exposure=-6.0, max_exposure=6.0)
    # make_shaper_1dlut(
    #     sample_num=4096,
    #     mid_gray=0.18, min_expoure=-10.0, max_exposure=10.0)
    # ctl_list = ["./ctl/rrt/RRT.ctl",
    #             "./ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl"]
    # make_rrt_odt_1dlut_and_3dlut(
    #     lut_1d_sample_num=65535,
    #     lut_3d_grid_num=129,
    #     mid_gray=0.18, min_expoure=-10.0, max_exposure=12.0,
    #     ctl_list=ctl_list)
    # plot_ctl_and_3dlut_result(ctl_list=ctl_list)
    comparison_between_1000nits_and_108nits()


def main_func():
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    experiment_func()

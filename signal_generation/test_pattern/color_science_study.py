#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BT.2407を実装するぞい！
あと色彩工学も勉強するぞい！
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import test_pattern_generator2 as tpg
import plot_utility as pu
import common as cmn
import colour
import sympy
import imp
imp.reload(tpg)


def lab_increment_data(sample_num=7):
    """
    ある壮大な計画に必要なデータの一部を生成する。
    CIELAB空間を斜めに切って、Chroma-Lightness平面を作り、
    そこで外壁の形をプロットしたい。それに必要なデータを作る。

    Parameters
    ----------
    sample_num : int
        sample number for each data.

    Returns
    -------
    main_data : array_like
        主で使うデータ
    sub_data : array_like
        副で使うデータ

    Example
    -------
    >>> lab_increment_data(sample_num=7)
    >>> main_data: [ 0.    0.25  0.5   0.75  1.    1.    1.    1.    1.  ]
    >>> sub_data:  [ 0.    0.    0.    0.    0.    0.25  0.5   0.75  1.  ]

    """
    if sample_num % 2 == 0:
        raise ValueError('"sample_num" must be odd number!')
    half_num = sample_num // 2 + 1
    main_data = np.ones(sample_num)
    main_data[:half_num] = np.linspace(0, 1, half_num)
    sub_data = (1 - main_data)[::-1]

    return main_data, sub_data


def judge(logic, if_true, if_false):
    if logic:
        return if_true
    else:
        return if_false


def rgbmyc_data_for_lab(sample_num=7):
    """
    ある壮大な計画に必要なデータの一部を生成する。
    CIELAB空間を斜めに切って、Chroma-Lightness平面を作り、
    そこで外壁の形をプロットしたい。それに必要なデータを作る。

    Parameters
    ----------
    sample_num : int
        sample number for each data.

    Returns
    -------
    data : array_like
        RGBMYCのLAB確認用データ。shape = sample_num x 6 x 3
    """
    base = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 0, 1), (1, 1, 0), (0, 1, 1)]
    main, sub = lab_increment_data(sample_num)
    data = []
    for element in base:
        data.append(np.dstack((judge(element[0], main, sub),
                               judge(element[1], main, sub),
                               judge(element[2], main, sub))))
    data = np.vstack(data)

    return data


def get_chroma(lab):
    return ((lab[..., 1] ** 2) + (lab[..., 2] ** 2)) ** 0.5


def get_hue(lab):
    return np.arctan(lab[..., 2]/lab[..., 1])


def plot_lab_leaf(sample_num=9):
    """
    chroma-lightness の leafをプロットする。
    RGBMYCの6パターンでプロットする。

    Parameters
    ----------
    sample_num : int
        sample number for each data.

    """

    rgb = rgbmyc_data_for_lab(sample_num)
    lab_709 = rgb_to_lab_d65(rgb=rgb, name="ITU-R BT.709")
    lab_2020 = rgb_to_lab_d65(rgb=rgb, name="ITU-R BT.2020")

    l_709 = lab_709[..., 0]
    l_2020 = lab_2020[..., 0]

    c_709 = get_chroma(lab_709)
    c_2020 = get_chroma(lab_2020)

    h_709 = get_hue(lab_709)
    h_2020 = get_hue(lab_2020)

    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title="Title",
                          graph_title_size=None,
                          xlabel="Chroma",
                          ylabel="Lightness",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=None,
                          ylim=None,
                          xtick=None,
                          ytick=None,
                          xtick_size=None, ytick_size=None,
                          linewidth=3)
    ax1.plot(c_709[0, ...], l_709[0, ...], '-o',
             c="#FF8080", label="BT.709" + " " + str(h_709[0, ...]))
    ax1.plot(c_2020[0, ...], l_2020[0, ...], '-o',
             c="#FF4040", label="BT.2020" + " " + str(h_2020[0, ...]))
    plt.legend(loc='upper right')
    plt.show()


def rgb_to_lab_d65(rgb, name="ITU-R BT.709"):

    illuminant_XYZ = tpg.D65_WHITE
    illuminant_RGB = tpg.D65_WHITE
    chromatic_adaptation_transform = 'CAT02'
    rgb_to_xyz_matrix = tpg.get_rgb_to_xyz_matrix(name)
    large_xyz = colour.RGB_to_XYZ(rgb, illuminant_RGB,
                                  illuminant_XYZ, rgb_to_xyz_matrix,
                                  chromatic_adaptation_transform)
    lab = colour.XYZ_to_Lab(large_xyz, illuminant_XYZ)

    return lab


def plot_ab_pane_of_lab(sample_num):
    """
    L*a*b*空間の a*b* 平面をプロットする。
    Hueの考え方がxyY空間と同じか確認するため。
    データは RGBMYC の6種類（RGBベース）。

    Parameters
    ----------
    sample_num : int
        sample number for each data.

    """

    rgb = rgbmyc_data_for_lab(sample_num)
    lab_709 = rgb_to_lab_d65(rgb=rgb, name="ITU-R BT.709")
    lab_2020 = rgb_to_lab_d65(rgb=rgb, name="ITU-R BT.2020")
    color_709 = ["#FF8080", "#80FF80", "#8080FF",
                 "#FF80FF", "#FFFF80", "#80FFFF"]
    color_2020 = ["#FF0000", "#00FF00", "#0000FF",
                  "#FF00FF", "#FFFF00", "#00FFFF"]

    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title="Title",
                          graph_title_size=None,
                          xlabel="a*",
                          ylabel="b*",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=None,
                          ylim=None,
                          xtick=None,
                          ytick=None,
                          xtick_size=None, ytick_size=None,
                          linewidth=3)
    for idx in range(rgb.shape[0]):
        ax1.plot(lab_709[idx, :, 1], lab_709[idx, :, 2], '-o',
                 c=color_709[idx], label="BT.709_" + str(idx))
        ax1.plot(lab_2020[idx, :, 1], lab_2020[idx, :, 2], '-o',
                 c=color_2020[idx], label="BT.2020_" + str(idx))
    plt.legend(loc='lower left')
    plt.show()


def plot_lab_color_space(name='ITU-R BT.709', grid_num=17):
    data = cmn.get_3d_grid_cube_format(grid_num)

    lab = rgb_to_lab_d65(rgb=data, name=name)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel("a*")
    ax.set_ylabel("b*")
    ax.set_zlabel("L*")
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])

    color_data = data.copy().reshape((grid_num**3, 3))

    ax.scatter(lab[..., 1], lab[..., 2], lab[..., 0], c=color_data)
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_lab_color_space('ITU-R BT.709', 33)
    # lab_increment_data(sample_num=9)
    # print(rgbmyc_data_for_lab(sample_num=5))
    # plot_lab_leaf(sample_num=101)
    plot_ab_pane_of_lab(sample_num=11)

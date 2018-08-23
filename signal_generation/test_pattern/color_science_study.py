#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BT.2407を実装するぞい！
あと色彩工学も勉強するぞい！
"""

import os
import numpy as np
import matplotlib.pyplot as plt
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


def get_rotation_matrix(degree=0):
    rad = (degree % 360) / 180.0 * np.pi
    rot = np.array([[np.cos(rad), -np.sin(rad)],
                    [np.sin(rad), np.cos(rad)]])

    return rot


def vector_rotation_2dim(vector, degree):
    rot = get_rotation_matrix(degree)
    x = vector[..., 0]
    y = vector[..., 1]

    temp_x = x * rot[0][0] + y * rot[0][1]
    temp_y = x * rot[1][0] + y * rot[1][1]

    result = np.column_stack((temp_x, temp_y))

    return result


def lab_to_rgb_d65(lab, name='ITU-R BT.2020'):

    illuminant_XYZ = tpg.D65_WHITE
    illuminant_RGB = tpg.D65_WHITE
    chromatic_adaptation_transform = 'CAT02'
    xyz_to_rgb_matrix = tpg.get_xyz_to_rgb_matrix(name)

    # Lab to XYZ
    large_xyz = colour.Lab_to_XYZ(lab, illuminant_XYZ)

    # XYZ to RGB
    rgb = colour.XYZ_to_RGB(large_xyz, illuminant_XYZ, illuminant_RGB,
                            xyz_to_rgb_matrix,
                            chromatic_adaptation_transform)

    return rgb


def try_chroma_lightness_to_rgb(samples=1024, hue=0):
    """
    L*a*b の Chroma-Lightness 平面の外枠の調査。
    RGB値として存在できない値の挙動を見る。
    """

    ab_max = 220
    l = np.ones((samples)) * 50
    chroma = np.linspace(0, 1, samples)
    a_base = np.linspace(0, ab_max, samples)
    b_base = np.zeros_like(a_base)
    ab = np.column_stack((a_base, b_base))
    ab = vector_rotation_2dim(ab, degree=hue)

    lab = np.dstack((l, ab[..., 0], ab[..., 1]))
    rgb = lab_to_rgb_d65(lab, name='ITU-R BT.2020')

    print(rgb)


def get_maximum_true_index(x):
    """
    x = [True, False, True, True, False, Flase]
    的な配列から、True となる 最大 index を求める。
    上の例だと index = 3 が求めたい値
    """
    for idx in range(x.shape[-1])[::-1]:
        if x[..., idx]:
            return idx
    raise ValueError("True index was not found")
    return 0


def get_max_lab_value(lab, name='ITU-R BT.2020'):

    rgb = lab_to_rgb_d65(lab, name)

    # RGB個別に[0:1] の範囲内か確認する
    ok_rgb = (rgb >= 0) & (rgb <= 1.0)

    # R, G, B の結果をまとめて1次元にする
    ok_val_list = ok_rgb[..., 0] & ok_rgb[..., 1] & ok_rgb[..., 2]

    # True を維持する最大の index を得る
    idx = get_maximum_true_index(ok_val_list)

    return lab[0, idx]


def get_lab_edge(hue=120):
    """
    探索により Chroma-Lightness平面をプロットするためのエッジを求める。
    とりあえず BT.2020 と BT.709 の平面を想定。

    やり方
    ------
    ```l = np.linspace(0, 100, l_samples)```
    で L* を準備する。

    各 L* に対して hue に応じた a*b* 値を算出する。

    L*a*b* を RGB に戻す。そして RGB値が [0:1] を維持している
    最大の a*b* を求める。
    """
    l_samples = 1024
    ab_samples = 1024
    ab_max = 220
    l_max = 100

    l_base = np.ones((ab_samples))  # 後で Lab にまとめるので ab_sample を指定
    a_base = np.linspace(0, ab_max, ab_samples)
    b_base = np.zeros_like(a_base)
    ab_base = np.column_stack((a_base, b_base))
    ab = vector_rotation_2dim(ab_base, degree=hue)

    lab_709 = np.zeros((l_samples, 3))
    lab_2020 = np.zeros((l_samples, 3))
    rgb = np.zeros((l_samples, 3))

    for l_idx in range(l_samples):
        l = l_base * l_max / l_samples * l_idx
        lab = np.dstack((l, ab[..., 0], ab[..., 1]))

        lab_709[l_idx] = get_max_lab_value(lab, name='ITU-R BT.709')
        lab_2020[l_idx] = get_max_lab_value(lab, name='ITU-R BT.2020')
        rgb[l_idx] = lab_to_rgb_d65(lab_709[l_idx], name='ITU-R BT.709')

    return lab_709, lab_2020, rgb


def plot_chroma_lightness_pane(lab709, lab2020, hue):
    chroma_709 = get_chroma(lab709)
    chroma_2020 = get_chroma(lab2020)
    title = "Constant Hue Angle Plane h={:d}°".format(hue)

    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title=title,
                          graph_title_size=None,
                          xlabel="Chroma",
                          ylabel="Lightness",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=[0, 220],
                          ylim=[0, 100],
                          xtick=None,
                          ytick=None,
                          xtick_size=None, ytick_size=None,
                          linewidth=3)
    ax1.plot(chroma_709, lab709[..., 0], c="#808080", label='BT.709')
    ax1.plot(chroma_2020, lab2020[..., 0], c="#000000", label='BT.2020')
    plt.legend(loc='upper right')
    plt.show()


def plot_chroma_lightness_plane_multi():
    v_num = 6
    h_hum = 2
    hue = np.linspace(0, 360, v_num * h_hum, endpoint=False)

    fig = plt.figure(figsize=(10, 14))

    for idx in range(v_num * h_hum):
        lab_709, lab_2020, rgb = get_lab_edge(hue[idx])
        chroma_709 = get_chroma(lab_709)
        chroma_2020 = get_chroma(lab_2020)
        rgb = rgb[rgb.shape[0] // 2] ** (1/2.2)

        ax1 = fig.add_subplot(v_num, h_hum, idx + 1)
        ax1.set_xlim([0, 220])
        ax1.set_ylim([0, 110])
        ax1.set_xlabel("C*")
        ax1.set_ylabel("L*")
        ax1.text(170, 90, "h={}".format(hue[idx]))
        ax1.plot(chroma_709, lab_709[..., 0], c=rgb, label='BT.709', alpha=0.5)
        ax1.plot(chroma_2020, lab_2020[..., 0], c=rgb, label='BT.2020')
    plt.legend(loc='lower right')
    plt.show()


def get_l_focal(hue=45):
    """
    hueから L_focal を得る
    """

    # まずは L_cusp を求める
    # ---------------------
    lab_709, lab_2020, rgb = get_lab_edge(hue)
    chroma_709 = get_chroma(lab_709)
    chroma_2020 = get_chroma(lab_2020)

    bt709_cusp_idx = np.argmax(chroma_709)
    bt2020_cusp_idx = np.argmax(chroma_2020)

    bt709_point = sympy.Point(chroma_709[bt709_cusp_idx],
                              lab_709[bt709_cusp_idx, 0])
    bt2020_point = sympy.Point(chroma_2020[bt2020_cusp_idx],
                               lab_2020[bt2020_cusp_idx, 0])
    chroma_line = sympy.Line(bt709_point, bt2020_point)
    lightness_line = sympy.Line(sympy.Point(0, 0), sympy.Point(0, 100))
    intersection = sympy.intersection(chroma_line, lightness_line)[0].evalf()
    l_cusp = np.array(intersection)

    # BT.2407 に従って補正
    # ---------------------

    # plot
    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title=None,
                          graph_title_size=None,
                          xlabel="Chroma",
                          ylabel="Lightness",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=[0, 220],
                          ylim=[0, 100],
                          xtick=None,
                          ytick=None,
                          xtick_size=None, ytick_size=None,
                          linewidth=3)
    ax1.plot(chroma_709, lab_709[..., 0], c="#808080", label='BT.709')
    ax1.plot(chroma_2020, lab_2020[..., 0], c="#000000", label='BT.2020')
    ax1.plot(chroma_709[bt709_cusp_idx], lab_709[bt709_cusp_idx, 0], 'or',
             markersize=10, alpha=0.5)
    ax1.plot(chroma_2020[bt2020_cusp_idx], lab_2020[bt2020_cusp_idx, 0], 'or',
             markersize=10, alpha=0.5)
    ax1.plot(l_cusp[0], l_cusp[1], 'ok', markersize=10, alpha=0.5)
    # annotation
    ax1.annotate(r'L^*_{cusp}', xy=(l_cusp[0], l_cusp[1]),
                 xytext=(l_cusp[0] + 10, l_cusp[1] + 10),
                 arrowprops=dict(facecolor='black', shrink=0.1))
    ax1.plot([chroma_2020[bt2020_cusp_idx], l_cusp[0]],
             [lab_2020[bt2020_cusp_idx, 0], l_cusp[1]], '--k', alpha=0.3)
    plt.legend(loc='upper right')
    plt.show()


def get_l_cusp(hue=0):
    lab_709, lab_2020, rgb = get_lab_edge(hue)
    chroma_709 = get_chroma(lab_709)
    chroma_2020 = get_chroma(lab_2020)

    bt709_cusp_idx = np.argmax(chroma_709)
    bt2020_cusp_idx = np.argmax(chroma_2020)

    bt709_point = sympy.Point(chroma_709[bt709_cusp_idx],
                              lab_709[bt709_cusp_idx, 0])
    bt2020_point = sympy.Point(chroma_2020[bt2020_cusp_idx],
                               lab_2020[bt2020_cusp_idx, 0])
    chroma_line = sympy.Line(bt709_point, bt2020_point)
    lightness_line = sympy.Line(sympy.Point(0, 0), sympy.Point(0, 100))
    intersection = sympy.intersection(chroma_line, lightness_line)[0].evalf()
    l_cusp = np.array(intersection)

    return l_cusp[1]


def plot_l_cusp():
    x = np.arange(0, 360, 10)
    y = np.zeros_like(x)
    for idx, hue in enumerate(x):
        y[idx] = get_l_cusp(hue)

    # plot
    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title=None,
                          graph_title_size=None,
                          xlabel="Hue",
                          ylabel="L_cusp",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=[0, 360],
                          ylim=[0, 150],
                          xtick=None,
                          ytick=None,
                          xtick_size=None, ytick_size=None,
                          linewidth=3)
    ax1.plot(x, y, label='L_cusp')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_lab_color_space('ITU-R BT.709', 33)
    # lab_increment_data(sample_num=9)
    # print(rgbmyc_data_for_lab(sample_num=5))
    # plot_lab_leaf(sample_num=101)
    # plot_ab_pane_of_lab(sample_num=11)
    # try_chroma_lightness_to_rgb(samples=5, hue=120)
    # hue = 150
    # lab709, lab2020 = get_lab_edge(hue)
    # plot_chroma_lightness_pane(lab709, lab2020, hue)
    # plot_chroma_lightness_plane_multi()
    # get_l_focal(hue=5)
    plot_l_cusp()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gamut確認用のテストパターンを作る
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import test_pattern_generator2 as tpg
import plot_utility as pu
import colour
import sympy
# from PIL import Image
from PIL import ImageCms
import imp
imp.reload(tpg)
imp.reload(ImageCms)


def _get_specific_monitor_primaries(filename="./icc_profile/gamut.icc"):
    """
    iccプロファイルからモニターのprimary情報を取得する

    Parameters
    ----------
    filename : string
        filename of the icc profile.

    Returns
    -------
    array_like
        prmaries. [[rx, ry], [gx, gy], [bx, by], [rx, ry]]

    """
    icc_profile = ImageCms.getOpenProfile(filename)
    r = icc_profile.profile.red_primary
    g = icc_profile.profile.green_primary
    b = icc_profile.profile.blue_primary
    primaries = [r[1][:-1], g[1][:-1], b[1][:-1], r[1][:-1]]

    return np.array(primaries)


def get_intersection_secondary():
    """
    BT.2020 の Secondary と D65 を結ぶ直線と
    BT.709 の Gamut が交差する点を求める
    """

    secondary, _ = tpg.get_secondaries(name='ITU-R BT.2020')
    primary, _ = tpg.get_primaries(name='ITU-R BT.709')

    white_point = sympy.Point(tpg.D65_WHITE[0], tpg.D65_WHITE[1])

    secondary_points = [sympy.Point(secondary[x][0], secondary[x][1])
                        for x in range(3)]
    primary_points = [sympy.Point(primary[x][0], primary[x][1])
                      for x in range(4)]

    secondary_lines = [sympy.Line(secondary_points[x], white_point)
                       for x in range(3)]
    primary_lines = [sympy.Line(primary_points[(x+2) % 3],
                                primary_points[(x+3) % 3])
                     for x in range(3)]

    # 交点求める。evalf() して式の評価も済ませておく
    # -------------------------------------------
    intersections = [sympy.intersection(secondary_lines[x],
                                        primary_lines[x])[0].evalf()
                     for x in range(3)]

    # 後で扱いやすいように xy の配列に変換しておく
    # -----------------------------------------
    intersections = [[intersections[x].x, intersections[x].y]
                     for x in range(3)]

    return np.array(intersections)


def get_intersection_primary():
    """
    BT.2020 の Primary と D65 を結ぶ直線と
    BT.709 の Gamut が交差する点を求める
    """

    bt2020_p, _ = tpg.get_primaries(name='ITU-R BT.2020')
    primary, _ = tpg.get_primaries(name='ITU-R BT.709')

    white_point = sympy.Point(tpg.D65_WHITE[0], tpg.D65_WHITE[1])

    bt2020_p_points = [sympy.Point(bt2020_p[x][0], bt2020_p[x][1])
                       for x in range(3)]
    primary_points = [sympy.Point(primary[x][0], primary[x][1])
                      for x in range(4)]

    bt2020_p_lines = [sympy.Line(bt2020_p_points[x], white_point)
                      for x in range(3)]

    # よく考えたら、どの線と交差するかは gamut の形で決まるんだった…マニュアルで。
    # ----------------------------------------------------------------------
    primary_lines = [sympy.Line(primary_points[2],
                                primary_points[3]),
                     sympy.Line(primary_points[1],
                                primary_points[2]),
                     sympy.Line(primary_points[1],
                                primary_points[2])]

    # 交点求める。evalf() して式の評価も済ませておく
    # -------------------------------------------
    intersections = [sympy.intersection(bt2020_p_lines[x],
                                        primary_lines[x])[0].evalf()
                     for x in range(3)]

    # 後で扱いやすいように xy の配列に変換しておく
    # -----------------------------------------
    intersections = [[intersections[x].x, intersections[x].y]
                     for x in range(3)]

    return np.array(intersections)


def _check_clip_level(src='ITU-R BT.709', dst='ITU-R BT.709'):
    """
    YUV2RGBのミスマッチが発生した場合に見えなくなるような
    テストパターンを作る。そのための事前調査をするよん☆
    """

    # 単調増加するRGB単色パターンを作成する
    # ------------------------------------
    sample_num = 1024
    max_input_value = sample_num - 1
    gradation = np.arange(sample_num)
    zeros = np.zeros_like(gradation)

    r_grad = np.dstack((gradation, zeros, zeros))
    g_grad = np.dstack((zeros, gradation, zeros))
    b_grad = np.dstack((zeros, zeros, gradation))
    img = np.vstack((r_grad, g_grad, b_grad))

    src_wights = colour.YCBCR_WEIGHTS[src]
    dst_wights = colour.YCBCR_WEIGHTS[dst]
    ycbcr = colour.RGB_to_YCbCr(img, K=src_wights, in_bits=10, in_legal=False,
                                in_int=True,
                                out_bits=10, out_legal=False, out_int=True)
    after_img = colour.YCbCr_to_RGB(ycbcr, K=dst_wights, in_bits=10,
                                    in_int=True, in_legal=False, out_bits=10,
                                    out_legal=False, out_int=True)
    after_img = colour.models.eotf_ST2084(after_img / max_input_value)
    after_img[after_img > 1000] = 1000

    title = "src coef = {:s}, dst coef = {:s}".format(src, dst)
    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title=title,
                          graph_title_size=None,
                          xlabel="Input Video Level",
                          ylabel="Output Video Level (ST2084)",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=[720, 820],
                          ylim=[600, 1050],
                          # xtick=[0, 256, 512, 768, 1000, 1023],
                          ytick=None,
                          xtick_size=None, ytick_size=None,
                          linewidth=3,
                          minor_xtick_num=None,
                          minor_ytick_num=None)
    # ax1.plot(gradation, img[0, :, 0], '-r', alpha=0.5, label="red")
    # ax1.plot(gradation, img[1, :, 1], '-g', alpha=0.5, label="green")
    # ax1.plot(gradation, img[2, :, 2], '-b', alpha=0.5, label="blue")
    ax1.plot(gradation, after_img[0, :, 0], '-or', label="red")
    ax1.plot(gradation, after_img[1, :, 1], '-og', label="green")
    ax1.plot(gradation, after_img[2, :, 2], '-ob', label="blue")
    plt.legend(loc='upper left')
    plt.show()


def _get_interpolated_xy(st, ed, sample_num):
    """
    テストパターンを作るために、xy色度図を等間隔に分割する

    Parameters
    ----------
    st : array_like
        start position on the xy chromaticity diagram.
    ed : array_like
        end position on the xy chromaticity diagram.
    sample_num : integer
        division number

    Returns
    -------
    array_like
        interpolated xy.

    """
    if st.ndim != 1 or ed.ndim != 1:
        raise ValueError("dimmention of the input data is invalid.")

    x = np.linspace(float(st[0]), float(ed[0]), sample_num)
    y = np.linspace(float(st[1]), float(ed[1]), sample_num)

    return np.dstack((x, y)).reshape((sample_num, 2))


def _get_test_scatter_data(sample_num=6):

    color_space_name = 'ITU-R BT.2020'
    primaries, primary_rgb = tpg.get_primaries(color_space_name)
    secondaries, secondary_rgb = tpg.get_secondaries(color_space_name)

    primary_intersections = get_intersection_primary()
    secondary_intersections = get_intersection_secondary()

    ed_xy = np.vstack((primaries[:3, :], secondaries))
    st_xy = np.vstack((primary_intersections, secondary_intersections))
    patch_xy = [_get_interpolated_xy(st_xy[idx], ed_xy[idx], sample_num)
                for idx in range(st_xy.shape[0])]

    patch_xy = np.array(patch_xy)
    rgb = tpg.xy_to_rgb(patch_xy, 'ITU-R BT.2020')
    rgb = rgb ** (1/2.2)
    rgb = rgb.reshape((rgb.shape[0] * rgb.shape[1], rgb.shape[2]))

    # return xy, rgb.reshape((rgb.shape[0] * rgb.shape[1], 3))
    return patch_xy, rgb


def _gen_ycbcr_ng_combination_checker():
    # ok_low_level = 186  # 744(10bit)
    # ok_high_level = 193  # 772(10bit)

    # ng_low_level = 193  # 772(10bit)
    # ng_high_level = 198  # 792(10bit)
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # _check_clip_level(src='ITU-R BT.709', dst='ITU-R BT.601')
    # _check_clip_level(src='ITU-R BT.709', dst='ITU-R BT.2020')
    # _check_clip_level(src='ITU-R BT.2020', dst='ITU-R BT.601')
    # _check_clip_level(src='ITU-R BT.2020', dst='ITU-R BT.709')

    color_space_name = "ITU-R BT.2020"
    primaries = _get_specific_monitor_primaries()
    secondaries, secondary_rgb = tpg.get_secondaries(color_space_name)
    scatter_xy, scatter_rgb = _get_test_scatter_data(sample_num=10)
    # primary_intersections = get_intersection_primary()
    # secondary_intersections = get_intersection_secondary()
    # intersections = np.append(primary_intersections,
    #                           secondary_intersections, axis=0)
    tpg.plot_chromaticity_diagram(primaries=None,
                                  test_scatter=[scatter_xy, scatter_rgb])

    # get_intersection_secondary()

    # _get_interpolated_xy()

    # print(colour.RGB_COLOURSPACES["DCI-P3"].RGB_to_XYZ_matrix)
    # mtx = cc.get_rgb_to_xyz_matrix(gamut=cc.const_dci_p3_xy,
    #                                white=cc.const_d65_large_xyz)
    # print(mtx)
    # print(cc.const_dci_white_xyz)

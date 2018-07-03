#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gamut確認用のテストパターンを作る
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import test_pattern_generator2 as tpg
import plot_utility as pu
import common as cmn
import colour
import sympy
from colour.utilities.array import dot_vector
# from PIL import Image
from PIL import ImageCms
import imp
imp.reload(tpg)
imp.reload(ImageCms)

REVISION = 1

GAMUT_PATTERN_AREA_WIDTH = (12/16.0)
GAMUT_TOP_BOTTOM_SPACE = 0.05
GAMUT_LEFT_RIGHT_SPACE = 0.02
GAMUT_PATCH_SIZE = 0.07
GAMUT_PATCH_STRIPE_NUM = 5

INFO_AREA_WIDTH = 1.0 - GAMUT_PATTERN_AREA_WIDTH


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


def get_intersection_secondary(out_side_name='ITU-R BT.2020',
                               in_side_name='ITU-R BT.709'):
    """
    BT.2020 の Secondary と D65 を結ぶ直線と
    BT.709 の Gamut が交差する点を求める
    """

    secondary, _ = tpg.get_secondaries(name=out_side_name)
    primary, _ = tpg.get_primaries(name=in_side_name)

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


def get_intersection_primary(out_side_name='ITU-R BT.2020',
                             in_side_name='ITU-R BT.709'):
    """
    BT.2020 の Primary と D65 を結ぶ直線と
    BT.709 の Gamut が交差する点を求める
    """

    bt2020_p, _ = tpg.get_primaries(name=out_side_name)
    primary, _ = tpg.get_primaries(name=in_side_name)

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


def _get_primary_secondary_large_y(name):
    """
    Primary, Secondary の Y値を求める。
    色度パターン作成時に、パッチごとにYの値が変わらないように
    後で xy --> xyY 変換時に使用する。

    Parameters
    ----------
    name : string
        target color space name.

    Returns
    -------
    array_like
        large Y value for primary and secondary.

    """

    # RGBMYC の Y値リストを得る
    _, primary = tpg.get_primaries(name)
    _, secondary = tpg.get_secondaries(name)
    data = np.vstack((primary, secondary))
    rgb_to_xyz_mtx = colour.RGB_COLOURSPACES[name].RGB_to_XYZ_matrix
    data = tpg.do_matrix(img=data, mtx=rgb_to_xyz_mtx)

    return data[..., 1]


def _get_test_scatter_data(sample_num=6):

    color_space_name = 'ITU-R BT.2020'
    inter_section_space_name = 'ITU-R BT.709'

    primaries, primary_rgb = tpg.get_primaries(color_space_name)
    secondaries, secondary_rgb = tpg.get_secondaries(color_space_name)

    primary_intersections\
        = get_intersection_primary(out_side_name=color_space_name,
                                   in_side_name=inter_section_space_name)
    secondary_intersections\
        = get_intersection_secondary(out_side_name=color_space_name,
                                     in_side_name=inter_section_space_name)

    ed_xy = np.vstack((primaries[:3, :], secondaries))
    st_xy = np.vstack((primary_intersections, secondary_intersections))
    patch_xy = [_get_interpolated_xy(st_xy[idx], ed_xy[idx], sample_num)
                for idx in range(st_xy.shape[0])]

    patch_xy = np.array(patch_xy)
    rgb = tpg.xy_to_rgb(patch_xy, color_space_name, normalize=False)
    rgb = rgb ** (1/2.2)
    rgb = rgb.reshape((rgb.shape[0] * rgb.shape[1], rgb.shape[2]))

    # return xy, rgb.reshape((rgb.shape[0] * rgb.shape[1], 3))
    return patch_xy, rgb


def _get_gamut_check_data(name):
    """
    RGB2XYZ変換が想定通り行われているか確認するための
    テストデータを作成する。
    """

    sample_num = 7
    base = (np.linspace(0, 1, sample_num) ** (2.0))[::-1]
    ones = np.ones_like(base)

    r = np.dstack((ones, base, base))
    g = np.dstack((base, ones, base))
    b = np.dstack((base, base, ones))
    rgb = np.append(np.append(r, g, axis=0), b, axis=0)

    illuminant_XYZ = tpg.D65_WHITE
    illuminant_RGB = tpg.D65_WHITE
    chromatic_adaptation_transform = 'CAT02'
    rgb_to_xyz_matrix = tpg.get_rgb_to_xyz_matrix(name)
    large_xyz = colour.models.RGB_to_XYZ(rgb, illuminant_RGB, illuminant_XYZ,
                                         rgb_to_xyz_matrix,
                                         chromatic_adaptation_transform)

    xy = colour.models.XYZ_to_xy(large_xyz, illuminant_XYZ)

    return xy, rgb.reshape((rgb.shape[0] * rgb.shape[1], 3))


def _gen_ycbcr_ng_combination_checker():
    # ok_low_level = 186  # 744(10bit)
    # ok_high_level = 193  # 772(10bit)

    # ng_low_level = 193  # 772(10bit)
    # ng_high_level = 198  # 792(10bit)
    pass


def composite_info_data(base_img, **kwargs):
    img_width = base_img.shape[1]

    fig_img = cv2.imread('temp_fig.png',
                         cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1]
    width = int(img_width * INFO_AREA_WIDTH)
    rate = width / fig_img.shape[1]
    height = int(fig_img.shape[0] * rate)

    fig_img = cv2.resize(fig_img, (width, height))
    fig_img = np.uint16(fig_img) * 0x100

    base_img[0:height, -width:, :] = fig_img


def composite_gamut_csf_pattern(base_img, patch_rgb, patch_num):
    img_width = base_img.shape[1]
    img_height = base_img.shape[0]

    width = int(img_width * GAMUT_PATTERN_AREA_WIDTH)
    height = img_height

    img = np.zeros((height, width, 3), dtype=np.uint16)

    h_num = patch_num
    v_num = 6  # RGBMYC

    left_space = int(img_width * GAMUT_LEFT_RIGHT_SPACE)
    top_space = int(img_height * GAMUT_TOP_BOTTOM_SPACE)

    patch_width = int(img_width * GAMUT_PATCH_SIZE)
    patch_height = patch_width

    ws_target_len = width - (2 * left_space) - patch_width * h_num
    width_space = cmn.equal_devision(ws_target_len, h_num - 1)
    width_space.insert(0, 0)

    hs_target_len = height - (2 * top_space) - patch_height * v_num
    height_space = cmn.equal_devision(hs_target_len, v_num - 1)
    height_space.insert(0, 0)

    v_ed = top_space
    for v_idx in range(v_num):
        v_st = v_ed + height_space[v_idx]
        v_ed = v_st + patch_height
        h_ed = left_space
        for h_idx in range(h_num):
            lv1 = patch_rgb[v_idx * h_num + h_num - 1] * 0xFFC0
            lv2 = patch_rgb[v_idx * h_num + h_idx] * 0xFFC0
            patch = tpg.get_csf_color_image(width=patch_width,
                                            height=patch_height,
                                            lv1=np.uint16(np.round(lv1)),
                                            lv2=np.uint16(np.round(lv2)),
                                            stripe_num=GAMUT_PATCH_STRIPE_NUM)
            h_st = h_ed + width_space[h_idx]
            h_ed = h_st + patch_width
            # print(v_st, v_ed, h_st, h_ed)
            # print(img[v_st:v_ed, h_st:h_ed, :].shape)
            img[v_st:v_ed, h_st:h_ed, :] = patch

    base_img[0:height, 0:width, :] = img


def gen_gamut_test_pattern(width=3840, height=2160):
    """
    BT.709 の外側の Gamut の表示具合を確認する
    Test Pattern を作成する
    """
    img = np.ones((height, width, 3), dtype=np.uint16)
    img = img * 10000

    # パッチデータ作成
    # -------------------------
    patch_num = 8
    patch_xy, patch_rgb = _get_test_scatter_data(sample_num=patch_num)
    specific_primaries = _get_specific_monitor_primaries()
    tpg.plot_chromaticity_diagram(monitor_primaries=specific_primaries,
                                  test_scatter=[patch_xy, patch_rgb])

    composite_info_data(img)
    composite_gamut_csf_pattern(img, patch_rgb, patch_num)

    tpg.preview_image(img, 'rgb')

    fname_str = "gamut_checker_rev{:02d}_{:d}x{:d}.tif"
    fname = fname_str.format(REVISION, width, height)
    cv2.imwrite(fname, img[..., ::-1])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # _check_clip_level(src='ITU-R BT.709', dst='ITU-R BT.601')
    # _check_clip_level(src='ITU-R BT.709', dst='ITU-R BT.2020')
    # _check_clip_level(src='ITU-R BT.2020', dst='ITU-R BT.601')
    # _check_clip_level(src='ITU-R BT.2020', dst='ITU-R BT.709')

    # color_space_name = "ITU-R BT.2020"
    # # color_space_name = "DCI-P3"
    # primaries = _get_specific_monitor_primaries()
    # secondaries, secondary_rgb = tpg.get_secondaries(color_space_name)
    # scatter_xy, scatter_rgb = _get_test_scatter_data(sample_num=6)
    # # scatter_xy, scatter_rgb = _get_gamut_check_data('ITU-R BT.709')

    # # primary_intersections = get_intersection_primary()
    # # secondary_intersections = get_intersection_secondary()
    # # intersections = np.append(primary_intersections,
    # #                           secondary_intersections, axis=0)
    # tpg.plot_chromaticity_diagram(primaries=None,
    #                               test_scatter=[scatter_xy, scatter_rgb])

    _normalize_with_primary_secondary(name='ITU-R BT.2020')
    # gen_gamut_test_pattern(1920, 1080)
    # gen_gamut_test_pattern(3840, 2160)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
評価用のテストパターン作成ツール集

"""

import os
import cv2
import common as cmn
import color_convert as cc
from scipy import linalg
import plot_utility as pu
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from colour.colorimetry import CMFS, ILLUMINANTS
from colour.models import XYZ_to_xy, xy_to_XYZ, XYZ_to_RGB, RGB_to_XYZ
from colour.models import xy_to_xyY, xyY_to_XYZ
from colour.utilities import normalise_maximum
from colour import models
from colour import RGB_COLOURSPACES
from scipy.spatial import Delaunay
from scipy.ndimage.filters import convolve
import common
import imp
imp.reload(pu)


CMFS_NAME = 'CIE 1931 2 Degree Standard Observer'
D65_WHITE = ILLUMINANTS[CMFS_NAME]['D65']


def preview_image(img, order='rgb', over_disp=False):
    if order == 'rgb':
        cv2.imshow('preview', img[:, :, ::-1])
    elif order == 'bgr':
        cv2.imshow('preview', img)
    else:
        raise ValueError("order parameter is invalid")

    if over_disp:
        cv2.resizeWindow('preview', )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def equal_devision(length, div_num):
    """
    # 概要
    length を div_num で分割する。
    端数が出た場合は誤差拡散法を使って上手い具合に分散させる。
    """
    base = length / div_num
    ret_array = [base for x in range(div_num)]

    # 誤差拡散法を使った辻褄合わせを適用
    # -------------------------------------------
    diff = 0
    for idx in range(div_num):
        diff += math.modf(ret_array[idx])[0]
        if diff >= 1.0:
            diff -= 1.0
            ret_array[idx] = int(math.floor(ret_array[idx]) + 1)
        else:
            ret_array[idx] = int(math.floor(ret_array[idx]))

    # 計算誤差により最終点が +1 されない場合への対処
    # -------------------------------------------
    diff = length - sum(ret_array)
    if diff != 0:
        ret_array[-1] += diff

    # 最終確認
    # -------------------------------------------
    if length != sum(ret_array):
        raise ValueError("the output of equal_division() is abnormal.")

    return ret_array


def do_matrix(img, mtx):
    """
    img に対して mtx を適用する。
    """
    base_shape = img.shape

    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    ro = r * mtx[0][0] + g * mtx[0][1] + b * mtx[0][2]
    go = r * mtx[1][0] + g * mtx[1][1] + b * mtx[1][2]
    bo = r * mtx[2][0] + g * mtx[2][1] + b * mtx[2][2]

    out_img = np.dstack((ro, go, bo)).reshape(base_shape)

    return out_img


def _get_cmfs_xy():
    """
    xy色度図のプロットのための馬蹄形の外枠のxy値を求める。

    Returns
    -------
    array_like
        xy coordinate for chromaticity diagram

    """
    # 基本パラメータ設定
    # ------------------
    cmf = CMFS.get(CMFS_NAME)
    d65_white = D65_WHITE

    # 馬蹄形のxy値を算出
    # --------------------------
    cmf_xy = XYZ_to_xy(cmf.values, d65_white)

    return cmf_xy


def get_primaries(name='ITU-R BT.2020'):
    """
    prmary color の座標を求める


    Parameters
    ----------
    name : str
        a name of the color space.

    Returns
    -------
    array_like
        prmaries. [[rx, ry], [gx, gy], [bx, by], [rx, ry]]

    """
    primaries = RGB_COLOURSPACES[name].primaries
    primaries = np.append(primaries, [primaries[0, :]], axis=0)

    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    return primaries, rgb


def xy_to_rgb(xy, name='ITU-R BT.2020', normalize='maximum', specific=None):
    """
    xy値からRGB値を算出する。
    いい感じに正規化もしておく。

    Parameters
    ----------
    xy : array_like
        xy value.
    name : string
        color space name.
    normalize : string
        normalize method. You can select 'maximum', 'specific' or None.

    Returns
    -------
    array_like
        rgb value. the value is normalized.
    """
    illuminant_XYZ = D65_WHITE
    illuminant_RGB = D65_WHITE
    chromatic_adaptation_transform = 'CAT02'
    large_xyz_to_rgb_matrix = get_xyz_to_rgb_matrix(name)
    if normalize == 'specific':
        xyY = xy_to_xyY(xy)
        xyY[..., 2] = specific
        large_xyz = xyY_to_XYZ(xyY)
    else:
        large_xyz = xy_to_XYZ(xy)

    rgb = XYZ_to_RGB(large_xyz, illuminant_XYZ, illuminant_RGB,
                     large_xyz_to_rgb_matrix,
                     chromatic_adaptation_transform)

    """
    そのままだとビデオレベルが低かったりするので、
    各ドット毎にRGB値を正規化＆最大化する。必要であれば。
    """
    if normalize == 'maximum':
        rgb = normalise_maximum(rgb, axis=-1)
    else:
        if(np.sum(rgb > 1.0) > 0):
            print("warning: over flow has occured at xy_to_rgb")
        if(np.sum(rgb < 0.0) > 0):
            print("warning: under flow has occured at xy_to_rgb")
        rgb[rgb < 0] = 0
        rgb[rgb > 1.0] = 1.0

    return rgb


def get_white_point(name):
    """
    white point を求める。CIE1931ベース。
    """
    if name != "DCI-P3":
        illuminant = RGB_COLOURSPACES[name].illuminant
        white_point = ILLUMINANTS[CMFS_NAME][illuminant]
    else:
        white_point = ILLUMINANTS[CMFS_NAME]["D65"]

    return white_point


def get_rgb_to_xyz_matrix(name):
    """
    RGB to XYZ の Matrix を求める。
    DCI-P3 で D65 の係数を返せるように内部関数化した。
    """
    if name != "DCI-P3":
        rgb_to_xyz_matrix = RGB_COLOURSPACES[name].RGB_to_XYZ_matrix
    else:
        rgb_to_xyz_matrix\
            = cc.get_rgb_to_xyz_matrix(gamut=cc.const_dci_p3_xy,
                                       white=cc.const_d65_large_xyz)

    return rgb_to_xyz_matrix


def get_xyz_to_rgb_matrix(name):
    """
    XYZ to RGB の Matrix を求める。
    DCI-P3 で D65 の係数を返せるように内部関数化した。
    """
    if name != "DCI-P3":
        xyz_to_rgb_matrix = RGB_COLOURSPACES[name].XYZ_to_RGB_matrix
    else:
        rgb_to_xyz_matrix\
            = cc.get_rgb_to_xyz_matrix(gamut=cc.const_dci_p3_xy,
                                       white=cc.const_d65_large_xyz)
        xyz_to_rgb_matrix = linalg.inv(rgb_to_xyz_matrix)

    return xyz_to_rgb_matrix


def get_secondaries(name='ITU-R BT.2020'):
    """
    secondary color の座標を求める

    Parameters
    ----------
    name : str
        a name of the color space.

    Returns
    -------
    array_like
        secondaries. the order is magenta, yellow, cyan.

    """
    secondary_rgb = np.array([[1.0, 0.0, 1.0],
                              [1.0, 1.0, 0.0],
                              [0.0, 1.0, 1.0]])
    illuminant_XYZ = D65_WHITE
    illuminant_RGB = D65_WHITE
    chromatic_adaptation_transform = 'CAT02'
    rgb_to_xyz_matrix = get_rgb_to_xyz_matrix(name)
    large_xyz = RGB_to_XYZ(secondary_rgb, illuminant_RGB,
                           illuminant_XYZ, rgb_to_xyz_matrix,
                           chromatic_adaptation_transform)

    xy = XYZ_to_xy(large_xyz, illuminant_XYZ)

    return xy, secondary_rgb.reshape((3, 3))


def plot_chromaticity_diagram(rate=480/755.0*2, **kwargs):
    # キーワード引数の初期値設定
    # ------------------------------------
    monitor_primaries = kwargs.get('monitor_primaries', None)
    secondaries = kwargs.get('secondaries', None)
    test_scatter = kwargs.get('test_scatter', None)
    intersection = kwargs.get('intersection', None)

    # プロット用データ準備
    # ---------------------------------
    xy_image = get_chromaticity_image()
    cmf_xy = _get_cmfs_xy()

    bt709_gamut, _ = get_primaries('ITU-R BT.709')
    bt2020_gamut, _ = get_primaries('ITU-R BT.2020')
    dci_p3_gamut, _ = get_primaries('DCI-P3')

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
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', lw=3.5*rate, label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', lw=3.5*rate, label=None)
    ax1.plot(bt709_gamut[:, 0], bt709_gamut[:, 1], c="#0080FF",
             label="BT.709", lw=3*rate)
    ax1.plot(bt2020_gamut[:, 0], bt2020_gamut[:, 1], c="#FFD000",
             label="BT.2020", lw=3*rate)
    ax1.plot(dci_p3_gamut[:, 0], dci_p3_gamut[:, 1], c="#C03030",
             label="DCI-P3", lw=3*rate)
    if monitor_primaries is not None:
        ax1.plot(monitor_primaries[:, 0], monitor_primaries[:, 1],
                 c="#202020", label="???", lw=3*rate)
    if secondaries is not None:
        xy, rgb = secondaries
        ax1.scatter(xy[..., 0], xy[..., 1], s=700*rate, marker='s', c=rgb,
                    edgecolors='#404000', linewidth=2*rate)
    if test_scatter is not None:
        xy, rgb = test_scatter
        ax1.scatter(xy[..., 0], xy[..., 1], s=300*rate, marker='s', c=rgb,
                    edgecolors='#404040', linewidth=2*rate)
    if intersection is not None:
        ax1.scatter(intersection[..., 0], intersection[..., 1],
                    s=300*rate, marker='s', c='#CCCCCC',
                    edgecolors='#404040', linewidth=2*rate)

    ax1.imshow(xy_image, extent=(0, 1, 0, 1))
    plt.legend(loc='upper right')
    plt.savefig('temp_fig.png', bbox_inches='tight')
    # plt.show()


def get_chromaticity_image(samples=1024, antialiasing=True, bg_color=0.9):
    """
    xy色度図の馬蹄形の画像を生成する

    Returns
    -------
    ndarray
        rgb image.
    """

    """
    色域設定。sRGBだと狭くて少し変だったのでBT.2020に設定。
    若干色が薄くなるのが難点。暇があれば改良したい。
    """
    # color_space = models.BT2020_COLOURSPACE
    color_space = models.S_GAMUT3_COLOURSPACE

    # 馬蹄形のxy値を算出
    # --------------------------
    cmf_xy = _get_cmfs_xy()

    """
    馬蹄の内外の判別をするために三角形で領域分割する(ドロネー図を作成)。
    ドロネー図を作れば後は外積計算で領域の内外を判別できる（たぶん）。

    なお、作成したドロネー図は以下のコードでプロット可能。
    1点補足しておくと、```plt.triplot``` の第三引数は、
    第一、第二引数から三角形を作成するための **インデックス** のリスト
    になっている。[[0, 1, 2], [2, 4, 3], ...]的な。

    ```python
    plt.figure()
    plt.triplot(xy[:, 0], xy[:, 1], triangulation.simplices.copy(), '-o')
    plt.title('triplot of Delaunay triangulation')
    plt.show()
    ```
    """
    triangulation = Delaunay(cmf_xy)

    """
    ```triangulation.find_simplex()``` で xy がどのインデックスの領域か
    調べることができる。戻り値が ```-1``` の場合は領域に含まれないため、
    0以下のリストで領域判定の mask を作ることができる。
    """
    xx, yy\
        = np.meshgrid(np.linspace(0, 1, samples), np.linspace(1, 0, samples))
    xy = np.dstack((xx, yy))
    mask = (triangulation.find_simplex(xy) < 0).astype(np.float)

    # アンチエイリアシングしてアルファチャンネルを滑らかに
    # ------------------------------------------------
    if antialiasing:
        kernel = np.array([
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0],
        ]).astype(np.float)
        kernel /= np.sum(kernel)
        mask = convolve(mask, kernel)

    # ネガポジ反転
    # --------------------------------
    mask = 1 - mask[:, :, np.newaxis]

    # xy のメッシュから色を復元
    # ------------------------
    illuminant_XYZ = D65_WHITE
    illuminant_RGB = color_space.whitepoint
    chromatic_adaptation_transform = 'CAT02'
    large_xyz_to_rgb_matrix = color_space.XYZ_to_RGB_matrix
    large_xyz = xy_to_XYZ(xy)
    rgb = XYZ_to_RGB(large_xyz, illuminant_XYZ, illuminant_RGB,
                     large_xyz_to_rgb_matrix,
                     chromatic_adaptation_transform)

    """
    そのままだとビデオレベルが低かったりするので、
    各ドット毎にRGB値を正規化＆最大化する。
    """
    rgb = normalise_maximum(rgb, axis=-1)

    # mask 適用
    # -------------------------------------
    mask_rgb = np.dstack((mask, mask, mask))
    rgb *= mask_rgb

    # 背景色をグレーに変更
    # -------------------------------------
    bg_rgb = np.ones_like(rgb)
    bg_rgb *= (1 - mask_rgb) * bg_color

    rgb += bg_rgb

    rgb = rgb ** (1/2.2)

    return rgb


def get_csf_color_image(width=640, height=480,
                        lv1=np.uint16(np.array([1.0, 1.0, 1.0]) * 1023 * 0x40),
                        lv2=np.uint16(np.array([1.0, 1.0, 1.0]) * 512 * 0x40),
                        stripe_num=18):
    """
    長方形を複数個ズラして重ねることでCSFパターンっぽいのを作る。
    入力信号レベルは16bitに限定する。

    Parameters
    ----------
    width : numeric.
        width of the pattern image.
    height : numeric.
        height of the pattern image.
    lv1 : numeric
        video level 1. this value must be 10bit.
    lv2 : numeric
        video level 2. this value must be 10bit.
    stripe_num : numeric
        number of the stripe.

    Returns
    -------
    array_like
        a cms pattern image.
    """
    width_list = equal_devision(width, stripe_num)
    height_list = equal_devision(height, stripe_num)
    h_pos_list = equal_devision(width // 2, stripe_num)
    v_pos_list = equal_devision(height // 2, stripe_num)
    lv1_16bit = lv1
    lv2_16bit = lv2
    img = np.zeros((height, width, 3), dtype=np.uint16)
    
    width_temp = width
    height_temp = height
    h_pos_temp = 0
    v_pos_temp = 0
    for idx in range(stripe_num):
        lv = lv1_16bit if (idx % 2) == 0 else lv2_16bit
        temp_img = np.ones((height_temp, width_temp, 3), dtype=np.uint16)
        # temp_img *= lv
        temp_img[:, :] = lv
        ed_pos_h = h_pos_temp + width_temp
        ed_pos_v = v_pos_temp + height_temp
        img[v_pos_temp:ed_pos_v, h_pos_temp:ed_pos_h] = temp_img
        width_temp -= width_list[stripe_num - 1 - idx]
        height_temp -= height_list[stripe_num - 1 - idx]
        h_pos_temp += h_pos_list[idx]
        v_pos_temp += v_pos_list[idx]

    return img


def plot_xyY_color_space(name='ITU-R BT.2020', samples=1024,
                         antialiasing=True):
    """
    SONY の HDR説明資料にあるような xyY の図を作る。

    Parameters
    ----------
    name : str
        name of the target color space.

    Returns
    -------
    None

    """

    # 馬蹄の領域判別用データ作成
    # --------------------------
    primary_xy, _ = get_primaries(name=name)
    triangulation = Delaunay(primary_xy)

    xx, yy\
        = np.meshgrid(np.linspace(0, 1, samples), np.linspace(1, 0, samples))
    xy = np.dstack((xx, yy))
    mask = (triangulation.find_simplex(xy) < 0).astype(np.float)

    # アンチエイリアシングしてアルファチャンネルを滑らかに
    # ------------------------------------------------
    if antialiasing:
        kernel = np.array([
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0],
        ]).astype(np.float)
        kernel /= np.sum(kernel)
        mask = convolve(mask, kernel)

    # ネガポジ反転
    # --------------------------------
    mask = 1 - mask[:, :, np.newaxis]

    # xy のメッシュから色を復元
    # ------------------------
    illuminant_XYZ = D65_WHITE
    illuminant_RGB = RGB_COLOURSPACES[name].whitepoint
    chromatic_adaptation_transform = 'CAT02'
    large_xyz_to_rgb_matrix = get_xyz_to_rgb_matrix(name)
    rgb_to_large_xyz_matrix = get_rgb_to_xyz_matrix(name)
    large_xyz = xy_to_XYZ(xy)
    rgb = XYZ_to_RGB(large_xyz, illuminant_XYZ, illuminant_RGB,
                     large_xyz_to_rgb_matrix,
                     chromatic_adaptation_transform)

    """
    そのままだとビデオレベルが低かったりするので、
    各ドット毎にRGB値を正規化＆最大化する。
    """
    rgb_org = normalise_maximum(rgb, axis=-1)

    # mask 適用
    # -------------------------------------
    mask_rgb = np.dstack((mask, mask, mask))
    rgb = rgb_org * mask_rgb
    rgba = np.dstack((rgb, mask))

    # こっからもういちど XYZ に変換。Yを求めるために。
    # ---------------------------------------------
    large_xyz2 = RGB_to_XYZ(rgb, illuminant_RGB, illuminant_XYZ,
                            rgb_to_large_xyz_matrix,
                            chromatic_adaptation_transform)

    # ログスケールに変換する準備
    # --------------------------
    large_y = large_xyz2[..., 1] * 1000
    large_y[large_y < 1] = 1.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_wireframe(xy[..., 0], xy[..., 1], np.log10(large_y),
    #                   rcount=100, ccount=100)
    ax.plot_surface(xy[..., 0], xy[..., 1], np.log10(large_y),
                    rcount=64, ccount=64, facecolors=rgb_org)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Y")
    ax.set_zticks([0, 1, 2, 3])
    ax.set_zticklabels([1, 10, 100, 1000])

    # chromatcity_image の取得。z=0 の位置に貼り付ける
    # ----------------------------------------------
    cie1931_rgb = get_chromaticity_image(samples=samples, bg_color=0.0)

    alpha = np.zeros_like(cie1931_rgb[..., 0])
    rgb_sum = np.sum(cie1931_rgb, axis=-1)
    alpha[rgb_sum > 0.00001] = 1
    cie1931_rgb = np.dstack((cie1931_rgb[..., 0], cie1931_rgb[..., 1],
                             cie1931_rgb[..., 2], alpha))
    zz = np.zeros_like(xy[..., 0])
    ax.plot_surface(xy[..., 0], xy[..., 1], zz,
                    facecolors=cie1931_rgb)

    plt.show()


def log_tick_formatter(val, pos=None):
    return "{:.0e}".format(10**val)


def get_3d_grid_cube_format(grid_num=4):
    """
    # 概要
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), ...
    みたいな配列を返す。
    CUBE形式の3DLUTを作成する時に便利。
    """

    base = np.linspace(0, 1, grid_num)
    ones_x = np.ones((grid_num, grid_num, 1))
    ones_y = np.ones((grid_num, 1, grid_num))
    ones_z = np.ones((1, grid_num, grid_num))
    r_3d = base[np.newaxis, np.newaxis, :] * ones_x
    g_3d = base[np.newaxis, :, np.newaxis] * ones_y
    b_3d = base[:, np.newaxis, np.newaxis] * ones_z
    r_3d = r_3d.flatten()
    g_3d = g_3d.flatten()
    b_3d = b_3d.flatten()

    return np.dstack((r_3d, g_3d, b_3d))


def quadratic_bezier_curve(t, p0, p1, p2, samples=1024):
    # x = ((1 - t) ** 2) * p0[0] + 2 * (1 - t) * t * p1[0]\
    #     + (t ** 2) * p2[0]
    # y = ((1 - t) ** 2) * p0[1] + 2 * (1 - t) * t * p1[1]\
    #     + (t ** 2) * p2[1]

    x = ((1 - t) ** 2) * p0[0] + 2 * (1 - t) * t * p1[0]\
        + (t ** 2) * p2[0]
    y = ((1 - t) ** 2) * p0[1] + 2 * (1 - t) * t * p1[1]\
        + (t ** 2) * p2[1]

    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title="Title",
                          graph_title_size=None,
                          xlabel="X Axis Label", ylabel="Y Axis Label",
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
    ax1.plot(x, y, label='aaa')
    plt.legend(loc='upper left')
    plt.show()


def gen_step_gradation(width=1024, height=128, step_num=17,
                       bit_depth=10, color=(1.0, 1.0, 1.0),
                       direction='h', debug=False):
    """
    # 概要
    階段状に変化するグラデーションパターンを作る。
    なお、引数の調整により正確に1階調ずつ変化するパターンも作成可能。

    # 注意事項
    正確に1階調ずつ変化するグラデーションを作る場合は
    ```step_num = (2 ** bit_depth) + 1```
    となるようにパラメータを指定すること。具体例は以下のExample参照。

    # Example
    ```
    grad_8 = gen_step_gradation(width=grad_width, height=grad_height,
                                step_num=257, bit_depth=8,
                                color=(1.0, 1.0, 1.0), direction='h')

    grad_10 = gen_step_gradation(width=grad_width, height=grad_height,
                                 step_num=1025, bit_depth=10,
                                 color=(1.0, 1.0, 1.0), direction='h')
    ```
    """
    max = 2 ** bit_depth

    # グラデーション方向設定
    # ----------------------
    if direction == 'h':
        pass
    else:
        temp = height
        height = width
        width = temp

    if (max + 1 != step_num):
        """
        1階調ずつの増加では無いパターン。
        末尾のデータが 256 や 1024 になるため -1 する。
        """
        val_list = np.linspace(0, max, step_num)
        val_list[-1] -= 1
    else:
        """
        正確に1階調ずつ変化するパターン。
        末尾のデータが 256 や 1024 になるため除外する。
        """
        val_list = np.linspace(0, max, step_num)[0:-1]
        step_num -= 1  # step_num は 引数で余計に +1 されてるので引く

        # 念のため1階調ずつの変化か確認
        # ---------------------------
        diff = val_list[1:] - val_list[0:-1]
        if (diff == 1).all():
            pass
        else:
            raise ValueError("calculated value is invalid.")

    # まずは水平1LINEのグラデーションを作る
    # -----------------------------------
    step_length_list = common.equal_devision(width, step_num)
    step_bar_list = []
    for step_idx, length in enumerate(step_length_list):
        step = [np.ones((length)) * color[c_idx] * val_list[step_idx]
                for c_idx in range(3)]
        if direction == 'h':
            step = np.dstack(step)
            step_bar_list.append(step)
            step_bar = np.hstack(step_bar_list)
        else:
            step = np.dstack(step).reshape((length, 1, 3))
            step_bar_list.append(step)
            step_bar = np.vstack(step_bar_list)

    # ブロードキャストを利用して2次元に拡張する
    # ------------------------------------------
    if direction == 'h':
        img = step_bar * np.ones((height, 1, 3))
    else:
        img = step_bar * np.ones((1, height, 3))

    # np.uint16 にコンバート
    # ------------------------------
    # img = np.uint16(np.round(img * (2 ** (16 - bit_depth))))

    if debug:
        preview_image(img, 'rgb')

    return img


def merge(img_a, img_b, pos=(0, 0)):
    """
    img_a に img_b をマージする。
    img_a にデータを上書きする。

    pos = (horizontal_st, vertical_st)
    """
    b_width = img_b.shape[1]
    b_height = img_b.shape[0]

    img_a[pos[1]:b_height+pos[1], pos[0]:b_width+pos[0]] = img_b


def dot_pattern(dot_size=4, repeat=4, color=(1.0, 1.0, 1.0)):
    """
    dot pattern 作る。
    dot_size: 1なら 1dot, 2なら2dot.
    repeat: 4なら high-lowのペアが4組.
    color: [0:1] で正規化した色を指定
    """
    pixel_num = dot_size * 2 * repeat
    even_logic = [(np.arange(pixel_num) % (dot_size * 2)) - dot_size < 0]
    even_logic = np.dstack((even_logic, even_logic, even_logic))
    odd_logic = np.logical_not(even_logic)
    color = np.array(color).reshape((1, 1, 3))

    even_line = (np.ones((1, pixel_num, 3)) * even_logic) * color
    odd_line = (np.ones((1, pixel_num, 3)) * odd_logic) * color

    even_block = np.repeat(even_line, dot_size, axis=0)
    odd_block = np.repeat(odd_line, dot_size, axis=0)

    pair_block = np.vstack((even_block, odd_block))

    img = np.vstack([pair_block for x in range(repeat)])

    # tpg.preview_image(img)

    return img


def complex_dot_pattern(kind_num=3, whole_repeat=2,
                        fg_color=(1.0, 1.0, 1.0), bg_color=(0.15, 0.15, 0.15)):
    """
    kind_num: 何段階の大きさよ用意するか
    whole_repeat: kind_num * 2 のセットを何回ループするのか？
    """
    max_dot_width = 2 ** kind_num
    img_list = []
    for size_idx in range(kind_num)[::-1]:
        dot_size = 2 ** size_idx
        repeat = max_dot_width // dot_size
        dot_img = dot_pattern(dot_size, repeat, fg_color)
        img_list.append(dot_img)
        img_list.append(np.ones_like(dot_img) * bg_color)
        # preview_image(dot_img)

    line_upper_img = np.hstack(img_list)
    line_upper_img = np.hstack([line_upper_img for x in range(whole_repeat)])
    line_lower_img = line_upper_img.copy()[:, ::-1, :]
    h_unit_img = np.vstack((line_upper_img, line_lower_img))

    img = np.vstack([h_unit_img for x in range(kind_num * whole_repeat)])
    preview_image(img)
    cv2.imwrite("hoge.tiff", np.uint8(img * 0xFF)[..., ::-1])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_chromaticity_diagram()
    # plot_xyY_color_space(name='ITU-R BT.2020', samples=256)
    # samples = 1024
    # p0 = np.array([0.5, 0.5])
    # p1 = np.array([0.75, 1.0])
    # p2 = np.array([1.0, 1.0])
    # x = np.linspace(0, 1.0, samples)
    # quadratic_bezier_curve(x, p0, p1, p2, samples)
    # dot_pattern(dot_size=32, repeat=4, color=(1.0, 1.0, 1.0))
    complex_dot_pattern(kind_num=3, whole_repeat=1, fg_color=(1.0, 1.0, 1.0),
                        bg_color=(0.15, 0.15, 0.15))

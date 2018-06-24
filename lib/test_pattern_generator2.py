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
import numpy as np
from colour.colorimetry import CMFS, ILLUMINANTS
from colour.models import XYZ_to_xy, xy_to_XYZ, XYZ_to_RGB, RGB_to_XYZ
from colour.utilities import normalise_maximum
from colour import models
from colour import RGB_COLOURSPACES
from scipy.spatial import Delaunay
from scipy.ndimage.filters import convolve
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


def xy_to_rgb(xy, name='ITU-R BT.2020'):
    """
    xy値からRGB値を算出する。
    いい感じに正規化もしておく。

    Parameters
    ----------
    xy : array_like
        xy value.
    name : color space name.

    Returns
    -------
    array_like
        rgb value. the value is normalized.

    """
    illuminant_XYZ = D65_WHITE
    illuminant_RGB = D65_WHITE
    chromatic_adaptation_transform = 'CAT02'
    large_xyz_to_rgb_matrix = get_xyz_to_rgb_matrix(name)
    large_xyz = xy_to_XYZ(xy)
    rgb = XYZ_to_RGB(large_xyz, illuminant_XYZ, illuminant_RGB,
                     large_xyz_to_rgb_matrix,
                     chromatic_adaptation_transform)

    """
    そのままだとビデオレベルが低かったりするので、
    各ドット毎にRGB値を正規化＆最大化する。
    """
    rgb = normalise_maximum(rgb, axis=-1)

    return rgb


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


def plot_chromaticity_diagram(rate=480/751.0*2, **kwargs):
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


def get_chromaticity_image(samples=1024, antialiasing=True):
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
    bg_rgb *= (1 - mask_rgb) * 0.9

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
    width_list = cmn.equal_devision(width, stripe_num)
    height_list = cmn.equal_devision(height, stripe_num)
    h_pos_list = cmn.equal_devision(width // 2, stripe_num)
    v_pos_list = cmn.equal_devision(height // 2, stripe_num)
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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_chromaticity_diagram()

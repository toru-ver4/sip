#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
評価用のテストパターン作成ツール集

"""

import os
import cv2
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
    # 基本パラメータ設定
    # ------------------
    cmf = CMFS.get(CMFS_NAME)
    d65_white = D65_WHITE

    # 馬蹄形のxy値を算出
    # --------------------------
    cmf_xy = XYZ_to_xy(cmf.values, d65_white)

    return cmf_xy


def get_primaries(name='ITU-R BT.2020'):
    primaries = RGB_COLOURSPACES[name].primaries
    primaries = np.append(primaries, [primaries[0, :]], axis=0)

    return primaries


def _get_test_scatter_data():
    sample_num = 7
    base = (np.linspace(0, 1, sample_num) ** (2.0))[::-1]
    ones = np.ones_like(base)

    r = np.dstack((ones, base, base))
    g = np.dstack((base, ones, base))
    b = np.dstack((base, base, ones))
    rgb = np.append(np.append(r, g, axis=0), b, axis=0)

    color_space = models.BT2020_COLOURSPACE
    illuminant_XYZ = D65_WHITE
    illuminant_RGB = color_space.whitepoint
    chromatic_adaptation_transform = 'CAT02'
    rgb_to_xyz_matrix = color_space.RGB_to_XYZ_matrix
    large_xyz = RGB_to_XYZ(rgb, illuminant_RGB, illuminant_XYZ,
                           rgb_to_xyz_matrix,
                           chromatic_adaptation_transform)

    xy = XYZ_to_xy(large_xyz)

    rgb = rgb ** (1/2.2)

    return xy, rgb.reshape((rgb.shape[0] * rgb.shape[1], 3))


def plot_chromaticity_diagram(primaries=None):
    xy_image = get_chromaticity_image()
    rate = 1.5
    cmf_xy = _get_cmfs_xy()

    bt2020_gamut = get_primaries('ITU-R BT.2020')
    dci_p3_gamut = get_primaries('DCI-P3')

    ax1 = pu.plot_1_graph(fontsize=15 * rate,
                          figsize=(8 * rate, 9 * rate),
                          graph_title="CIE1931 Chromaticity Diagram",
                          graph_title_size=None,
                          xlabel=None, ylabel=None,
                          axis_label_size=None,
                          legend_size=14 * rate,
                          xlim=(0, 0.8),
                          ylim=(0, 0.9),
                          xtick=[x * 0.1 for x in range(9)],
                          ytick=[x * 0.1 for x in range(10)],
                          xtick_size=10 * rate,
                          ytick_size=10 * rate,
                          linewidth=2 * rate,
                          minor_xtick_num=2,
                          minor_ytick_num=2)
    ax1.plot(cmf_xy[..., 0], cmf_xy[..., 1], '-k', label=None)
    ax1.plot((cmf_xy[-1, 0], cmf_xy[0, 0]), (cmf_xy[-1, 1], cmf_xy[0, 1]),
             '-k', label=None)
    ax1.plot(bt2020_gamut[:, 0], bt2020_gamut[:, 1], c="#FFD000",
             label="BT.2020", lw=3*rate)
    ax1.plot(dci_p3_gamut[:, 0], dci_p3_gamut[:, 1], c="#FF3030",
             label="DCI-P3", lw=3*rate)
    if primaries is not None:
        ax1.plot(primaries[:, 0], primaries[:, 1], c="#202020",
                 label="???", lw=3*rate)
    ax1.imshow(xy_image, extent=(0, 1, 0, 1))
    xy, rgb = _get_test_scatter_data()
    ax1.scatter(xy[..., 0], xy[..., 1], s=1500, marker='s', c=rgb,
                edgecolors='#404040', linewidth=2*rate)
    plt.legend(loc='upper right')
    plt.show()


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
    そのままだとビデオレベルひ低かったりするので、
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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_chromaticity_diagram()

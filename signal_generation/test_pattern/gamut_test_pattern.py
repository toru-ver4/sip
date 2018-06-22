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
import colour
# from PIL import Image
from PIL import ImageCms
import imp
imp.reload(tpg)
imp.reload(ImageCms)


def _get_monitor_primaries(filename="./icc_profile/gamut.icc"):
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
        

    Returns
    -------
    array_like
        prmaries. [[rx, ry], [gx, gy], [bx, by], [rx, ry]]

    Examples
    --------
    >>> chromaticity_diagram_plot_CIE1931()  # doctest: +SKIP
    """
    pass


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


def _gen_ycbcr_ng_combination_checker():
    ok_low_level = 186  # 744(10bit)
    ok_high_level = 193  # 772(10bit)

    ng_low_level = 193  # 772(10bit)
    ng_high_level = 198  # 792(10bit)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # primaries = _get_monitor_primaries()
    # print(primaries)
    # tpg.plot_chromaticity_diagram(primaries=primaries)
    _check_clip_level(src='ITU-R BT.709', dst='ITU-R BT.601')
    _check_clip_level(src='ITU-R BT.709', dst='ITU-R BT.2020')
    _check_clip_level(src='ITU-R BT.2020', dst='ITU-R BT.601')
    _check_clip_level(src='ITU-R BT.2020', dst='ITU-R BT.709')

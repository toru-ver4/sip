#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# スペクトルからカラーチェッカー値を算出する

## どうにかしたい点
"""

import os
import numpy as np
import colorimetry as cm
from colour.colorimetry.spectrum import SpectralShape
from colour.algebra import SpragueInterpolator, LinearInterpolator,\
    CubicSplineInterpolator
import matplotlib.pyplot as plt
from colour import XYZ_to_xy


def compare_sprague_and_spline():
    """
    SpragueInterpolator, CubicSplineInterpolator で
    Color Checker のスペクトルを 10nm --> 1nm に補間した場合の
    比較をプロットして確認した。
    """
    sprague = cm.load_colorchecker_spectrum(SpragueInterpolator)
    spline = cm.load_colorchecker_spectrum(CubicSplineInterpolator)
    original = cm.load_colorchecker_spectrum(None)

    # plot
    # ----------------------------------
    v_num = 4
    h_num = 6
    plt.rcParams["font.size"] = 18
    f, axarr = plt.subplots(v_num, h_num, sharex='col', sharey='row',
                            figsize=(24, 16))
    for idx in range(24):
        h_idx = idx % h_num
        v_idx = idx // h_num
        axarr[v_idx, h_idx].grid()
        if v_idx == (v_num - 1):
            axarr[v_idx, h_idx].set_xlabel("wavelength [nm]")
        if h_idx == 0:
            axarr[v_idx, h_idx].set_ylabel("reflectance")
        axarr[v_idx, h_idx].set_xlim(380, 730)
        axarr[v_idx, h_idx].set_ylim(0, 1.0)
        axarr[v_idx, h_idx].set_xticks([400, 500, 600, 700])
        axarr[v_idx, h_idx].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        x3 = sprague.wavelengths
        y3 = sprague.values[:, idx]
        axarr[v_idx, h_idx].plot(x3, y3, '-o', label='sprague')
        x2 = spline.wavelengths
        y2 = spline.values[:, idx]
        axarr[v_idx, h_idx].plot(x2, y2, '-o', label='spline')
        x1 = original.wavelengths
        y1 = original.values[:, idx]
        axarr[v_idx, h_idx].plot(x1, y1, '-o', label='original')
    # plt.savefig('temp_fig.png', bbox_inches='tight')
    plt.show()


def make_color_chekcer_linear_img(cmfs_name=cm.CIE1931, temperature=6500):
    """
    color checker の RGB値（Linear）を計算する。
    """
    # load & get basic data
    color_checker_spd = cm.load_colorchecker_spectrum()
    cmfs_spd = cm.load_cmfs(cmfs_name)
    illuminant_spd = cm.get_day_light_spd(temperature=6500, interval=1)


    # calc large xyz


def calc_day_light_xy(temperature=6500):
    day_light_spd = cm.get_day_light_spd(temperature)
    cmfs = cm.load_cmfs(cm.CIE1931)

    # trim
    shape = cm.calc_appropriate_shape(day_light_spd, cmfs)
    shape = SpectralShape(380, 780)
    day_light_spd.trim(shape)
    cmfs.trim(shape)
    print(cmfs.values)

    # calc
    large_x = np.sum(day_light_spd.values * cmfs.values[:, 0])
    large_y = np.sum(day_light_spd.values * cmfs.values[:, 1])
    large_z = np.sum(day_light_spd.values * cmfs.values[:, 2])
    normalize_coef = 100 / large_y
    large_xyz = [large_x * normalize_coef,
                 large_y * normalize_coef,
                 large_z * normalize_coef]

    print(large_xyz)
    print(XYZ_to_xy(large_xyz))


def test_func():
    # compare_sprague_and_spline()
    # make_color_chekcer_linear_img(cm.CIE1931)
    calc_day_light_xy(6500)
    # make_color_chekcer_linear_img(cm.CIE2015_2)


def main_func():
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_func()
    main_func()

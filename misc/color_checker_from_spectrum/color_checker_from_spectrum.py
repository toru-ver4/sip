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
import color_space as cs
import transfer_functions as tf
import test_pattern_generator2 as tpg

SRGB_CS = cs.RGB_COLOURSPACES[cs.SRTB]
BT709_CS = cs.RGB_COLOURSPACES[cs.BT709]
BT2020_CS = cs.RGB_COLOURSPACES[cs.BT2020]
P3_D65_CS = cs.RGB_COLOURSPACES[cs.P3_D65]


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


def calc_day_light_xy(temperature=6500, cmfs_name=cm.CIE1931):
    day_light_spd = cm.get_day_light_spd(temperature)
    cmfs = cm.load_cmfs(cmfs_name)
    normalize_coef = cm.get_nomalize_large_y_coef(
        d_light_before_trim=day_light_spd, cmfs_before_trim=cmfs)

    # trim
    shape = cm.calc_appropriate_shape(day_light_spd, cmfs)
    day_light_spd.trim(shape)
    cmfs.trim(shape)

    # calc
    large_xyz = [np.sum(day_light_spd.values * cmfs.values[:, x])
                 * normalize_coef for x in range(3)]
    print(large_xyz)
    print(XYZ_to_xy(large_xyz))


def make_color_chekcer_linear_value(cmfs_name=cm.CIE1931, temperature=6500,
                                    color_space=SRGB_CS):
    """
    color checker の RGB値（Linear）を計算する。
    """
    # load & get basic data
    color_checker_spd = cm.load_colorchecker_spectrum()
    cmfs_spd = cm.load_cmfs(cmfs_name)
    day_light_spd = cm.get_day_light_spd(temperature=temperature, interval=1)

    # calc large xyz
    large_xyz = cm.colorchecker_spectrum_to_large_xyz(
        d_light=day_light_spd, color_checker=color_checker_spd, cmfs=cmfs_spd)

    # large xyz to rgb(linear)
    rgb_linear = cm.color_checker_large_xyz_to_rgb(large_xyz, color_space)

    return rgb_linear


def make_color_chekcer_value(cmfs_name=cm.CIE1931, temperature=6500,
                             color_space=SRGB_CS, oetf_name=tf.SRGB):
    """
    color checker を OETF でエンコードしたRGB値を作る。
    """
    linear_rgb = make_color_chekcer_linear_value(cmfs_name=cmfs_name,
                                                 temperature=temperature,
                                                 color_space=color_space)
    encoded_rgb = tf.oetf(linear_rgb, oetf_name)

    return encoded_rgb


def test_plot(cmfs_name=cm.CIE1931, temperature=6500,
              color_space=SRGB_CS, oetf_name=tf.SRGB):
    rgb = make_color_chekcer_value(
        cmfs_name=cmfs_name, temperature=temperature,
        color_space=color_space, oetf_name=oetf_name)
    rgb2 = make_color_chekcer_value(
        cmfs_name=cmfs_name, temperature=temperature,
        color_space=color_space, oetf_name=oetf_name)
    rgb = np.uint8(np.round(rgb * 0xFF))
    rgb2 = np.uint8(np.round(rgb2 * 0xFF))
    tpg.plot_color_checker_image(rgb, rgb2)


def temperature_convert_test(
        cmfs_name=cm.CIE1931, temperature=6500,
        color_space=SRGB_CS, oetf_name=tf.SRGB):
    full_spectrum_rgb = make_color_chekcer_value(
        cmfs_name=cmfs_name, temperature=temperature,
        color_space=color_space, oetf_name=oetf_name)

    base_d65_rgb_linear = make_color_chekcer_linear_value(
        cmfs_name=cmfs_name, temperature=6500,
        color_space=color_space)
    temp_conv_rgb_linear = cm.temperature_convert(
        base_d65_rgb_linear, 6500, temperature,
        chromatic_adaptation='CAT02', color_space=color_space)
    temp_conv_rgb = tf.oetf(temp_conv_rgb_linear, oetf_name)

    full_spectrum_rgb = np.uint8(np.round(full_spectrum_rgb * 0xFF))
    temp_conv_rgb = np.uint8(np.round(temp_conv_rgb * 0xFF))
    tpg.plot_color_checker_image(full_spectrum_rgb, temp_conv_rgb)


def test_func():
    # compare_sprague_and_spline()
    # calc_day_light_xy(6500, cm.CIE1931)
    # calc_day_light_xy(5000, cm.CIE1931)
    # calc_day_light_xy(6500, cm.CIE2015_2)
    # calc_day_light_xy(5000, cm.CIE2015_2)
    # make_color_chekcer_linear_value(cm.CIE1931, 6500, SRGB_CS)
    # test_plot(cmfs_name=cm.CIE1931, temperature=6500,
    #           color_space=BT709_CS, oetf_name=tf.SRGB)
    temperature_convert_test(
        cmfs_name=cm.CIE1931, temperature=2000,
        color_space=SRGB_CS, oetf_name=tf.SRGB)


def main_func():
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_func()
    main_func()

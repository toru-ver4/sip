#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
スペクトルからカラーチェッカー値を算出する
"""

import os
import numpy as np
from colour.colorimetry import STANDARD_OBSERVERS_CMFS
from colour.colorimetry.spectrum import SpectralShape
from colour.colorimetry import ILLUMINANTS
import re
import matplotlib.pyplot as plt
import plot_utility as pu
from colour.temperature import CCT_to_xy_CIE_D
from colour import sd_CIE_illuminant_D_series
from colour.utilities import tstack
from colour.algebra import SpragueInterpolator, LinearInterpolator,\
    CubicSplineInterpolator
from colour.colorimetry import MultiSpectralDistribution
from colour.notation import munsell_colour_to_xyY
from colour.models import sRGB_COLOURSPACE
from colour import xyY_to_XYZ, XYZ_to_RGB
from colour.models import oetf_sRGB

CMFS_NAME = 'CIE 1931 2 Degree Standard Observer'
D65_WHITE = ILLUMINANTS[CMFS_NAME]['D65']
C_WHITE = ILLUMINANTS[CMFS_NAME]['C']

CIE1931 = 'CIE 1931 2 Degree Standard Observer'
CIE1964 = 'CIE 1964 10 Degree Standard Observer'
CIE2015_2 = 'CIE 2012 2 Degree Standard Observer'
CIE2015_10 = 'CIE 2012 10 Degree Standard Observer'

R_PLOT_COLOR = "#{:02x}{:02x}{:02x}".format(255, 75, 0)
G_PLOT_COLOR = "#{:02x}{:02x}{:02x}".format(3, 175, 122)
B_PLOT_COLOR = "#{:02x}{:02x}{:02x}".format(0, 90, 255)

D65_xy = np.array([0.3127, 0.3290])


def modify_d65_csv():
    """CIE S 014-2 のファイルを作る"""
    src_file = "./src_data/d65_CIE_S_014-2_org.csv"
    dst_file = "./src_data/d65_CIE_S_014-2.csv"
    pattern = re.compile(r'^(\d+) (\d+,\d+)(\s*\d*) (\d+,\d+)(\s*\d*)')
    out_buf = []

    with open(src_file, 'r') as f:
        for line in f:
            line = line.rstrip()
            m = pattern.match(line)
            if m is None:
                print("yabai!")
            base = "{}, {}, {}\n"
            wavelength = m.group(1)
            s_a = m.group(2).replace(',', '.') + m.group(3).replace(" ", "")
            s_d = m.group(4).replace(',', '.') + m.group(5).replace(" ", "")
            out_buf.append(base.format(wavelength, s_a, s_d))

    with open(dst_file, 'w') as f:
        f.write("".join(out_buf))


def get_xyz_wavelength(msd, base):
    """
    糞CSVファイルから wavelength を生成する
    """
    if len(base) == 2:
        wavelength = str(msd) + base
    else:
        wavelength = base

    ret_msd = int(msd) + 1 if base == "99" else msd

    return ret_msd, wavelength


def generate_xyz_cmfs(base0, base1, base2, base3, base4):
    high = base0.replace(",", ".")
    low = base2.replace(" ", "")
    lowlow = base3.replace(" ", "")
    lowlowlow = base4.replace(" ", "")
    return high + base1 + low + lowlow + lowlowlow


def modify_xyz_csv():
    """CIE S 014-1 のファイルを作る"""
    src_file = "./src_data/CMFs_CIE_S_014-1_org.csv"
    dst_file = "./src_data/CMFs_CIE_S_014-1.csv"

    pattern = re.compile(r'^(\d+) (\d+,\d+) (\d+) (\d+)(\s*\d*)(\s*\d*) (\d+,\d+) (\d+) (\d+)(\s*\d*)(\s*\d*) (\d+,\d+) (\d+) (\d+)(\s*\d*)(\s*\d*) (\d.*)')
    out_buf = []

    msd = 3
    with open(src_file, 'r') as f:
        for line in f:
            line = line.rstrip()
            m = pattern.match(line)
            if m is None:
                print("yabai!")
            else:
                msd, wavelength = get_xyz_wavelength(msd, m.group(1))
                out_str = "{}, {}, {}, {}\n"
                x = generate_xyz_cmfs(m.group(2), m.group(3), m.group(4),
                                      m.group(5), m.group(6))
                y = generate_xyz_cmfs(m.group(7), m.group(8), m.group(9),
                                      m.group(10), m.group(11))
                z = generate_xyz_cmfs(m.group(12), m.group(13), m.group(14),
                                      m.group(15), m.group(16))
                out_buf.append(out_str.format(wavelength, x, y, z))

    with open(dst_file, 'w') as f:
        f.write("".join(out_buf))


def compare_cmfs_from_web():
    """
    http://cvrl.ioo.ucl.ac.uk/cie.htm のデータと
    CIE S 014-1 のファイルを比較
    """
    cvrl_file = "./src_data/CIE1931_1nm_cvrl.csv"
    cie_file = "./src_data/CMFs_CIE_S_014-1.csv"

    cvrl = np.loadtxt(cvrl_file, delimiter=',').T
    print(cvrl.shape)
    cie = np.loadtxt(cie_file, delimiter=',').T
    print(cie.shape)

    diff = np.abs(cvrl - cie)
    print(np.max(diff))


def compare_d65_spd_from_web():
    cvrl_file = "./src_data/d65_1nm_cvrl.csv"
    cie_file = "./src_data/d65_CIE_S_014-2.csv"

    cvrl = np.loadtxt(cvrl_file, delimiter=',').T
    print(cvrl.shape)
    cie = np.loadtxt(cie_file, delimiter=',').T
    print(cie.shape)

    diff = np.abs(cvrl[1] - cie[2])
    print(np.max(diff))


def check_cmfs():
    shape = SpectralShape(300, 830, 1)
    print(STANDARD_OBSERVERS_CMFS[CIE1931].trim(shape))
    print(D_illuminant_relative_spd(D65_xy).trim(shape))


def plot_cmfs():
    cmfs_file = "./src_data/CMFs_CIE_S_014-1.csv"
    cmfs = np.loadtxt(cmfs_file, delimiter=',').T
    wl = cmfs[0].flatten()
    x = cmfs[1].flatten()
    y = cmfs[2].flatten()
    z = cmfs[3].flatten()

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        graph_title="CIE 1931 2-deg, XYZ CMFs",
        xlabel="wavelength [nm]",
        ylabel="tristimulus values")
    ax1.plot(wl, x, '-r', label='x')
    ax1.plot(wl, y, '-g', label='y')
    ax1.plot(wl, z, '-b', label='z')
    plt.legend(loc='upper left')
    plt.show()


def plot_d65():
    d65_file = "./src_data/d65_CIE_S_014-2.csv"
    d65 = np.loadtxt(d65_file, delimiter=',').T
    wl = d65[0].flatten()
    d65_spd = d65[2].flatten()
    d65_spd_sprague = make_day_light_by_calculation(
        temperature=6504.0, interpolater=SpragueInterpolator)
    d65_spd_linear = make_day_light_by_calculation(
        temperature=6504.0, interpolater=LinearInterpolator)

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        linewidth=5,
        graph_title="CIE Illuminant D65",
        xlabel="wavelength [nm]",
        ylabel="rerative spectral power distributions")
    ax1.plot(wl, d65_spd, '-', color=R_PLOT_COLOR,
             label='D65 CIE S 014-2')
    ax1.plot(wl, d65_spd_linear[1, ...], '--', lw=2, color=G_PLOT_COLOR,
             label='D65 Calc with LinearInterpolation')
    ax1.plot(wl, d65_spd_sprague[1, ...], '-', lw=2, color=B_PLOT_COLOR,
             label='D65 Calc with SpragueInterpolation')
    plt.legend(loc='upper left')
    plt.show()


def make_day_light_by_calculation(temperature=6500,
                                  interpolater=None,
                                  interval=1):
    """
    計算でD光源を作る。

    interpolater: SpragueInterpolator or LinearInterpolator
    """
    xy = CCT_to_xy_CIE_D(temperature)
    spd = sd_CIE_illuminant_D_series(xy)
    spd = spd.interpolate(SpectralShape(interval=interval),
                          interpolator=interpolater)
    # ret_value = np.dstack([spd.wavelengths, spd.values])

    # return ret_value.reshape((len(spd.values), 2)).T
    return spd


def make_multispectral_format_data(wavelengths, values, name="sample"):
    dic = dict(zip(np.uint16(wavelengths).tolist(), values.tolist()))

    return dic


def get_interpolater(interplation="linear"):
    if interplation == "linear":
        interpolator = LinearInterpolator
    elif interplation == "sprague":
        interpolator = SpragueInterpolator
    elif interplation == "spline":
        interpolator = CubicSplineInterpolator
    else:
        print("invalid 'interpolation' parameters.")
        interpolator = LinearInterpolator

    return interpolator


def interpolate_5nm_cmfs_data(spd, interplation="linear"):
    interpolator = get_interpolater(interplation)

    temp = spd.copy()
    temp.interpolate(SpectralShape(interval=1), interpolator=interpolator)

    return temp


def make_5nm_cmfs_spd():
    """
    CVRLの5nmのSpectralPowerDistributionを作る。
    """
    cmfs_5nm_file = "./src_data/CIE1931_5nm_cvrl.csv"
    cmfs_5nm = np.loadtxt(cmfs_5nm_file, delimiter=',').T
    wavelength_5nm = cmfs_5nm[0]
    values_5nm = tstack((cmfs_5nm[1], cmfs_5nm[2], cmfs_5nm[3]))
    m_data = make_multispectral_format_data(
        wavelength_5nm, values_5nm, "CIE1931_5nm_data")
    cmfs_spd_5nm = MultiSpectralDistribution(m_data)

    return cmfs_spd_5nm


def compare_1nm_value_and_target_value(spd, org_cmfs):
    """
    各種変換方式と、公式の1nmのCMFSの誤差を比較
    """
    # intp_methods = ["linear", "sprague", "spline"]
    intp_methods = ["sprague", "spline"]
    spds = [interpolate_5nm_cmfs_data(spd, intp_method)
            for intp_method in intp_methods]
    wl = spds[0].wavelengths
    diffs = [org_cmfs - spds[idx].values for idx in range(len(intp_methods))]
    error_rates = np.array([diff / org_cmfs for diff in diffs])
    error_rates[np.isinf(error_rates)] = 0
    error_rates[np.isnan(error_rates)] = 0

    ax1 = pu.plot_1_graph(
        fontsize=20,
        figsize=(10, 8),
        linewidth=3,
        graph_title="Comparison of interpolation methods.",
        xlabel="Wavelength [nm]",
        ylabel='Error rate of y [%]',
        xlim=[490, 510],
        ylim=[-0.1, 0.1])

    for idx in range(len(intp_methods)):
        y = error_rates[idx, :, 2] * 100
        ax1.plot(wl, y, '-o', label=intp_methods[idx])
    plt.legend(loc='lower right')
    plt.show()


def make_1nm_step_cmfs_from_5nm_step():
    """
    http://cvrl.ioo.ucl.ac.uk/ の5nmデータから1nmデータ作る
    """
    cmfs_1nm_file = "./src_data/CIE1931_1nm_cvrl.csv"
    cmfs_1nm = np.loadtxt(cmfs_1nm_file, delimiter=',')
    cmfs_1nm_value = cmfs_1nm[:, 1:]

    cmfs_spd_5nm = make_5nm_cmfs_spd()
    compare_1nm_value_and_target_value(cmfs_spd_5nm, cmfs_1nm_value)


def calc_d65_white_xy():
    """
    とりあえず D65 White の XYZ および xy を求めてみる。
    """
    temperature = 6500 * 1.4388 / 1.4380
    d65_spd = make_day_light_by_calculation(temperature=temperature,
                                            interpolater=LinearInterpolator,
                                            interval=5)
    print(d65_spd)


def xyY_to_rgb_with_illuminant_c(xyY):
    """
    C光源のXYZ値をD65光源のRGB値に変換する
    """
    large_xyz = xyY_to_XYZ(xyY)
    illuminant_XYZ = C_WHITE
    illuminant_RGB = D65_WHITE
    chromatic_adaptation_transform = 'CAT02'
    xyz_to_rgb_matrix = sRGB_COLOURSPACE.XYZ_to_RGB_matrix
    rgb = XYZ_to_RGB(large_xyz, illuminant_XYZ,
                     illuminant_RGB, xyz_to_rgb_matrix,
                     chromatic_adaptation_transform)

    return rgb


def get_reiwa_color():
    """
    reiwa color を得たい。
    梅：3.4RP7.4/6.8
    菫：7.1P2.9/3
    桜：2.8RP8.8/2.7
    """
    ume = "3.4RP 7.4/6.8"
    sumire = "7.1P 2.9/3"
    sakura = "2.8RP 8.8/2.7"
    reiwa_munsell_colors = [ume, sumire, sakura]
    reiwa_xyY_colors = [munsell_colour_to_xyY(x)
                        for x in reiwa_munsell_colors]
    print(reiwa_xyY_colors)
    reiwa_rgb_colors = [xyY_to_rgb_with_illuminant_c(xyY)
                        for xyY in reiwa_xyY_colors]
    reiwa_rgb_colors = np.array([np.round((oetf_sRGB(rgb)) * 255)
                                 for rgb in reiwa_rgb_colors])
    print(reiwa_rgb_colors)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # check_cmfs()
    # modify_d65_csv()
    # modify_xyz_csv()
    # compare_cmfs_from_web()
    # compare_d65_spd_from_web()
    # plot_cmfs()
    # plot_d65()
    # make_1nm_step_cmfs_from_5nm_step()
    # calc_d65_white_xy()
    get_reiwa_color()

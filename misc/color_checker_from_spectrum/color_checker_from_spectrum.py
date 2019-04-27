#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
スペクトルからカラーチェッカー値を算出する
"""

import os
import numpy as np
from colour.colorimetry import STANDARD_OBSERVERS_CMFS
from colour.colorimetry.spectrum import SpectralShape
from colour import D_illuminant_relative_spd
import re
import matplotlib.pyplot as plt
import plot_utility as pu
from colour.temperature import CCT_to_xy_CIE_D
from colour import D_illuminant_relative_spd
from colour.utilities import numpy_print_options
from colour.algebra import SpragueInterpolator, LinearInterpolator


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
                                  interpolater=None):
    """
    計算でD光源を作る。

    interpolater: SpragueInterpolator or LinearInterpolator
    """
    xy = CCT_to_xy_CIE_D(temperature)
    spd = D_illuminant_relative_spd(xy)
    spd = spd.interpolate(SpectralShape(interval=1),
                          interpolator=interpolater)
    ret_value = np.dstack([spd.wavelengths, spd.values])

    return ret_value.reshape((len(spd.values), 2)).T


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # check_cmfs()
    # modify_d65_csv()
    # modify_xyz_csv()
    # compare_cmfs_from_web()
    # compare_d65_spd_from_web()
    # plot_cmfs()
    plot_d65()

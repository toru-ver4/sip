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

CIE1931 = 'CIE 1931 2 Degree Standard Observer'
CIE1964 = 'CIE 1964 10 Degree Standard Observer'
CIE2015_2 = 'CIE 2012 2 Degree Standard Observer'
CIE2015_10 = 'CIE 2012 10 Degree Standard Observer'

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
    src_file = "./src_data/d65_CIE_S_014-1_org.csv"
    dst_file = "./src_data/d65_CIE_S_014-1.csv"

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


def check_cmfs():
    shape = SpectralShape(300, 830, 1)
    print(STANDARD_OBSERVERS_CMFS[CIE1931].trim(shape))
    print(D_illuminant_relative_spd(D65_xy).trim(shape))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # check_cmfs()
    # modify_d65_csv()
    modify_xyz_csv()

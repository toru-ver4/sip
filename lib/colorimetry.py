#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Colorimetry
"""

import os
import numpy as np
from colour.colorimetry.spectrum import SpectralShape
from colour.algebra import SpragueInterpolator, LinearInterpolator
from colour.colorimetry import MultiSpectralDistribution
from colour.utilities import tstack

CIE1931 = 'CIE 1931 2 Degree Standard Observer'
CIE1964 = 'CIE 1964 10 Degree Standard Observer'
CIE2015_2 = 'CIE 2012 2 Degree Standard Observer'
CIE2015_10 = 'CIE 2012 10 Degree Standard Observer'


def _make_multispectral_format_data(wavelengths, values):
    dic = dict(zip(np.uint16(wavelengths).tolist(), values.tolist()))

    return dic


def load_colorchecker_spectrum(interpolator=SpragueInterpolator):
    """
    Babel Average 2012 のデータをロード。
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv = base_dir + "/data/babel_spectrum_2012.csv"
    data = np.loadtxt(csv, delimiter=',', skiprows=0)

    wavelength = data[0, :]
    values = tstack([data[x, :] for x in range(1, 25)])
    m_data = _make_multispectral_format_data(wavelength, values)
    color_checker_spd = MultiSpectralDistribution(m_data)

    if interpolator is not None:
        color_checker_spd.interpolate(SpectralShape(interval=1),
                                      interpolator=interpolator)

    return color_checker_spd


# def load_cmfs


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(load_colorchecker_spectrum())

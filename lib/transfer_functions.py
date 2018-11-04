#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Transfer Functions モジュール

## 概要
OETF/EOTF の管理。Colour Science for Python っぽく
名称から OETF/EOTF にアクセスできるようにする。
また、ビデオレベルと輝度レベルの相互変換もお手軽にできるようにする。

## 設計思想
In-Out は原則 [0:1] のレンジで行う。
別途、Global変数として最大輝度をパラメータとして持ち、
輝度⇔信号レベルの相互変換はその変数を使って行う。
"""

import os
import numpy as np
import colour
import plot_utility as pu
import matplotlib.pyplot as plt

# NAME
GAMMA24 = 'Gamma 2.4'
ST2084 = 'SMPTE ST2084'
HLG = 'BT.2100 HLG'
LOGC = 'ARRI LOG_C'
CANON_LOG3 = 'Cannon Log3'
VLOG = 'Panasonic VLog'
SLOG3 = "SONY S-Log3"

PEAK_LUMINANCE = {GAMMA24: 100, ST2084: 10000, HLG: 1000,
                  VLOG: 10, CANON_LOG3: 10, LOGC: 10, SLOG3: 10}


def oetf(x, name=GAMMA24):
    """
    輝度値[cd/m2]からOETFを算出して返す。

    Parameters
    ----------
    x : numeric or array_like
        scene luminance. range is [0:1]
    name : unicode
        GAMMA24, ST2084, HLG, ... and so on.

    Returns
    -------
    numeric or ndarray
        encoded video level.

    Examples
    --------
    >>> oetf(0.18, GAMMA24)
    0.5 付近の値
    >>> oetf(1.0, GAMMA24)
    1.0
    >>> oetf(0.01, ST2084)
    0.5 付近の値
    """

    if name == GAMMA24:
        y = x ** (1/2.4)
    elif name == HLG:
        y = colour.models.eotf_reverse_BT2100_HLG(x * 1000)
    elif name == ST2084:
        # fix me!
        y = colour.models.oetf_ST2084(x * 10000)
    else:
        raise ValueError("invalid transfer fucntion name")

    return y


def oetf_from_luminance(x, name=GAMMA24):
    """
    輝度値[cd/m2]からOETFを算出して返す。

    Parameters
    ----------
    x : numeric or array_like
        luminance. unit is [cd/m2].
    name : unicode
        GAMMA24, ST2084, HLG, ... and so on.

    Returns
    -------
    numeric or ndarray
        encoded video level.

    Examples
    --------
    >>> oetf_from_luminance(18, GAMMA24)
    0.5 付近の値
    >>> oetf_from_luminance(100, GAMMA24)
    1.0
    >>> oetf_from_luminance(100, ST2084)
    0.5 付近の値
    """
    return oetf(x / PEAK_LUMINANCE[name], name)


def eotf(x, name=GAMMA24):
    """
    Code Value から輝度値を算出する。
    返り値は[0:1]で正規化された値。

    Parameters
    ----------
    x : numeric or array_like
        code value. range is [0:1]
    name : unicode
        GAMMA24, ST2084, HLG, ... and so on.

    Returns
    -------
    numeric or ndarray
        decoded video level. range is [0:1]

    Examples
    --------
    >>> eotf(0.5, GAMMA24)
    0.18 付近の値
    >>> oetf(1.0, GAMMA24)
    1.0
    >>> oetf(0.5, ST2084)
    0.01 付近の値
    """
    if name == GAMMA24:
        y = x ** 2.4
    elif name == ST2084:
        # fix me!
        y = colour.models.eotf_ST2084(x) / PEAK_LUMINANCE[name]
    elif name == HLG:
        y = colour.models.eotf_BT2100_HLG(x) / PEAK_LUMINANCE[name]
    else:
        raise ValueError("invaid tranfer function name")

    return y


def eotf_to_luminance(x, name=GAMMA24):
    """
    Code Value から輝度値を算出する。
    戻り値は [cd/m2] の単位。

    Parameters
    ----------
    x : numeric or array_like
        code value. range is [0:1]
    name : unicode
        GAMMA24, ST2084, HLG, ... and so on.

    Returns
    -------
    numeric or ndarray
        encoded video level.

    Examples
    --------
    >>> eotf(0.5, GAMMA24)
    18 付近の値
    >>> oetf(1.0, GAMMA24)
    100
    >>> oetf(0.5, ST2084)
    100 付近の値
    """
    return eotf(x, name) * PEAK_LUMINANCE[name]


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    x = np.linspace(0, 1, 1024)
    g_name = GAMMA24
    x_luminance = x * PEAK_LUMINANCE[g_name]
    y = oetf_from_luminance(x_luminance, g_name)
    x2 = eotf_to_luminance(y, g_name)
    ax1 = pu.plot_1_graph()
    ax1.plot(x_luminance, x2)
    plt.show()

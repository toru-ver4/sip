#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import os
import numpy as np
import plot_utility as pu
import matplotlib.pyplot as plt


def get_bt2100_pq_curve(x, debug=False):
    """
    参考：ITU-R Recommendation BT.2100-0
    """
    m1 = 2610/16384
    m2 = 2523/4096 * 128
    c1 = 3424/4096
    c2 = 2413/4096 * 32
    c3 = 2392/4096 * 32

    x = np.array(x)
    bunsi = (x ** (1/m2)) - c1
    bunsi[bunsi < 0] = 0
    bunbo = c2 - (c3 * (x ** (1/m2)))
    luminance = (bunsi / bunbo) ** (1/m1)

    return luminance * 10000


def get_bt2100_hlg_curve(x, debug=False):
    """
    参考：ITU-R Recommendation BT.2100-0
    """
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073

    under = (x <= 0.5) * 4 * (x ** 2)
    over = (x > 0.5) * (np.exp((x - c) / a) + b)

    y = (under + over) / 12.0

    if debug:
        ax1 = pu.plot_1_graph()
        ax1.plot(x, y)
        plt.show()

    return y


def log_decoding_CanonLog(clog_ire):
    """
    copied from http://colour.readthedocs.io .
    """
    x = np.where(clog_ire < 0.0730597,
                 -(10 ** ((0.0730597 - clog_ire) / 0.529136) - 1) / 10.1596,
                 (10 ** ((clog_ire - 0.0730597) / 0.529136) - 1) / 10.1596)

    return x


def canon_log_oetf(x, plot=False):
    """
    Defines the *Canon Log* log encoding curve / opto-electronic transfer
    function. This function is based on log_encoding_CanonLog in Colour.

    Parameters
    ----------
    x : numeric or ndarray
        Linear data.

    Returns
    -------
    numeric or ndarray
        Canon Log normalized code value(10bit).

    Notes
    -----
    -   output *y* code value is converted to *normalized code value*
        as follows: `NCV = (IRE * (940 - 64) + 64 / 1023)` .

    Examples
    --------
    >>> linear_max = canon_log_eotf(1.0, plot=True)
    >>> x = np.linspace(0, 1, 1024)
    >>> y = canon_log_eotf(x * linear_max, plot=True)
    """

    clog_ire = np.where(x < log_decoding_CanonLog(0.0730597),
                        -(0.529136 * (np.log10(-x * 10.1596 + 1)) - 0.0730597),
                        0.529136 * np.log10(10.1596 * x + 1) + 0.0730597)

    code_value = (clog_ire * (940 - 64) + 64) / 1023

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Title",
                              graph_title_size=None,
                              xlabel="X Axis Label", ylabel="Y Axis Label",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3)
        ax1.plot(x, code_value, label="Canon Log OETF")
        plt.legend(loc='upper left')
        plt.show()

    return code_value


def canon_log_eotf(x, plot=False):
    """
    Defines the *Canon Log* log decoding curve / electro-optical transfer
    function. This function is based on log_decoding_CanonLog in Colour.

    Parameters
    ----------
    x : numeric or ndarray
        normalized code value. Interval is [0:1] .

    Returns
    -------
    numeric or ndarray
        Linear data y.

    Notes
    -----
    -   Input *x* code value is converted to *IRE* as follows:
        `IRE = (CV - 64) / (940 - 64)`.

    Examples
    --------
    >>> x = np.linspace(0, 1, 1024)
    >>> y = canon_log_eotf(x, plot=True)
    """

    # convert x to the 10bit code value
    code_value = x * 1023

    # convert code_value to ire value
    clog_ire = (code_value - 64) / (940 - 64)

    y = np.where(clog_ire < 0.0730597,
                 -(10 ** ((0.0730597 - clog_ire) / 0.529136) - 1) / 10.1596,
                 (10 ** ((clog_ire - 0.0730597) / 0.529136) - 1) / 10.1596)

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Title",
                              graph_title_size=None,
                              xlabel="X Axis Label", ylabel="Y Axis Label",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3)
        ax1.plot(x, y, label="Canon Log EOTF")
        plt.legend(loc='upper left')
        plt.show()

    return y


def _log_decoding_CanonLog2(clog2_ire):
    """
    copied from http://colour.readthedocs.io .
    """

    x = np.where(
        clog2_ire < 0.035388128,
        -(10 ** ((0.035388128 - clog2_ire) / 0.281863093) - 1) / 87.09937546,
        (10 ** ((clog2_ire - 0.035388128) / 0.281863093) - 1) / 87.09937546)

    return x


def canon_log2_oetf(x, plot=False):
    """
    Defines the *Canon Log 2* log encoding curve / opto-electronic transfer
    function. This function is based on log_encoding_CanonLog2 in Colour.

    Parameters
    ----------
    x : numeric or ndarray
        Linear data.

    Returns
    -------
    numeric or ndarray
        Canon Log2 normalized code value(10bit).

    Notes
    -----
    -   output *y* code value is converted to *normalized code value*
        as follows: `NCV = (IRE * (940 - 64) + 64 / 1023)` .

    Examples
    --------
    >>> linear_max = canon_log2_eotf(1.0, plot=True)
    >>> x = np.linspace(0, 1, 1024)
    >>> y = canon_log2_eotf(x * linear_max, plot=True)
    """

    clog2_ire = np.where(
        x < _log_decoding_CanonLog2(0.035388128),
        -(0.281863093 * (np.log10(-x * 87.09937546 + 1)) - 0.035388128),
        0.281863093 * np.log10(x * 87.09937546 + 1) + 0.035388128)

    code_value = (clog2_ire * (940 - 64) + 64) / 1023

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Title",
                              graph_title_size=None,
                              xlabel="X Axis Label", ylabel="Y Axis Label",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3)
        ax1.plot(x, code_value, label="Canon Log2 OETF")
        plt.legend(loc='upper left')
        plt.show()

    return code_value


def canon_log2_eotf(x, plot=False):
    """
    Defines the *Canon Log 2* log decoding curve / electro-optical transfer
    function. This function is based on log_decoding_CanonLog2 in Colour.

    Parameters
    ----------
    x : numeric or ndarray
        normalized code value. Interval is [0:1] .

    Returns
    -------
    numeric or ndarray
        Linear data y.

    Notes
    -----
    -   Input *x* code value is converted to *IRE* as follows:
        `IRE = (CV - 64) / (940 - 64)`.

    Examples
    --------
    >>> x = np.linspace(0, 1, 1024)
    >>> y = canon_log2_eotf(x, plot=True)
    """

    # convert x to the 10bit code value
    code_value = x * 1023

    # convert code_value to ire value
    clog2_ire = (code_value - 64) / (940 - 64)    

    y = np.where(
        clog2_ire < 0.035388128,
        -(10 ** ((0.035388128 - clog2_ire) / 0.281863093) - 1) / 87.09937546,
        (10 ** ((clog2_ire - 0.035388128) / 0.281863093) - 1) / 87.09937546)

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Title",
                              graph_title_size=None,
                              xlabel="X Axis Label", ylabel="Y Axis Label",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3)
        ax1.plot(x, y, label="Canon Log2 EOTF")
        plt.legend(loc='upper left')
        plt.show()

    return y


def _log_decoding_CanonLog3(clog3_ire):
    """
    copied from http://colour.readthedocs.io .
    """
    x = np.select(
        (clog3_ire < 0.04076162, clog3_ire <= 0.105357102,
         clog3_ire > 0.105357102),
        (-(10 ** ((0.069886632 - clog3_ire) / 0.42889912) - 1) / 14.98325,
         (clog3_ire - 0.073059361) / 2.3069815,
         (10 ** ((clog3_ire - 0.069886632) / 0.42889912) - 1) / 14.98325))

    return x


def canon_log3_oetf(x, plot=False):
    """
    Defines the *Canon Log 3* log encoding curve / opto-electronic transfer
    function. This function is based on log_encoding_CanonLog3 in Colour.

    Parameters
    ----------
    x : numeric or ndarray
        Linear data.

    Returns
    -------
    numeric or ndarray
        Canon Log3 normalized code value(10bit).

    Notes
    -----
    -   output *y* code value is converted to *normalized code value*
        as follows: `NCV = (IRE * (940 - 64) + 64 / 1023)` .

    Examples
    --------
    >>> linear_max = canon_log3_eotf(1.0, plot=True)
    >>> x = np.linspace(0, 1, 1024)
    >>> y = canon_log3_eotf(x * linear_max, plot=True)
    """
    clog3_ire = np.select(
        (x < _log_decoding_CanonLog3(0.04076162),
         x <= _log_decoding_CanonLog3(0.105357102),
         x > _log_decoding_CanonLog3(0.105357102)),
        (-(0.42889912 * (np.log10(-x * 14.98325 + 1)) - 0.069886632),
         2.3069815 * x + 0.073059361,
         0.42889912 * np.log10(x * 14.98325 + 1) + 0.069886632))

    code_value = (clog3_ire * (940 - 64) + 64) / 1023

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Title",
                              graph_title_size=None,
                              xlabel="X Axis Label", ylabel="Y Axis Label",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3)
        ax1.plot(x, code_value, label="Canon Log3 OETF")
        plt.legend(loc='upper left')
        plt.show()

    return code_value


def canon_log3_eotf(x, plot=False):
    """
    Defines the *Canon Log 3* log decoding curve / electro-optical transfer
    function. This function is based on log_decoding_CanonLog3 in Colour.

    Parameters
    ----------
    x : numeric or ndarray
        normalized code value. Interval is [0:1] .

    Returns
    -------
    numeric or ndarray
        Linear data y.

    Notes
    -----
    -   Input *x* code value is converted to *IRE* as follows:
        `IRE = (CV - 64) / (940 - 64)`.

    Examples
    --------
    >>> x = np.linspace(0, 1, 1024)
    >>> y = canon_log3_eotf(x, plot=True)
    """

    # convert x to the 10bit code value
    code_value = x * 1023

    # convert code_value to ire value
    ire = (code_value - 64) / (940 - 64)

    y = np.select(
        (ire < 0.04076162, ire <= 0.105357102,
         ire > 0.105357102),
        (-(10 ** ((0.069886632 - ire) / 0.42889912) - 1) / 14.98325,
         (ire - 0.073059361) / 2.3069815,
         (10 ** ((ire - 0.069886632) / 0.42889912) - 1) / 14.98325))

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Title",
                              graph_title_size=None,
                              xlabel="X Axis Label", ylabel="Y Axis Label",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3)
        ax1.plot(x, y, label="Canon Log3 EOTF")
        plt.legend(loc='upper left')
        plt.show()

    return y


def get_1dlut_from_cube_format(fname):
    """
    Get the 1d-lut data from *.cube* data.

    Parameters
    ----------
    fname : strings.
            filename of *.cobe* file.

    Returns
    -------
    ndarray
        lut data lut.

    Notes
    -----
    none.

    Examples
    --------
    >>> lut = get_1dlut_from_cube_format(fname="./data/hoge_data.cube")
    """

    lut = np.loadtxt(fname, delimiter=" ", skiprows=6)
    lut = lut.T[0, :]

    return lut


def plot_diff(a, b):
    """
    plot the difference between a and b.

    Parameters
    ----------
    a : ndarray
        data 1
    b : ndarray
        data 2

    Returns
    -------
    none.

    Notes
    -----
    none.

    Examples
    --------
    >>> a = np.linspace(0, 1, 1024)
    >>> b = np.arange(1024) / 1023
    >>> plot_diff(a, b)
    """
    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title="Diff",
                          graph_title_size=None,
                          xlabel="Code Value", ylabel="Difference (Linear)",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=(0, 1024),
                          ylim=None,
                          xtick=[x * 128 for x in range(0, 1024//128+1)],
                          ytick=None,
                          xtick_size=None, ytick_size=None,
                          linewidth=3)
    ax1.plot((a-b), '-o', label="diff")
    plt.legend(loc='upper left')
    plt.show()


def plot_log_graph():
    stops = 29
    sample_num = 1024

    # Canon Log1
    max_val1 = canon_log_eotf(1.0, plot=False)
    x = np.linspace(0, stops, sample_num)
    logx1 = (1 / 2**(x)) * max_val1
    clog1 = canon_log_oetf(logx1, plot=False)

    # Canon Log2
    max_val2 = canon_log2_eotf(1.0, plot=False)
    x = np.linspace(0, stops, sample_num)
    logx2 = (1 / 2**(x)) * max_val2
    clog2 = canon_log2_oetf(logx2, plot=False)

    # Canon Log3
    max_val3 = canon_log3_eotf(1.0, plot=False)
    x = np.linspace(0, stops, sample_num)
    logx3 = (1 / 2**(x)) * max_val3
    clog3 = canon_log3_oetf(logx3, plot=False)

    # ACEScct
    max_val_aces_cct = 65504
    x = np.linspace(0, stops, sample_num)
    logx_aces_cct = (1 / 2**(x)) * max_val_aces_cct
    aces_cct = aces_cct_oetf(logx_aces_cct, plot=False)

    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(16, 10),
                          graph_title="Characteristics of the OETF",
                          graph_title_size=None,
                          xlabel="Relative Stop from 18% Gray",
                          ylabel="10bit Code Value",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=(-10, 10),
                          ylim=(0, 1024),
                          xtick=[x for x in range(-10, 11)],
                          ytick=[x * 64 for x in range(1024//64 + 1)],
                          xtick_size=None, ytick_size=None,
                          linewidth=3)
    x_axis = np.log2(logx1 / 0.20)
    y_axis = clog1 * 1023
    ax1.plot(x_axis, y_axis, 'r-', label="Hoge Log")
    x_axis = np.log2(logx2 / 0.20)
    y_axis = clog2 * 1023
    ax1.plot(x_axis, y_axis, 'b-', label="Hoge Log2")
    x_axis = np.log2(logx3 / 0.20)
    y_axis = clog3 * 1023
    ax1.plot(x_axis, y_axis, 'g-', label="Hoge Log3")
    x_axis = np.log2(logx_aces_cct / aces_cct_oetf(0.18))
    y_axis = aces_cct * 1023
    ax1.plot(x_axis, y_axis, 'c-', label="ACEScct")
    plt.legend(loc='upper left')
    plt.show()


def aces_cct_oetf(x, plot=False):
    """
    Defines the *ACEScct* log decoding curve / electro-optical transfer
    function.

    reference : http://j.mp/S-2016-001_

    Parameters
    ----------
    x : numeric or ndarray
        normalized code value. Interval is [0:65504] .

    Returns
    -------
    numeric or ndarray
        Linear data y.

    Notes
    -----
    None

    Examples
    --------
    >>> x = np.linspace(0, 1, 1024)
    >>> y = aces_cct_eotf(x * 65504, plot=True)
    """
    y = np.select(
        (x <= 0.0078125,
         x > 0.0078125),
        (10.5402377416545 * x + 0.0729055341958355,
         (np.log2(x) + 9.72)/17.52))

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Title",
                              graph_title_size=None,
                              xlabel="X Axis Label", ylabel="Y Axis Label",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3)
        ax1.plot(x, y, label="ACEScct OETF")
        plt.legend(loc='upper left')
        plt.show()

    return y


def aces_cct_eotf(x, plot=False):
    """
    Defines the *ACEScct* log decoding curve / electro-optical transfer
    function. 

    reference : http://j.mp/S-2016-001_

    Parameters
    ----------
    x : numeric or ndarray
        normalized code value. Interval is [0:1] .

    Returns
    -------
    numeric or ndarray
        Linear data y.

    Notes
    -----
    None

    Examples
    --------
    >>> max_val = aces_cct_oetf(65504)
    >>> x = np.linspace(0, 1, 1024)
    >>> y = aces_cct_eotf(x * max_val, plot=True)
    """
    y = np.select(
        (x <= 0.155251141552511,
         x < (np.log2(65504) + 9.72) / 17.52,
         x >= (np.log2(65504) + 9.72) / 17.52),
        ((x - 0.0729055341958355) / 10.5402377416545,
         2 ** (x * 17.52 - 9.72),
         65504))

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Title",
                              graph_title_size=None,
                              xlabel="X Axis Label", ylabel="Y Axis Label",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3)
        ax1.plot(x, y, label="ACEScct EOTF")
        plt.legend(loc='upper left')
        plt.show()

    return y


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    x = np.linspace(0, 1, 1024)
    # max_val = canon_log3_eotf(1.0, plot=False)
    # clog3 = canon_log3_oetf(x * max_val, plot=False)
    # linear = canon_log3_eotf(x, plot=False)
    # print(max_val)
    # max_val = canon_log_eotf(1.0)
    # clog = canon_log_oetf(x * max_val, plot=True)
    # y = canon_log_eotf(x, plot=True)
    # print(max_val)

    # lut = get_1dlut_from_cube_format("./data/cl1.cube")
    # plot_diff(y, lut)
    plot_log_graph()
    # aces_cct = aces_cct_oetf(x * 65504, plot=True)
    # result = aces_cct_eotf(aces_cct, plot=True)

    # ax1 = pu.plot_1_graph(fontsize=20,
    #                       figsize=(10, 8),
    #                       graph_title="Title",
    #                       graph_title_size=None,
    #                       xlabel="X Axis Label", ylabel="Y Axis Label",
    #                       axis_label_size=None,
    #                       legend_size=17,
    #                       xlim=None,
    #                       ylim=None,
    #                       xtick=None,
    #                       ytick=None,
    #                       xtick_size=None, ytick_size=None,
    #                       linewidth=3)
    # ax1.plot(x * 65504, result, label="ACEScct EOTF")
    # plt.legend(loc='upper left')
    # plt.show()

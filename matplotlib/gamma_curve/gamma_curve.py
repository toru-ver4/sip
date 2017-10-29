#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import os
import numpy as np
import plot_utility as pu
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    x = np.linspace(0, 1, 1024)
    max_val = canon_log3_eotf(1.0, plot=False)
    clog3 = canon_log3_oetf(x * max_val, plot=False)
    linear = canon_log3_eotf(clog3, plot=False)
    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title="Diff",
                          graph_title_size=None,
                          xlabel="input (linear)", ylabel="output (linear)",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=None,
                          ylim=None,
                          xtick=None,
                          ytick=None,
                          xtick_size=None, ytick_size=None,
                          linewidth=3)
    ax1.plot(x * max_val, linear, '-o', label="linearity")
    plt.legend(loc='upper left')
    plt.show()

    print(max_val)
    # lut = get_1dlut_from_cube_format("./data/cl3.cube")
    # plot_diff(y, lut)

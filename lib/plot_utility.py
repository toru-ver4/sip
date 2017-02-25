#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 概要
plot補助ツール群

# 参考
* [matplotlibでグラフの文字サイズを大きくする](https://goo.gl/E5fLxD)
* [Customizing matplotlib](http://matplotlib.org/users/customizing.html)

"""

import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorsys


cycle_num = 6
v_offset = 0.2
s = np.arange(cycle_num) / (cycle_num - 1) * (1 - v_offset) + v_offset
s = s[::-1]

r_cycle = []
g_cycle = []
b_cycle = []

for s_val in s:
    r, g, b = colorsys.hsv_to_rgb(0.0, s_val, 0.9)
    color = "#{:02X}{:02X}{:02X}".format(np.uint8(np.round(r * 0xFF)),
                                         np.uint8(np.round(g * 0xFF)),
                                         np.uint8(np.round(b * 0xFF)))
    r_cycle.append(color)
    r, g, b = colorsys.hsv_to_rgb(0.3, s_val, 0.9)
    color = "#{:02X}{:02X}{:02X}".format(np.uint8(np.round(r * 0xFF)),
                                         np.uint8(np.round(g * 0xFF)),
                                         np.uint8(np.round(b * 0xFF)))
    g_cycle.append(color)
    r, g, b = colorsys.hsv_to_rgb(0.6, s_val, 0.9)
    color = "#{:02X}{:02X}{:02X}".format(np.uint8(np.round(r * 0xFF)),
                                         np.uint8(np.round(g * 0xFF)),
                                         np.uint8(np.round(b * 0xFF)))
    b_cycle.append(color)


def _set_common_parameters(fontsize, **kwargs):
    # font size
    # ---------------------------------------
    if fontsize:
        plt.rcParams["font.size"] = fontsize

    if 'tick_size' in kwargs and kwargs['tick_size']:
        plt.rcParams['xtick.labelsize'] = kwargs['tick_size']
        plt.rcParams['ytick.labelsize'] = kwargs['tick_size']

    if 'xtick_size' in kwargs and kwargs['xtick_size']:
        plt.rcParams['xtick.labelsize'] = kwargs['xtick_size']

    if 'ytick_size' in kwargs and kwargs['ytick_size']:
        plt.rcParams['ytick.labelsize'] = kwargs['ytick_size']

    if 'axis_label_size' in kwargs and kwargs['axis_label_size']:
        plt.rcParams['axes.labelsize'] = kwargs['axis_label_size']

    if 'graph_title_size' in kwargs and kwargs['graph_title_size']:
        plt.rcParams['axes.titlesize'] = kwargs['graph_title_size']

    if 'legend_size' in kwargs and kwargs['legend_size']:
        plt.rcParams['legend.fontsize'] = kwargs['legend_size']

    # plot style
    # ---------------------------------------
    if 'grid' in kwargs:
        if kwargs['grid']:
            plt.rcParams['axes.grid'] = True
        else:
            plt.rcParams['axes.grid'] = False
    else:
        plt.rcParams['axes.grid'] = True

    # line style
    # ---------------------------------------
    if 'linewidth' in kwargs and kwargs['linewidth']:
        plt.rcParams['lines.linewidth'] = kwargs['linewidth']

    if 'prop_cycle' in kwargs and kwargs['prop_cycle']:
        plt.rcParams['axes.prop_cycle'] = kwargs['prop_cycle']


def plot_1_graph(fontsize=12, **kwargs):
    _set_common_parameters(fontsize=fontsize, **kwargs)

    if 'figsize' in kwargs and kwargs['figsize']:
        figsize = kwargs['figsize']
    else:
        figsize = (12, 8)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)

    if 'xlim' in kwargs and kwargs['xlim']:
        ax1.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])

    if 'ylim' in kwargs and kwargs['ylim']:
        ax1.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])

    if 'graph_title' in kwargs and kwargs['graph_title']:
        ax1.set_title(kwargs['graph_title'])

    if 'xlabel' in kwargs and kwargs['xlabel']:
        ax1.set_xlabel(kwargs['xlabel'])

    if 'ylabel' in kwargs and kwargs['ylabel']:
        ax1.set_ylabel(kwargs['ylabel'])

    if 'xtick' in kwargs and kwargs['xtick']:
        ax1.set_xticks(kwargs['xtick'])

    if 'ytick' in kwargs and kwargs['ytick']:
        ax1.set_yticks(kwargs['ytick'])

    # Adjust the position
    # ------------------------------------
    fig.tight_layout()

    return ax1


def _check_hsv_space():
    """
    # 概要
    Linestyle で 明度が徐々か変わるやつを作りたいんだけど、
    HSVの値がイマイチ分からないのでプロットしてみる。
    """

    h_num = 11
    s_num = 11

    h = np.arange(h_num) / (h_num - 1)
    s = np.arange(s_num) / (s_num - 1)

    f, axarr = plt.subplots(h_num, s_num, sharex='col', sharey='row',
                            figsize=(16, 16))
    for idx in range(h_num * s_num):
        h_idx = idx % h_num
        v_idx = idx // h_num
        r, g, b = colorsys.hsv_to_rgb(h[h_idx], s[v_idx], 0.9)
        color = "#{:02X}{:02X}{:02X}".format(np.uint8(np.round(r * 0xFF)),
                                             np.uint8(np.round(g * 0xFF)),
                                             np.uint8(np.round(b * 0xFF)))
        axarr[v_idx, h_idx].add_patch(
            patches.Rectangle(
                (0, 0), 1.0, 1.0, facecolor=color
            )
        )
    plt.show()


if __name__ == '__main__':
    # _check_hsv_space()

    # sample code for plot_1_graph()
    # -------------------------------
    x = np.arange(1024) / 1023
    gamma_list = [1.0, 1.2, 1.5, 1.9, 2.4, 3.0]
    label_list = ["gamma " + str(x) for x in gamma_list]
    y_list = [x ** gamma for gamma in gamma_list]
    ax1 = plot_1_graph(fontsize=20,
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
                       linewidth=3,
                       prop_cycle=cycler(color=g_cycle))
    for y, label in zip(y_list, label_list):
        ax1.plot(x, y, label=label)
    plt.legend(loc='upper left')
    plt.show()
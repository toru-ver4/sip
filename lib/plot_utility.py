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
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


if __name__ == '__main__':
    # sample code for plot_1_graph()
    # -------------------------------
    x = np.arange(1024) / 1023
    y = x ** 2.2
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
                       linewidth=None)
    ax1.plot(x, y, label="Legend")
    plt.legend(loc='upper left')
    plt.show()

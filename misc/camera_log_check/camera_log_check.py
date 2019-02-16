#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
カメラログのプロット
"""

import os
import numpy as np
import transfer_functions as tf
import matplotlib.pyplot as plt
import plot_utility as pu


def get_log_scale_x(sample_num=1024, stops=20, exp_rate=2.0):
    """
    Camera Log の OETF を横軸Stops でプロットするための
    linear light の入力データ列を作る。

    x = np.linspace(1/(2**stops), 1.0, 1024)

    だと低階調側がスカスカになるので、power(x, exp_rate) してる。
    が、そうすると stops が狂うので最初の x_min で帳尻合わせをしてる。
    """
    x_min = (1/(2**stops)) ** (1/exp_rate)
    x_base = np.linspace(x_min, 1.0, 1024)
    x = x_base ** exp_rate

    return x


def plot_n_log_basic():
    reflection = False
    x = np.linspace(0, 1, 1024)
    y = tf.n_log_decoding(x, out_reflection=reflection)

    ax1 = pu.plot_1_graph(linewidth=3)
    ax1.plot(x, y, '-o', label="N-Log EOTF")
    plt.legend(loc='upper left')
    plt.show()

    x_max = tf.n_log_decoding(1.0, out_reflection=reflection)
    x = np.linspace(0, 1, 1024) * x_max
    y = tf.n_log_encoding(x, in_reflection=reflection)

    ax1 = pu.plot_1_graph(linewidth=3)
    ax1.plot(x, y, label="N-Log OETF")
    plt.legend(loc='upper left')
    plt.show()

    y2 = tf.n_log_decoding(y, out_reflection=reflection)

    ax1 = pu.plot_1_graph(linewidth=3)
    ax1.plot(x, y2, label="Linear")
    plt.legend(loc='upper left')
    plt.show()


def plot_n_log_stops():
    x_base = get_log_scale_x(sample_num=1024, stops=20, exp_rate=3.0)
    x_max = tf.n_log_decoding(1.0)
    x = x_base * x_max
    gray18_linear_light = 0.20

    y = tf.n_log_encoding(x) * 1023

    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title="N-Log Characteristics",
                          graph_title_size=None,
                          xlabel="Input linear light stops",
                          ylabel="10bit code value",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=[-8, 8],
                          ylim=[0, 1024],
                          xtick=[x for x in range(-8, 9)],
                          ytick=[x * 128 for x in range(8)],
                          xtick_size=None, ytick_size=None,
                          linewidth=2)
    ax1.plot(np.log2(x/gray18_linear_light), y, '-o', label="N-Log OETF")
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_n_log_basic()
    plot_n_log_stops()

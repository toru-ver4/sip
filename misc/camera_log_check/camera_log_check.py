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


def get_log_scale_x(sample_num=1024, x_max=1.0, stops=20):
    """
    Camera Log の OETF を横軸Stops でプロットするための
    linear light の入力データ列を作る。
    """
    x_min = x_max/(2**stops)
    exp_min = np.log2(x_min)
    exp_max = np.log2(x_max)
    exp = np.linspace(exp_min, exp_max, sample_num)
    x = 2 ** exp

    return x


def plot_log_basic(name=tf.FLOG, reflection=False):
    if name == tf.FLOG:
        encode_func = tf.f_log_encoding
        decode_func = tf.f_log_decoding
    elif name == tf.NLOG:
        encode_func = tf.n_log_encoding
        decode_func = tf.n_log_decoding
    elif name == tf.DLOG:
        encode_func = tf.d_log_encoding
        decode_func = tf.d_log_decoding
    else:
        raise ValueError("not supported log name")

    x = np.linspace(0, 1, 1024)
    y = decode_func(x, out_reflection=reflection)

    ax1 = pu.plot_1_graph(linewidth=3)
    ax1.plot(x, y, label=name + " EOTF")
    plt.legend(loc='upper left')
    plt.show()

    x_max = decode_func(1.0, out_reflection=reflection)
    x = np.linspace(0, 1, 1024) * x_max
    y = encode_func(x, in_reflection=reflection)

    ax1 = pu.plot_1_graph(linewidth=3)
    ax1.plot(x, y, label=name + " OETF")
    plt.legend(loc='upper left')
    plt.show()

    y2 = decode_func(y, out_reflection=reflection)

    ax1 = pu.plot_1_graph(linewidth=3)
    ax1.plot(x, y2, label="Linear")
    plt.legend(loc='upper left')
    plt.show()

    print(np.max(np.abs(y2 - x)))
    print(y2)


def plot_n_log_stops():
    x_max = tf.n_log_decoding(1.0)
    x = get_log_scale_x(sample_num=64, x_max=x_max, stops=20)

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


def plot_f_log_stops():
    x_max = tf.n_log_decoding(1.0)
    x = get_log_scale_x(sample_num=1024, x_max=x_max, stops=20)

    gray18_linear_light = 0.20

    y = tf.f_log_encoding(x) * 1023

    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title="F-Log Characteristics",
                          graph_title_size=None,
                          xlabel="Input linear light stops",
                          ylabel="10bit code value",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=[-8, 6],
                          ylim=[0, 1024],
                          xtick=[x for x in range(-8, 9)],
                          ytick=[x * 64 for x in range(17)],
                          xtick_size=None, ytick_size=None,
                          linewidth=3)
    ax1.plot(np.log2(x/gray18_linear_light), y, 'g-', label="F-Log OETF")
    plt.legend(loc='upper left')
    plt.show()


def plot_d_log_stops():
    x_max = tf.d_log_decoding(1.0)
    x = get_log_scale_x(sample_num=64, x_max=x_max, stops=20)
    gray18_linear_light = 0.20

    y = tf.d_log_encoding(x) * 1023

    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title="D-Log Characteristics",
                          graph_title_size=None,
                          xlabel="Exposure (f-stops).",
                          ylabel="10bit code value",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=[-8, 8],
                          ylim=[0, 1024],
                          xtick=[x for x in range(-8, 9)],
                          ytick=[x * 128 for x in range(9)],
                          xtick_size=None, ytick_size=None,
                          linewidth=3)
    ax1.plot(np.log2(x/gray18_linear_light), y, 'k-o', label="D-Log OETF")
    plt.legend(loc='upper left')
    plt.show()


def centerd_spins(ax):
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def plot_log_stops():
    oetf_list = [tf.LOGC, tf.SLOG3, tf.DLOG, tf.FLOG, tf.NLOG]
    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(16, 10),
                          graph_title="Characteristics of the camera log",
                          graph_title_size=None,
                          xlabel="Exposure [stops].",
                          ylabel="10bit code value",
                          axis_label_size=None,
                          legend_size=19,
                          xlim=[-8, 8],
                          ylim=[0, 1024],
                          xtick=[x for x in range(-8, 9)],
                          ytick=[x * 128 for x in range(9)],
                          xtick_size=None, ytick_size=None,
                          linewidth=3)
    # centerd_spins(ax1)
    for oetf_name in oetf_list:
        x_max = tf.MAX_VALUE[oetf_name]
        x = get_log_scale_x(sample_num=64, x_max=x_max, stops=20)
        x2 = x / 0.20 if oetf_name != tf.LOGC else x / 0.18
        y = tf.oetf(x / x_max, oetf_name) * 1023
        ax1.plot(np.log2(x2), y, '-o', label=oetf_name)

    plt.legend(loc='upper left')
    plt.savefig('camera_logs.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_n_log_basic()
    # plot_n_log_stops()
    # enc_param = np.array([0.0, 0.2, 1.0, 16.4231816006])
    # print(tf.n_log_encoding(enc_param))
    # plot_f_log_basic()
    # check = np.array([0.0, 0.2, 1.0, 8.09036097832])
    # print(tf.f_log_encoding(check))
    # plot_f_log_stops()
    # plot_log_basic(tf.DLOG, reflection=False)
    # plot_d_log_stops()
    plot_log_stops()

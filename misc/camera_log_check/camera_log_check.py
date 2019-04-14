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
# import test_pattern_generator2 as tpg
# import TyImageIO as tyio


ALOG_MAX = 45.826263

ALOG_DATA = [94.98672465, 94.98672465, 94.98672465, 96.00137331, 96.00137331,
             96.00137331, 96.00137331, 96.00137331, 97.00041199, 97.00041199,
             97.00041199, 97.99945068, 98.99848936, 99.99752804, 100.99656672,
             101.9956054, 103.99368276, 106.99079881, 108.98887617,
             113.00064088,
             116.99679561, 122.9910277, 129.99990845, 137.9922179,
             148.99725338,
             161.00132753, 174.98786908, 189.98905928, 205.98928817,
             222.98855573,
             241.00247196, 260.99885557, 280.99523919, 301.99066148,
             323.00169375,
             344.99615473, 368.00526436, 390.99876402, 413.99226368,
             438.00041199,
             460.99391165, 485.00205997, 508.99459831, 533.00274662,
             557.99432364,
             582.00247196, 605.9950103, 631.0021973, 654.99473564,
             679.00288396,
             704.01007095, 728.00260929, 753.00979629, 777.00233463,
             802.00952163,
             827.00109865, 851.00924697, 876.00082399, 900.0089723,
             925.00054932,
             949.00869764, 974.00027466, 998.00842298, 1023]


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
    # oetf_list = [tf.LOGC, tf.SLOG3, tf.DLOG, tf.FLOG, tf.NLOG]
    oetf_list = [tf.LOGC, tf.SLOG3_REF]
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
        x2 = x / 0.20 if oetf_name == tf.SLOG3 else x / 0.18
        y = tf.oetf(x / x_max, oetf_name) * 1023
        ax1.plot(np.log2(x2), y, '-o', label=oetf_name)

    # Plot A-Log
    alog = ALOG_DATA
    x = get_log_scale_x(sample_num=64, x_max=ALOG_MAX, stops=20)
    x2 = x / 0.18
    ax1.plot(np.log2(x2), alog, '-o', label="Astrodesign A-Log??")

    plt.legend(loc='upper left')
    plt.savefig('camera_logs.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


# def make_0_to_1023_dpx_1st_line():
#     """
#     左上に0～1023のRampを埋め込んだDPXファイルを作る。
#     """
#     img = np.zeros((1080, 1920, 3), dtype=np.uint16)
#     ramp_base = np.arange(1024)
#     ramp = np.dstack((ramp_base, ramp_base, ramp_base))
#     img[0:100, 0:1024, :] = ramp

#     attr = {"oiio:BitsPerSample": 10}
#     writer = tyio.TyWriter(img / 1023, "ramp.dpx", attr)
#     writer.write()


# def make_a_log_oetf_exr():
#     """
#     A-Log OETF の結果を取得するための画像を作る。
#     """
#     sample_num = 64
#     ramp = get_log_scale_x(sample_num=sample_num, x_max=ALOG_MAX, stops=20)

#     img = np.zeros((1080, 1920, 3), dtype=np.float)
#     ramp_base = ramp
#     ramp = np.dstack((ramp_base, ramp_base, ramp_base))
#     img[0:100, 0:sample_num, :] = ramp

#     writer = tyio.TyWriter(img, "ramp_oetf.exr")
#     writer.write()


# def read_alog_eotf():
#     fname = "A-Log_EOTF00216000.exr"
#     reader = tyio.TyReader(fname)
#     img = reader.read()
#     ramp = img[0:1, 0:1024, 0].copy().flatten()
#     x = np.arange(1024)

#     ax1 = pu.plot_1_graph(fontsize=20,
#                           graph_title="Astrodesign A-Log EOTF??",
#                           figsize=(16, 10),
#                           xlabel="Code Value(10bit)",
#                           ylabel="Linear value",
#                           linewidth=3)
#     ax1.plot(x, ramp, label="Astrodesign A-Log??")
#     plt.legend(loc='upper left')
#     plt.show()

#     # with open("a-log_eotf.csv", 'w') as f:
#     #     for idx, value in enumerate(ramp):
#     #         buf = "{:d}, {:f}\n".format(idx, value)
#     #         f.write(buf)


# def read_alog_oetf():
#     sample_num = 64
#     # fname = "a_log_oetf00216000.dpx"
#     # reader = tyio.TyReader(fname)
#     # img = reader.read()
#     # ramp = img[0:1, 0:sample_num, 0].copy().flatten()
#     ramp = ALOG_DATA
#     x = get_log_scale_x(sample_num=sample_num, x_max=ALOG_MAX, stops=20)
#     x2 = x / 0.18
#     ax1 = pu.plot_1_graph(fontsize=20,
#                           graph_title="Astrodesign A-Log OETF??",
#                           figsize=(16, 10),
#                           xlabel="Stops",
#                           ylabel="Code Value(10bit)",
#                           linewidth=3)
#     ax1.plot(np.log2(x2), ramp, '-o', label="Astrodesign A-Log??")
#     plt.legend(loc='upper left')
#     plt.show()


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
    # make_0_to_1023_dpx_1st_line()
    # read_alog_eotf()
    # make_a_log_oetf_exr()
    # read_alog_oetf()

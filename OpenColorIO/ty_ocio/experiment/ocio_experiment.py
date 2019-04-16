#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
OCIOに関する疑問点を解消するための各種検証をする。

"""

import os
import numpy as np
import plot_utility as pu
import matplotlib.pyplot as plt
import TyImageIO as tyio


def mono_to_color(mono_data):
    return np.dstack((mono_data, mono_data, mono_data))


def make_gray_scale_exr():
    """
    A-Log OETF の結果を取得するための画像を作る。
    """
    sample_num = 1024
    x_0_1 = mono_to_color(np.linspace(0, 1, sample_num))
    x_m1_2 = mono_to_color(np.linspace(-1, 2, sample_num))
    x_m10_10 = mono_to_color(np.linspace(-10, 10, sample_num))
    x_m100_100 = mono_to_color(np.linspace(-100, 100, sample_num))
    x_m1000_1000 = mono_to_color(np.linspace(-1000, 1000, sample_num))
    x_m10000_10000 = mono_to_color(np.linspace(-10000, 10000, sample_num))

    img = np.zeros((1080, 1920, 3), dtype=np.float)
    img[0:1, 0:sample_num, :] = x_0_1
    img[1:2, 0:sample_num, :] = x_m1_2
    img[2:3, 0:sample_num, :] = x_m10_10
    img[3:4, 0:sample_num, :] = x_m100_100
    img[4:5, 0:sample_num, :] = x_m1000_1000
    img[5:6, 0:sample_num, :] = x_m10000_10000

    writer = tyio.TyWriter(img, "ocio_exp_indata.exr")
    writer.write()


def read_exr_data(fname):
    reader = tyio.TyReader(fname)
    img = reader.read()

    return img


def plot_gray_scale_exr_0_1():
    sample_num = 1024
    x_0_1 = np.linspace(0, 1, sample_num)
    x_m1_2 = np.linspace(-1, 2, sample_num)
    x_m10_10 = np.linspace(-10, 10, sample_num)

    img = read_exr_data("./exp_min_0_max_1_out.exr")
    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(16, 10),
                          graph_title="1DLUTの範囲が0～1の場合の挙動",
                          xlabel="non-linear value (before EOTF)",
                          ylabel="linear value (after EOTF)",
                          xlim=[-0.5, 1.5],
                          linewitdh=3)
    ax1.plot(x_0_1, img[0, 0:sample_num, 0].flatten(), label="0～1")
    ax1.plot(x_m1_2, img[1, 0:sample_num, 0].flatten(), label="-1～2")
    ax1.plot(x_m10_10, img[2, 0:sample_num, 0].flatten(), label="-10～10")
    plt.legend(loc='upper left')
    fname = './figures/graph_0_1.png'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_gray_scale_exr_m1_2():
    sample_num = 1024
    x_0_1 = np.linspace(0, 1, sample_num)
    x_m1_2 = np.linspace(-1, 2, sample_num)
    x_m10_10 = np.linspace(-10, 10, sample_num)

    img = read_exr_data("./exp_min_m1_max_2_out.exr")
    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(16, 10),
                          graph_title="1DLUTの範囲が -1～2 の場合の挙動",
                          xlabel="non-linear value (before EOTF)",
                          ylabel="linear value (after EOTF)",
                          xlim=[-1.5, 2.5],
                          linewitdh=3)
    ax1.plot(x_m10_10, img[2, 0:sample_num, 0].flatten(), label="-10～10")
    ax1.plot(x_m1_2, img[1, 0:sample_num, 0].flatten(), label="-1～2")
    ax1.plot(x_0_1, img[0, 0:sample_num, 0].flatten(), label="0～1")
    plt.legend(loc='upper left')
    fname = './figures/graph_m1_2.png'
    plt.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_gray_scale_exr()
    # plot_gray_scale_exr_0_1()
    plot_gray_scale_exr_m1_2()

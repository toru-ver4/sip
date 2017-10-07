#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自宅のEV2736でHDR表示を行うためにLUTを作る。
ついでに汎用性のある HDR to SDR 変換についても検討
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import color_convert as cc
import plot_utility as pu


def st2084_test():
    """
    ST2084 の OETF と EOTF が合っていそうか確認
    """
    x = np.linspace(0, 1, 1024)
    oetf = cc.linear_to_pq(x)
    eotf = cc.pq_to_linear(x)

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
                          linewidth=3,
                          prop_cycle=None)
    ax1.plot(x, eotf, label="EOTF")
    ax1.plot(x, oetf, label="OETF")
    plt.legend(loc='upper left')
    plt.show()


def gen_pq_to_1886_lut(target_bright=300, plot=False):
    x = np.linspace(0, 1, 1024)

    # ST2084 to Linear
    st2084_eotf = cc.pq_to_linear(x) * 10000

    # Hard Clipping
    st2084_eotf[st2084_eotf > target_bright] = target_bright

    # Normalize
    st2084_eotf = st2084_eotf / np.max(st2084_eotf)

    # Linear to REC1886
    y = st2084_eotf ** (1/2.4)

    # Plot
    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Debug Information",
                              graph_title_size=None,
                              xlabel="Video Level (normalized)",
                              ylabel="Output Value (normalized)",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3,
                              prop_cycle=None)
        ax1.plot(x, st2084_eotf, label="Clipped ST2084")
        ax1.plot(x, y, label="PQ_TO_REC1886_LUT")
        plt.legend(loc='upper left')
        plt.show()

    return y


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # gen_pq_to_1886_lut(1000)

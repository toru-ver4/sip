#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCIO調査
"""

import os
import lut as tylut
import numpy as np
import plot_utility as pu
import matplotlib.pyplot as plt
import transfer_functions as ty

REF_WHITE_LUMINANCE = 100  # unit is [nits]
SPI1D_SAMPLE_NUM = 4096


def plot_shaper():
    fname = "./aces_1.0.3/luts/Dolby_PQ_1000_nits_Shaper_to_linear.spi1d"
    title = os.path.basename(fname)
    x = tylut.load_1dlut_spi_format(fname)[..., 0]
    sample_num = len(x)
    y = np.linspace(0, 1, sample_num)
    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title=title,
                          graph_title_size=None,
                          xlabel="Linear (log scale)", ylabel="Output",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=None,
                          ylim=None,
                          xtick=None,
                          ytick=None,
                          xtick_size=None, ytick_size=None,
                          linewidth=3,
                          minor_xtick_num=None,
                          minor_ytick_num=None)
    ax1.plot(np.log2(x/0.18), y)
    plt.show()


def plot_rrt():
    fname = "./aces_1.0.3/luts/Dolby_PQ_4000_nits_Shaper.RRT.P3-D60_ST2084__4000_nits_.spi3d"
    title = os.path.basename(fname)
    x = tylut.load_3dlut_spi_format(fname)[0]
    print(x.shape)
    grid_num = 65
    x2 = np.linspace(0, 1, grid_num)
    y = []
    for idx in range(grid_num):
        line_idx = (grid_num ** 2) * idx + (grid_num ** 1) * idx\
            + idx
        y.append(x[line_idx, 0])

    y = np.array(y)

    ax1 = pu.plot_1_graph(fontsize=20,
                          figsize=(10, 8),
                          graph_title=title,
                          graph_title_size=None,
                          xlabel="Input", ylabel="Output",
                          axis_label_size=None,
                          legend_size=17,
                          xlim=None,
                          ylim=(0, 1),
                          xtick=None,
                          ytick=None,
                          xtick_size=None, ytick_size=None,
                          linewidth=3,
                          minor_xtick_num=None,
                          minor_ytick_num=None)
    ax1.plot(x2, y)
    plt.show()


def gen_pq_to_linear_lut_for_oiio():
    x = np.linspace(0, 1, SPI1D_SAMPLE_NUM)
    y = ty.eotf_to_luminance(x, ty.ST2084) / REF_WHITE_LUMINANCE
    fname = "./ty_ocio/luts/ST2084_to_Linear_10000nits.spi1d"
    tylut.save_1dlut_spi_format(lut=y, filename=fname, min=0.0, max=1.0)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_shaper()
    # plot_rrt()
    gen_pq_to_linear_lut_for_oiio()
    # gen_linear_to_pq_lut_for_oiio()

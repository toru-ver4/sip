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
import test_pattern_generator2 as tpg
import color_space as cs


REF_WHITE_LUMINANCE = 100  # unit is [nits]


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


def gen_pq_to_linear_lut_for_ocio():
    x = np.linspace(0, 1, 4096)
    y = ty.eotf_to_luminance(x, ty.ST2084) / REF_WHITE_LUMINANCE
    fname = "./ty_ocio/luts/ST2084_to_Linear_10000nits.spi1d"
    tylut.save_1dlut_spi_format(lut=y, filename=fname, min=0.0, max=1.0)


def gen_eotf_lut_for_ocio(name, sample_num=4096):
    """
    ocio向けの EOTF 1DLUT を作る。
    Matrix とかは別途用意してね！

    Parameters
    ----------
    name : strings
        the name of the gamma curve.
        select from **transfer_functions** module.
    sample_num : int
        sample number.

    Returns
    -------
        None.
    """
    x = np.linspace(0, 1, sample_num)
    y = ty.eotf_to_luminance(x, name) / REF_WHITE_LUMINANCE
    fname_base = "./ty_ocio/luts/{}_to_Linear.spi1d"
    fname = fname_base.format(name.replace(" ", "_"))
    tylut.save_1dlut_spi_format(lut=y, filename=fname, min=0.0, max=1.0)


def ocio_config_mtx_str(src_name, dst_name):
    mtx = cs.rgb2rgb_mtx(src_name, dst_name)
    a1 = ", ".join(map(str, mtx[0, :].tolist()))
    a2 = ", ".join(map(str, mtx[1, :].tolist()))
    a3 = ", ".join(map(str, mtx[2, :].tolist()))
    out_base = "{{matrix: [{:}, 0, {:}, 0, {:}, 0, 0, 0, 0, 1]}}"
    out_str = '- !<MatrixTransform> ' + out_base.format(a1, a2, a3)
    print(out_str)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_shaper()
    # plot_rrt()
    # gen_pq_to_linear_lut_for_ocio()
    # gen_linear_to_pq_lut_for_ocio()
    # gen_eotf_lut_for_ocio(ty.ST2084)
    # gen_eotf_lut_for_ocio(ty.LOGC)
    # gen_eotf_lut_for_ocio(ty.SLOG3)
    # gen_eotf_lut_for_ocio(ty.LOG3G10)
    # gen_eotf_lut_for_ocio(ty.GAMMA24)
    # gen_matrix_for_aces_ocio(ty.SLOG3)
    # print(cc.get_white_point_conv_matrix(cc.const_d60_large_xyz,
    #                                      cc.const_d65_large_xyz))
    ocio_config_mtx_str(cs.ACES_AP0, cs.BT709)
    ocio_config_mtx_str(cs.BT709, cs.ACES_AP0)

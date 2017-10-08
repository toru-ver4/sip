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
import common
from scipy import linalg


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


def gen_pq_to_sdr_lut(target_bright=300, sdr_gamma=2.4, plot=False):
    x = np.linspace(0, 1, 1024)

    # ST2084 to Linear
    st2084_eotf = cc.pq_to_linear(x) * 10000

    # Hard Clipping
    st2084_eotf[st2084_eotf > target_bright] = target_bright

    # Normalize
    st2084_eotf = st2084_eotf / np.max(st2084_eotf)

    # Linear to REC1886
    y = st2084_eotf ** (1/sdr_gamma)

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


def gen_pq2020_to_sdr_3dlut(target_bright=300, sdr_gamma=2.4):
    """
    ST2084 かつ 1000 or 300 nits かつ REC.2020 色域のディスプレイを
    Gamma2.2 かつ 300nits かつ REC.709 色域のディスプレイにマッピングする
    3DLUTを作成する
    """
    in_rgb = common.get_3d_grid_cube_format(33)
    
    # ST2084 to Linear
    lut = cc.pq_to_linear(in_rgb)

    # REC2020 gamut to REC709 gamut
    rec2020_to_xyz_mtx = cc.get_rgb_to_xyz_matrix(gamut=cc.const_rec2020_xy,
                                                  white=cc.const_d65_large_xyz)
    rec709_to_xyz_mtx = cc.get_rgb_to_xyz_matrix(gamut=cc.const_rec709_xy,
                                                 white=cc.const_d65_large_xyz)
    xyz_to_rec709_mtx = linalg.inv(rec709_to_xyz_mtx)
    rec2020_to_rec709_mtx = np.dot(rec2020_to_xyz_mtx, xyz_to_rec709_mtx)
    lut = cc.color_cvt(lut, rec2020_to_rec709_mtx)
    lut[lut > 1] = 1
    lut[lut < 0] = 0

    # Hard Clipping
    lut = lut * 10000
    lut[lut > target_bright] = target_bright

    # Normalize
    lut = lut / np.max(lut)

    # Linear to Power Gamma
    lut = lut ** (1/sdr_gamma)

    return lut


def out_1dlut_cube(lut, filename="out.cube"):
    """
    # brief
    output 1dlut at cube format.
    """
    with open(filename, 'w') as f:
        f.write("# DaVinci Resolve Cube (1D shaper LUT).\n")
        f.write("\n")
        f.write("LUT_1D_SIZE {}\n".format(lut.shape[0]))
        f.write("LUT_1D_INPUT_RANGE 0.0000000000 1.0000000000\n\n")
        
        for data in lut:
            f.write("{0} {0} {0}\n".format(data))


def out_3dlut_cube(lut, target_bright=300, sdr_gamma=2.4, filename="out.cube"):
    """
    # brief
    output 3dlut at cube format.
    """
    with open(filename, 'w') as f:
        f.write("# TORU YOSHIHARA SPECIAL LUT\n")
        f.write("\n")
        f.write("TITLE PQ{}_REC2020_TO_Gamma{}_REC709\n".format(target_bright,
                                                                sdr_gamma))
        f.write("LUT_3D_SIZE {:.0f}\n".format(lut.shape[1] ** (1/3)))
        f.write("\n")
        for data in lut[0]:
            f.write("{:.10f} {:.10f} {:.10f}\n".format(data[0], data[1], data[2]))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # gen_pq_to_1886_lut(1000)
    lut = gen_pq2020_to_sdr_3dlut(target_bright=300, sdr_gamma=2.2)
    out_3dlut_cube(lut, target_bright=300, sdr_gamma=2.2)

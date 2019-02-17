#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCIOマニュアルカスタムのためのパラメータを吐き出すよ。ゲロゲロ～。
"""

import os
import numpy as np
import colour
import color_convert as cc
from scipy import linalg
import imp
imp.reload(cc)


def get_to_aces_matrix_param(src_xy, src_white):
    dst_xy = cc.const_aces_ap0_xy
    dst_white = cc.const_d60_large_xyz
    white_conv_mtx = cc.get_white_point_conv_matrix(src_white, dst_white)
    src_to_xyz = cc.get_rgb_to_xyz_matrix(src_xy, src_white)
    dst_to_xyz = cc.get_rgb_to_xyz_matrix(dst_xy, dst_white)
    xyz_to_dst = linalg.inv(dst_to_xyz)

    mtx = xyz_to_dst.dot(white_conv_mtx.dot(src_to_xyz))
    print(mtx)

    src_color_space = colour.models.S_GAMUT3_COLOURSPACE
    dst_color_space = colour.models.ACES_2065_1_COLOURSPACE
    mtx = colour.RGB_to_RGB_matrix(src_color_space, dst_color_space,
                                   chromatic_adaptation_transform='CAT02')

    return mtx


if __name__ == '__main__':
    mtx = get_to_aces_matrix_param(src_xy=cc.const_s_gamut3_xy,
                                   src_white=cc.const_d65_large_xyz)
    print(mtx)

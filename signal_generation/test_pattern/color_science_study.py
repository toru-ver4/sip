#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BT.2407を実装するぞい！
あと色彩工学も勉強するぞい！
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import test_pattern_generator2 as tpg
import plot_utility as pu
import common as cmn
import colour
import sympy
import imp
imp.reload(tpg)


def lab_increment_data(sample_num=7):
    """
    ある壮大な計画に必要なデータの一部を生成する。
    CIELAB空間を斜めに切って、Chroma-Lightness平面を作り、
    そこで外壁の形をプロットしたい。それに必要なデータを作る。

    Parameters
    ----------
    sample_num : int
        sample number for each data.

    Returns
    -------
    main_data : array_like
        主で使うデータ
    sub_data : array_like
        副で使うデータ

    Example
    -------
    >>> lab_increment_data(sample_num=7)
    >>> main_data: [ 0.    0.25  0.5   0.75  1.    1.    1.    1.    1.  ]
    >>> sub_data:  [ 0.    0.    0.    0.    0.    0.25  0.5   0.75  1.  ]

    """
    if sample_num % 2 == 0:
        raise ValueError('"sample_num" must be odd number!')
    half_num = sample_num // 2 + 1
    main_data = np.ones(sample_num)
    main_data[:half_num] = np.linspace(0, 1, half_num)
    sub_data = (1 - main_data)[::-1]

    return main_data, sub_data


def judge(logic, if_true, if_false):
    if logic:
        return if_true
    else:
        return if_false


def rgbmyc_data_for_lab(sample_num=7):
    """
    ある壮大な計画に必要なデータの一部を生成する。
    CIELAB空間を斜めに切って、Chroma-Lightness平面を作り、
    そこで外壁の形をプロットしたい。それに必要なデータを作る。

    Parameters
    ----------
    sample_num : int
        sample number for each data.

    Returns
    -------
    data : array_like
        RGBMYCのLAB確認用データ。shape = sample_num x 6 x 3
    """
    base = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 0, 1), (1, 1, 0), (0, 1, 1)]
    main, sub = lab_increment_data(sample_num)
    data = []
    for element in base:
        data.append(np.dstack((judge(element[0], main, sub),
                               judge(element[1], main, sub),
                               judge(element[2], main, sub))))
    data = np.vstack(data)

    return data


def plot_lab_color_space(name='ITU-R BT.709', grid_num=17):
    data = cmn.get_3d_grid_cube_format(grid_num)

    illuminant_XYZ = tpg.D65_WHITE
    illuminant_RGB = tpg.D65_WHITE
    chromatic_adaptation_transform = 'CAT02'
    rgb_to_xyz_matrix = tpg.get_rgb_to_xyz_matrix(name)
    large_xyz = colour.RGB_to_XYZ(data, illuminant_RGB,
                                  illuminant_XYZ, rgb_to_xyz_matrix,
                                  chromatic_adaptation_transform)
    lab = colour.XYZ_to_Lab(large_xyz, illuminant_XYZ)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel("a*")
    ax.set_ylabel("b*")
    ax.set_zlabel("L*")
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])

    color_data = data.copy().reshape((grid_num**3, 3))

    ax.scatter(lab[..., 1], lab[..., 2], lab[..., 0], c=color_data)
    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # plot_lab_color_space('ITU-R BT.709', 33)
    # lab_increment_data(sample_num=9)
    print(rgbmyc_data_for_lab(sample_num=5))

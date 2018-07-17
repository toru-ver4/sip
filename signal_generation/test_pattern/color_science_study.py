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
    plot_lab_color_space('ITU-R BT.709', 33)

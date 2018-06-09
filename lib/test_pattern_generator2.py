#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
評価用のテストパターン作成ツール集

"""

import os
import cv2
import numpy as np
from colour.plotting import chromaticity_diagram_plot_CIE1931
from colour.colorimetry import CMFS, ILLUMINANTS
from colour.models import XYZ_to_xy, xy_to_XYZ, XYZ_to_RGB
from colour.utilities import normalise_maximum
from colour import models
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.ndimage.filters import convolve


def preview_image(img, order='rgb', over_disp=False):
    if order == 'rgb':
        cv2.imshow('preview', img[:, :, ::-1])
    elif order == 'bgr':
        cv2.imshow('preview', img)
    else:
        raise ValueError("order parameter is invalid")

    if over_disp:
        cv2.resizeWindow('preview', )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_chromaticity_diagram(antialiasing=True):
    samples = 512
    xx, yy\
        = np.meshgrid(np.linspace(0, 1, samples), np.linspace(1, 0, samples))
    xy = np.dstack((xx, yy))
    print(xy.shape)

    cmf = CMFS.get('CIE 1931 2 Degree Standard Observer')
    d65_white = ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
    cms_xy = XYZ_to_xy(cmf.values, d65_white)
    triangulation = Delaunay(cms_xy)

    # plt.figure()
    # plt.triplot(xy[:, 0], xy[:, 1], triangulation.simplices.copy(), '-o')
    # plt.title('triplot of Delaunay triangulation')
    # plt.show()
    # preview_image(xy)
    mask = (triangulation.find_simplex(xy) < 0).astype(np.float)
    if antialiasing:
        kernel = np.array([
            [0, 1, 0],
            [1, 2, 1],
            [0, 1, 0],
        ]).astype(np.float)
        kernel /= np.sum(kernel)
        mask = convolve(mask, kernel)
    mask = 1 - mask[:, :, np.newaxis]
    # print(mask)
    large_xyz = xy_to_XYZ(xy)

    # 対象カラースペースの設定
    # ------------------------
    color_space = models.BT2020_COLOURSPACE
    illuminant_XYZ = d65_white
    illuminant_RGB = color_space.whitepoint
    chromatic_adaptation_transform = 'CAT02'
    large_xyz_to_rgb_matrix = color_space.XYZ_to_RGB_matrix
    rgb = XYZ_to_RGB(large_xyz, illuminant_XYZ, illuminant_RGB,
                     large_xyz_to_rgb_matrix,
                     chromatic_adaptation_transform)
    rgb = normalise_maximum(rgb, axis=-1)
    rgb = rgb ** (1/2.2)
    mask_rgb = np.dstack((mask, mask, mask))
    # rgb[mask_rgb] = 0.0
    rgb = rgb * mask_rgb
    # preview_image(np.dstack((rgb, mask)))
    preview_image(rgb)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_chromaticity_diagram()

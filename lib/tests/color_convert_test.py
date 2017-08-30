#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# brief
test code
"""

import unittest
from nose.tools import ok_, eq_, raises
import numpy as np
import color_convert as ccv


class ColorConvertTestCase(unittest.TestCase):
    def test_get_rgb_to_xyz_matrix(self):
        expected = [[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]]
        result = ccv.get_rgb_to_xyz_matrix(gamut=ccv.const_sRGB_xy,
                                           white=ccv.const_d65_large_xyz)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(result.tolist()[i][j], expected[i][j])

    def test_color_cvt(self):
        mtx = ccv.get_rgb_to_xyz_matrix(gamut=ccv.const_sRGB_xy,
                                        white=ccv.const_d65_large_xyz)
        img = np.ones((2, 2, 3))
        """
        期待値の内訳は以下の通り。
        [0, 0] : min
        [0, 1] : max
        [1, 0] : 適当(YUV変換っぽいのをチョイス)
        [1, 1] : random
        """
        img[0, 0] = [0., 0., 0.]
        img[0, 1] = [1., 1., 1.]
        img[1, 0] = [0.3, 0.65, 0.05]
        img[1, 1] = [48303/0xFFFF, 23677/0xFFFF, 49530/0xFFFF]

        """
        ゴールデンデータはGoogleSpreadSheetで計算。
        なぜか小数第7位で誤差が出たので比較は第6位までにしてある。
        brucelindbloom のサイトでは 第7位まで合うんだけどね。
        """
        expected = [[[0., 0., 0.],
                     [0.95047, 1.0000001, 1.08883]],
                    [[0.36518326, 0.53225955, 0.130790175],
                     [0.5695625266, 0.4696761346, 0.7755330194]]]

        result = ccv.color_cvt(img, mtx)
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    self.assertAlmostEqual(result.tolist()[i][j][k],
                                           expected[i][j][k], 6)

    def test_is_inside_gamut(self):
        gamut = ccv.const_rec2020_xy
        ng_gamut_1 = np.array([0, 1, 2])
        ng_gamut_2 = np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7]])
        xy = np.array([[0.1, 0.3], [0.3, 0.3], [0.9, 0.3], [0.3, 0.5],
                       [0.708, 0.292], [0.170, 0.797], [0.131, 0.046],
                       [0.709, 0.292], [0.170, 0.798], [0.131, 0.045]])
        expected = np.array([False, True, False, True,
                             True, True, True,
                             False, False, False])
        result = ccv.is_inside_gamut(xy=xy, gamut=gamut)
        self.assertTrue(np.array_equal(result, expected))
        self.assertFalse(ccv.is_inside_gamut(xy=xy, gamut=xy))
        self.assertFalse(ccv.is_inside_gamut(xy=xy, gamut=ng_gamut_1))
        self.assertFalse(ccv.is_inside_gamut(xy=xy, gamut=ng_gamut_2))

    def test_lab_star_to_large_xyz_and_inverse(self):
        src_rgb_array = np.array([[255, 255, 255],
                                 [255, 192, 128],
                                 [4, 2, 8]]).reshape((1, 3, 3))
        dst_lab_array = np.array([[100.0, 0.0, 0.0],
                                 [82.9028, 17.3804, 41.2727],
                                 [0.0636, 0.2354, -0.5612]]).reshape((1, 3, 3))
        normalize_rgb = (src_rgb_array / 0xFF) ** 2.2
        large_xyz = ccv.rgb_to_large_xyz(rgb=normalize_rgb,
                                         gamut=ccv.const_sRGB_xy,
                                         white=ccv.const_d65_large_xyz)
        d65_to_d50_mtx\
            = ccv.get_white_point_conv_matrix(src=ccv.const_d65_large_xyz,
                                              dst=ccv.const_d50_large_xyz)
        large_xyz_d50 = ccv.color_cvt(large_xyz, d65_to_d50_mtx)
        lab_star = ccv.large_xyz_to_lab_star(large_xyz_d50,
                                             ccv.const_d50_large_xyz)

        for x, y in zip(lab_star.flatten(), dst_lab_array.flatten()):
            self.assertAlmostEqual(x, y, 4)

        # inverse
        # ----------------------------------
        large_xyz_d50 = ccv.lab_star_to_large_xyz(lab_star,
                                                  ccv.const_d50_large_xyz)
        d50_to_d65_mtx\
            = ccv.get_white_point_conv_matrix(src=ccv.const_d50_large_xyz,
                                              dst=ccv.const_d65_large_xyz)
        large_xyz_d65 = ccv.color_cvt(large_xyz_d50, d50_to_d65_mtx)
        rgb_lab = ccv.large_xyz_to_rgb(large_xyz=large_xyz_d65,
                                       gamut=ccv.const_sRGB_xy,
                                       white=ccv.const_d65_large_xyz)
        rgb_lab = (rgb_lab ** (1/2.2)) * 0xFF
        for x, y in zip(rgb_lab.flatten(), src_rgb_array.flatten()):
            self.assertAlmostEqual(x, y)

    def test_luv_star_to_large_xyz_and_inverse(self):
        src_rgb_array = np.array([[255, 255, 255],
                                 [255, 192, 128],
                                 [4, 2, 8]]).reshape((1, 3, 3))
        dst_luv_array = np.array([[100.0, 0.0, 0.0],
                                 [82.4637, 46.0138, 49.7742],
                                 [0.0678, 0.0113, -0.1911]]).reshape((1, 3, 3))
        normalize_rgb = (src_rgb_array / 0xFF) ** 2.2
        large_xyz = ccv.rgb_to_large_xyz(rgb=normalize_rgb,
                                         gamut=ccv.const_sRGB_xy,
                                         white=ccv.const_d65_large_xyz)
        luv_star = ccv.large_xyz_to_luv_star(large_xyz,
                                             ccv.const_d65_large_xyz)

        for x, y in zip(luv_star.flatten(), dst_luv_array.flatten()):
            self.assertAlmostEqual(x, y, 4)

        # inverse
        # ----------------------------------
        large_xyz = ccv.luv_star_to_large_xyz(luv_star,
                                              ccv.const_d65_large_xyz)
        rgb_luv = ccv.large_xyz_to_rgb(large_xyz=large_xyz,
                                       gamut=ccv.const_sRGB_xy,
                                       white=ccv.const_d65_large_xyz)
        rgb_luv = (rgb_luv ** (1/2.2)) * 0xFF
        for x, y in zip(rgb_luv.flatten(), src_rgb_array.flatten()):
            self.assertAlmostEqual(x, y)


if __name__ == '__main__':
    pass

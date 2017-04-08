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

    def test_lab_to_large_xyz(self):
        lab_data1 = np.array([42.101, 53.378, 28.19]).reshape((1, 1, 3))
        lab_data2 = np.array([96.539, -0.425, 1.186]).reshape((1, 1, 3))
        expected1 = np.array([21.6315, 12.5654, 3.8476])
        expected2 = np.array([87.8151, 91.3135, 73.9795])

        result1 = ccv.lab_to_large_xyz(lab_data1, ccv.const_d50_large_xyz)
        result2 = ccv.lab_to_large_xyz(lab_data2, ccv.const_d50_large_xyz)

        for i in range(3):
            self.assertAlmostEqual(result1[0][0][i], expected1[i], 4)
            self.assertAlmostEqual(result2[0][0][i], expected2[i], 4)


if __name__ == '__main__':
    pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# brief
test code
"""

import unittest
from nose.tools import ok_, eq_, raises
import numpy as np
import common


class CommontestCase(unittest.TestCase):
    def test_is_numpy_module(self):
        ok_(common.is_numpy_module(np.zeros((1))))
        ok_(not common.is_numpy_module(1))
        ok_(not common.is_numpy_module("1"))

    def test_is_correct_dtype(self):
        ok_types = {np.uint32, np.uint64}
        ok_(common.is_correct_dtype(np.ones((1), dtype=np.uint32), ok_types))
        ok_(common.is_correct_dtype(np.ones((1), dtype=np.uint64), ok_types))
        ok_(not common.is_correct_dtype(np.ones((1), dtype=np.int), ok_types))

    @raises(TypeError)
    def test_is_correct_dtype_exception_types(self):
        ng_types = [np.uint32, np.uint64]
        common.is_correct_dtype(np.ones((1), dtype=np.uint32),
                                ng_types)

    @raises(TypeError)
    def test_is_correct_dtype_exception_data(self):
        ok_types = {np.uint32, np.uint64}
        common.is_correct_dtype(1, ok_types)

    def test_equal_division(self):
        # normal pattern
        # -----------------
        result_no1 = [1] * 100
        eq_(common.equal_devision(100, 100), result_no1)
        result_no2 = [1] * 99 + [2]
        eq_(common.equal_devision(101, 100), result_no2)

        # minimum
        # ----------------
        result_no3 = [1]
        eq_(common.equal_devision(1, 1), result_no3)

        # abnormal
        # ---------------
        result_no4 = [0]
        eq_(common.equal_devision(0, 1), result_no4)
        result_no5 = [0] * 3
        eq_(common.equal_devision(0, 3), result_no5)

    def test_is_img_shape(self):
        data1 = np.ones((100, 100, 3))
        data2 = np.ones((100, 3))
        data3 = np.ones((100, 100, 3, 100))
        data4 = (100, 100, 3)
        ok_(common.is_img_shape(data1))
        ok_(not common.is_img_shape(data2))
        ok_(not common.is_img_shape(data3))
        with self.assertRaises(TypeError):
            common.is_img_shape(data4)

    def test_is_numpy_color_module(self):
        data1 = np.ones((100, 100, 3))
        data2 = np.ones((100, 3))
        data3 = np.ones((100, 100, 3, 100))
        data4 = (100, 100, 3)
        data5 = np.ones((3))
        data6 = [1, 1, 1]
        ok_(common.is_numpy_color_module(data1))
        ok_(common.is_numpy_color_module(data2))
        ok_(not common.is_numpy_color_module(data3))
        with self.assertRaises(TypeError):
            common.is_numpy_color_module(data4)
        ok_(common.is_numpy_color_module(data5))
        with self.assertRaises(TypeError):
            common.is_numpy_color_module(data6)


if __name__ == '__main__':
    pass

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# brief
test code
"""

import unittest
from nose.tools import ok_, raises
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


if __name__ == '__main__':
    pass

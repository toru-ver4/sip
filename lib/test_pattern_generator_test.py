#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# brief
test code
"""

import unittest
from nose.tools import eq_
import test_pattern_generator as tpg


class TpgtestCase(unittest.TestCase):
    def test_change_8bit_to_16bit(self):
        eq_(tpg.change_8bit_to_16bit(0), 0)
        eq_(tpg.change_8bit_to_16bit(255), 65280)
        eq_(tpg.change_8bit_to_16bit(256), 65536)

    def test_change_10bit_to_16bit(self):
        eq_(tpg.change_10bit_to_16bit(0), 0)
        eq_(tpg.change_10bit_to_16bit(1023), 0xFFC0)
        eq_(tpg.change_10bit_to_16bit(1024), 65536)

    def test_change_12bit_to_16bit(self):
        eq_(tpg.change_12bit_to_16bit(0), 0)
        eq_(tpg.change_12bit_to_16bit(4095), 0xFFF0)
        eq_(tpg.change_12bit_to_16bit(4096), 65536)

if __name__ == '__main__':
    pass

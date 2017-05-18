#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# brief
test code
"""

import unittest
from nose.tools import eq_, ok_
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

    def test_gen_step_gradation_v_1step(self):
        bit = 10
        color = (1.0, 1.0, 1.0)
        img = tpg.gen_step_gradation(width=50, height=1024, step_num=1025,
                                     bit_depth=bit, color=color,
                                     direction='v', debug=False)
        for c_idx in range(3):
            diff = img[1:, 0, c_idx] - img[0:-1, 0, c_idx]
            ref_val = int(round((2 ** (16 - bit)) * color[c_idx]))
            ok_((diff == ref_val).all())

    def test_gen_step_gradation_v_1step_2width(self):
        bit = 10
        color = (1.0, 1.0, 1.0)
        img = tpg.gen_step_gradation(width=50, height=2048, step_num=1025,
                                     bit_depth=bit, color=color,
                                     direction='v', debug=False)
        idx = [x * 2 + 1 for x in range(1024)]
        img = img[idx, :, :]
        for c_idx in range(3):
            diff = img[1:, 0, c_idx] - img[0:-1, 0, c_idx]
            ref_val = int(round((2 ** (16 - bit)) * color[c_idx]))
            ok_((diff == ref_val).all())

    def test_gen_step_gradation_h_1step(self):
        bit = 8
        color = (1.0, 1.0, 1.0)
        img = tpg.gen_step_gradation(width=256, height=50, step_num=257,
                                     bit_depth=bit, color=color,
                                     direction='h', debug=False)
        for c_idx in range(3):
            diff = img[0, 1:, c_idx] - img[0, 0:-1, c_idx]
            ref_val = int(round((2 ** (16 - bit)) * color[c_idx]))
            ok_((diff == ref_val).all())

    def test_gen_step_gradation_h_1step_4width(self):
        bit = 8
        color = (1.0, 1.0, 1.0)
        img = tpg.gen_step_gradation(width=1024, height=50, step_num=257,
                                     bit_depth=bit, color=color,
                                     direction='h', debug=False)
        idx = [x * 4 + 1 for x in range(256)]
        img = img[:, idx, :]
        for c_idx in range(3):
            diff = img[0, 1:, c_idx] - img[0, 0:-1, c_idx]
            ref_val = int(round((2 ** (16 - bit)) * color[c_idx]))
            ok_((diff == ref_val).all())

    def test_gen_step_gradation_h_17div_8bit(self):
        bit = 8
        color = (1.0, 1.0, 1.0)
        step_num = 17
        step_val = (2 ** bit) / (step_num - 1)
        img = tpg.gen_step_gradation(width=256+16, height=50,
                                     step_num=step_num,
                                     bit_depth=bit, color=color,
                                     direction='h', debug=False)
        idx = [x * 16 + 8 for x in range(17)]
        img_step = img[:, idx[0:-1], :]
        for c_idx in range(3):
            diff = img_step[0, 1:, c_idx] - img_step[0, 0:-1, c_idx]
            ref_val = int(round((2 ** (16 - bit)) * step_val * color[c_idx]))
            ref_val_last = ((2 ** bit) - 1) * (2 ** (16 - bit)) * color[c_idx]
            ref_val_last = int(round(ref_val_last))
            ok_((diff == ref_val).all())
            ok_((img[:, idx[-1], c_idx] == ref_val_last).all())

    def test_gen_step_gradation_h_17div_10bit(self):
        bit = 10
        color = (1.0, 1.0, 1.0)
        step_num = 17
        step_val = (2 ** bit) / (step_num - 1)
        img = tpg.gen_step_gradation(width=256+16, height=50,
                                     step_num=step_num,
                                     bit_depth=bit, color=color,
                                     direction='h', debug=False)
        idx = [x * 16 + 8 for x in range(17)]
        img_step = img[:, idx[0:-1], :]
        for c_idx in range(3):
            diff = img_step[0, 1:, c_idx] - img_step[0, 0:-1, c_idx]
            ref_val = int(round((2 ** (16 - bit)) * step_val * color[c_idx]))
            ref_val_last = ((2 ** bit) - 1) * (2 ** (16 - bit)) * color[c_idx]
            ref_val_last = int(round(ref_val_last))
            ok_((diff == ref_val).all())
            ok_((img[:, idx[-1], c_idx] == ref_val_last).all())


if __name__ == '__main__':
    pass

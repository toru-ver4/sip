#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SPI1D形式の1DLUTを作る
"""

import os
import numpy as np
import colour

VERSION = 1


def _get_st2084_eotf():
    sample_num = 1024
    x = np.linspace(0, 1, sample_num)
    y = colour.models.eotf_ST2084(x) / 10000.0

    return y


def make_spi1d(filename, data):
    data_min = np.min(data)
    data_max = np.max(data)
    data_len = data.shape[0]
    with open(filename, 'w') as f:
        f.write("Version {:d}\n".format(VERSION))
        f.write("From {:06f} {:06f}\n".format(data_min, data_max))
        f.write("Length {:d}\n".format(data_len))
        f.write("Components {:d}\n".format(1))
        f.write("{\n")
        for val in data:
            f.write("         {:01.13e}\n".format(val))
        f.write("}\n")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = _get_st2084_eotf()
    make_spi1d(filename="test.spi1d", data=data)

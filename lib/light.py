#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 概要
光に関するモジュール

# 使い方

"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def color_temp_to_small_xy(temperature):
    """
    # 概要
    色温度から xy座標を計算する

    # 注意事項
    temperature は numpy であること。1次元。
    """
    x_shita = 0.244063 + 0.09911 * (10 ** 3) / (temperature ** 1)\
        + 2.9678 * (10 ** 6) / (temperature ** 2)\
        - 4.607 * (10 ** 9) / (temperature ** 3)
    x_ue = 0.237040 + 0.24748 * (10 ** 3) / (temperature ** 1)\
        + 1.9018 * (10 ** 6) / (temperature ** 2)\
        - 2.0064 * (10 ** 9) / (temperature ** 3)
    x = x_ue * (temperature > 7000) + x_shita * (temperature <= 7000)
    y = -3.0000 * (x ** 2) + 2.870 * x - 0.275

    return x, y


def get_d_illuminants_coef():
    """
    # 概要
    D光源の算出に必要な係数(S0, S1, S2)を取得する
    """
    filename = 

if __name__ == '__main__':
    t = np.arange(4000, 10100, 100, dtype=np.float64)
    x, y = color_temp_to_small_xy(t)
    print(x)
    print(y)

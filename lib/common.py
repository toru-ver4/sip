#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 概要
どのモジュールからも使われそうな関数群

"""

import numpy as np
import math


def is_numpy_module(data):
    return type(data).__module__ == np.__name__


def is_correct_dtype(data, types={np.uint32, np.uint64}):
    """
    # brief
    for numpy instance only
    # note
    types must be a set.
    """
    if not is_numpy_module(data):
        raise TypeError("data must be a numpy instance")
    if not isinstance(types, set):
        raise TypeError("dtypes must be a set")

    return data.dtype.type in types


def equal_devision(length, div_num):
    """
    # 概要
    length を div_num で分割する。
    端数が出た場合は誤差拡散法を使って上手い具合に分散させる。
    """
    base = length / div_num
    ret_array = [base for x in range(div_num)]

    # 誤差拡散法を使った辻褄合わせを適用
    # -------------------------------------------
    diff = 0
    for idx in range(div_num):
        diff += math.modf(ret_array[idx])[0]
        if diff >= 1.0:
            diff -= 1.0
            ret_array[idx] = int(math.floor(ret_array[idx]) + 1)
        else:
            ret_array[idx] = int(math.floor(ret_array[idx]))

    # 計算誤差により最終点が +1 されない場合への対処
    # -------------------------------------------
    diff = length - sum(ret_array)
    if diff != 0:
        ret_array[-1] += diff

    # 最終確認
    # -------------------------------------------
    if length != sum(ret_array):
        raise ValueError("the output of equal_division() is abnormal.")

    return ret_array


if __name__ == '__main__':
    print(equal_devision(0, 3))
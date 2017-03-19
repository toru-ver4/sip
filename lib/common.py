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
    端数が出た場合は誤差拡散法っぽく良い感じにする。

    # 例(odd)
    equal_devision(9, 9)  => [1, 1, 1, 1, 1, 1, 1, 1, 1]
    equal_devision(10, 9) => [2, 1, 1, 1, 1, 1, 1, 1, 1]
    equal_devision(11, 9) => [2, 1, 1, 1, 2, 1, 1, 1, 1]
    equal_devision(12, 9) => [2, 1, 1, 1, 2, 1, 1, 1, 2]
    equal_devision(13, 9) => [2, 1, 2, 1, 2, 1, 1, 1, 2]
    equal_devision(14, 9) => [2, 1, 2, 1, 2, 1, 2, 1, 2]
    equal_devision(15, 9) => [2, 2, 2, 1, 2, 1, 2, 1, 2]
    equal_devision(16, 9) => [2, 2, 2, 2, 2, 1, 2, 1, 2]
    equal_devision(17, 9) => [2, 2, 2, 2, 2, 2, 2, 1, 2]

    # 例(even)
    equal_devision(10, 10) => [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    equal_devision(11, 10) => [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    equal_devision(12, 10) => [2, 1, 1, 1, 2, 1, 1, 1, 1, 1]
    equal_devision(13, 10) => [2, 1, 1, 1, 2, 1, 1, 1, 2, 1]
    equal_devision(14, 10) => [2, 1, 2, 1, 2, 1, 1, 1, 2, 1]
    equal_devision(15, 10) => [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
    equal_devision(16, 10) => [2, 2, 2, 1, 2, 1, 2, 1, 2, 1]
    equal_devision(17, 10) => [2, 2, 2, 2, 2, 1, 2, 1, 2, 1]
    equal_devision(18, 10) => [2, 2, 2, 2, 2, 2, 2, 1, 2, 1]
    equal_devision(19, 10) => [2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
    """
    base = length // div_num
    rest_val = length - (base * div_num)
    ret_array = [base] * div_num
    step = (div_num) // 2

    are = int(math.floor(math.log2(div_num)))
    are_list = [2 ** x for x in range(are)]
    print(are_list)


if __name__ == '__main__':
    equal_devision(17, 6)

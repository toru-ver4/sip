#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 概要
どのモジュールからも使われそうな関数群

"""

import numpy as np


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


if __name__ == '__main__':
    pass

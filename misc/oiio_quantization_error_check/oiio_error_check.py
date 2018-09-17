#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenImageIO で 10bit/12bit の DPXファイルを作成した際、
意図しない量子化誤差が発生しないか確認する。
"""

import os
import cv2
import OpenImageIO as oiio
import numpy as np
import matplotlib.pyplot as plt
import plot_utility as pu
import test_pattern_generator2 as tpg


def oiio_read_test(width=1024, height=768):
    """
    oiio でファイルを読む。

    分かったこと
    -----------
    read_image() の第一引数で型を指定できる。無指定だと float32 になる。
    """
    grad_10 = tpg.gen_step_gradation(width=width, height=height,
                                     step_num=1025, bit_depth=10,
                                     color=(1.0, 1.0, 1.0), direction='h')
    # tiff でテストデータ作成
    test_tiff = "test.tiff"
    cv2.imwrite(test_tiff, grad_10)

    # データの読み込みテスト
    img = oiio.ImageInput.open(test_tiff)
    img_spec = img.spec()
    img_width = img_spec.width
    img_height = img_spec.height
    img_channels = img_spec.nchannels
    img_data = img.read_image(oiio.INT16)
    print(img_width, img_height, img_channels)
    print(img_data)


def save_10bit_dpx(img, fname, attr=None):
    """
    10bit dpx形式で保存。

    Parameters
    ----------
    img : ndarray
        image data.
    fname : strings
        filename of the image.
    attr : ???
        attribute parameters for dpx.

    Returns
    -------
    -

    Examples
    --------
    >>> 
    >>> 
    >>> 
    """
    print(dir(oiio))


def _test_save_10bit_dpx(width=1024, height=768):
    grad_10 = tpg.gen_step_gradation(width=width, height=height,
                                     step_num=1025, bit_depth=10,
                                     color=(1.0, 1.0, 1.0), direction='h')
    save_10bit_dpx(grad_10, "test.dpx")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    oiio_read_test()
    _test_save_10bit_dpx()

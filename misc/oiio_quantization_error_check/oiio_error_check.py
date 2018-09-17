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
    img_input = oiio.ImageInput.open(test_tiff)
    if not img_input:
        raise Exception("Error: {}".format(oiio.geterror()))

    img_spec = img_input.spec()
    img_width = img_spec.width
    img_height = img_spec.height
    img_channels = img_spec.nchannels
    img_data = img_input.read_image(oiio.INT16)
    print(img_width, img_height, img_channels)
    print(img_data.shape)


def get_img_spec(img):
    xres = img.shape[1]
    yres = img.shape[0]
    nchannels = img.shape[2]

    return xres, yres, nchannels


def normalize(img):
    try:
        img_max_value = np.iinfo(img.dtype).max
    except:
        img_max_value = 1.0

    return np.double(img/img_max_value)


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

    xres, yres, nchannels = get_img_spec(img)
    img_out = oiio.ImageOutput.create(fname)
    if not img_out:
        raise Exception("Error: {}".format(oiio.geterror()))
    img_spec = oiio.ImageSpec(xres, yres, nchannels, oiio.UINT16)
    img_spec.attribute("oiio:BitsPerSample", 12)
    img_out.open(fname, img_spec)
    img_out.write_image(normalize(img))
    img_out.close()
    print(dir(img_out))
    print(dir(img_spec))
    print(img_spec.getattribute("oiio:BitsPerSample"))


def _test_save_10bit_dpx(width=1024, height=768):
    grad_10 = tpg.gen_step_gradation(width=width, height=height,
                                     step_num=1025, bit_depth=10,
                                     color=(1.0, 1.0, 1.0), direction='h')
    save_10bit_dpx(grad_10, "test12.dpx")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # oiio_read_test()
    _test_save_10bit_dpx()

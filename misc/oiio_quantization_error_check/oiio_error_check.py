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

    更に分かったこと
    -----------
    上記の情報は公式ドキュメントに普通に書いてある。
    まずはドキュメントを読もう。

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


def gen_out_img_spec(img, type_desk):
    xres = img.shape[1]
    yres = img.shape[0]
    nchannels = img.shape[2]
    img_spec = oiio.ImageSpec(xres, yres, nchannels, type_desk)

    return img_spec


def normalize_by_dtype(img):
    try:
        img_max_value = np.iinfo(img.dtype).max
    except:
        img_max_value = np.max(img)

    return np.double(img/img_max_value)


def np_img_to_oiio_type_desc(img):
    """
    numpy の image data から
    OIIO の TypeDesk 情報を得る。

    Parameters
    ----------
    img : ndarray
        image data.

    Returns
    -------
    TypeDesk
        a type desctipter for oiio module.
    """

    data_type = img.dtype.type

    if data_type == np.int8:
        return oiio.INT8
    if data_type == np.int16:
        return oiio.INT16
    if data_type == np.int32:
        return oiio.INT32
    if data_type == np.int64:
        return oiio.INT64
    if data_type == np.uint8:
        return oiio.UINT8
    if data_type == np.uint16:
        return oiio.UINT16
    if data_type == np.uint32:
        return oiio.UINT32
    if data_type == np.uint64:
        return oiio.UINT64
    if data_type == np.float16:
        return oiio.HALF
    if data_type == np.float32:
        return oiio.FLOAT
    if data_type == np.float64:
        return oiio.DOUBLE

    raise TypeError("unknown img format.")


def set_img_spec_attribute(img_spec, attr=None):
    """
    OIIO の ImageSpec に OIIO Attribute を設定する。

    Parameters
    ----------
    img_spec : OIIO ImageSpec
        specification of the image
    attr : oiio attribute
        attribute parameters for dpx.

    Returns
    -------
    -
    """

    if attr is None:
        return

    for key, value in attr.items():
        img_spec.attribute(key, value)


def save_img_using_oiio(img, fname, out_img_type_desc=oiio.UINT16, attr=None):
    """
    OIIO を使った画像保存。

    Parameters
    ----------
    img : ndarray
        image data.
    fname : strings
        filename of the image.
    out_img_type_desc : oiio.desc
        type descripter of img
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

    img_out = oiio.ImageOutput.create(fname)
    if not img_out:
        raise Exception("Error: {}".format(oiio.geterror()))
    out_img_spec = gen_out_img_spec(img, out_img_type_desc)
    set_img_spec_attribute(out_img_spec, attr)
    img_out.open(fname, out_img_spec)
    img_out.write_image(img)
    img_out.close()


def save_10bit_dpx(img, fname, out_img_type_desc=oiio.UINT16, attr=None):
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

    img_out = oiio.ImageOutput.create(fname)
    if not img_out:
        raise Exception("Error: {}".format(oiio.geterror()))
    out_img_spec = gen_out_img_spec(img, oiio.UINT16)
    out_img_spec.attribute("oiio:BitsPerSample", 12)
    img_out.open(fname, out_img_spec)
    img_out.write_image(normalize_by_dtype(img))
    img_out.close()
    print(dir(img_out))
    print(dir(out_img_spec))
    print(out_img_spec.getattribute("oiio:BitsPerSample"))


def _test_save_10bit_dpx(width=1024, height=768):
    grad_10 = tpg.gen_step_gradation(width=width, height=height,
                                     step_num=1025, bit_depth=10,
                                     color=(1.0, 1.0, 1.0), direction='h')
    save_10bit_dpx(grad_10, "test12.dpx")


def _test_save_various_format(width=1024, height=768):
    grad_10 = tpg.gen_step_gradation(width=width, height=height,
                                     step_num=1025, bit_depth=10,
                                     color=(1.0, 1.0, 1.0), direction='h')
    grad_10 = grad_10 / np.max(grad_10)

    attr_dpx = {"oiio:BitsPerSample": 12}
    save_img_using_oiio(grad_10, 'test_dpx_12bit.dpx',
                        out_img_type_desc=oiio.UINT16, attr=attr_dpx)

    attr_dpx = {"oiio:BitsPerSample": 10}
    save_img_using_oiio(grad_10, 'test_dpx_10bit.dpx',
                        out_img_type_desc=oiio.UINT16, attr=attr_dpx)

    attr_dpx = {"oiio:BitsPerSample": 16}
    save_img_using_oiio(grad_10, 'test_dpx_16bit.dpx',
                        out_img_type_desc=oiio.UINT16, attr=attr_dpx)

    save_img_using_oiio(grad_10, 'test_tiff_16bit.tiff',
                        out_img_type_desc=oiio.UINT16, attr=None)

    save_img_using_oiio(grad_10, 'test_tiff_8bit.tiff',
                        out_img_type_desc=oiio.UINT8, attr=None)

    save_img_using_oiio(grad_10, 'test_png_8bit.png',
                        out_img_type_desc=oiio.UINT8, attr=None)

    save_img_using_oiio(grad_10, 'test_png_16bit.png',
                        out_img_type_desc=oiio.UINT16, attr=None)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # oiio_read_test()
    # _test_save_10bit_dpx()
    _test_save_various_format()

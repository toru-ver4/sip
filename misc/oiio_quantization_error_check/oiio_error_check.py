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
import re

ATTR_PATTERN = re.compile("^(.*): (.*)$")


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
    attr : dictionary
        attribute parameters.

    Returns
    -------
    -
    """

    if attr is None:
        return

    for key, value in attr.items():
        if isinstance(value, list) or isinstance(value, tuple):
            img_spec.attribute(key, value[0], value[1])
        else:
            img_spec.attribute(key, value)


def get_img_spec_attribute(img_spec):
    """
    OIIO の ImageSpec から OIIO Attribute を取得する。

    Parameters
    ----------
    img_spec : OIIO ImageSpec
        specification of the image

    Returns
    -------
    attr : dictionary
        attribute parameters.
    """
    attr = {}
    for idx in range(len(img_spec.extra_attribs)):
        key = img_spec.extra_attribs[idx].name
        value = img_spec.extra_attribs[idx].value
        attr[key] = value
    return attr


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
    attr : dictionary
        attribute parameters.

    Returns
    -------
    -

    Examples
    --------
    see ```_test_save_various_format()```

    """

    img_out = oiio.ImageOutput.create(fname)
    if not img_out:
        raise Exception("Error: {}".format(oiio.geterror()))
    out_img_spec = gen_out_img_spec(img, out_img_type_desc)
    set_img_spec_attribute(out_img_spec, attr)
    img_out.open(fname, out_img_spec)
    img_out.write_image(img)
    img_out.close()


def load_img_using_oiio(fname):
    """
    OIIO を使った画像読込。

    Parameters
    ----------
    fname : strings
        filename of the image.

    Returns
    -------
    img : ndarray
        image data.
    attr : dictionary
        attribute parameters for dpx.

    Examples
    --------
    see ```_test_load_various_format()```

    """
    # データの読み込みテスト
    img_input = oiio.ImageInput.open(fname)
    if not img_input:
        raise Exception("Error: {}".format(oiio.geterror()))

    img_spec = img_input.spec()
    attr = get_img_spec_attribute(img_spec)
    typedesc = img_spec.format
    img_data = img_input.read_image(typedesc)

    return img_data, attr


def timecode_str_to_bcd(time_code_str):
    """
    '01:23:45:12' のようなタイムコードの文字列表記を
    0x01234512 に変換する。

    Examples
    --------
    >>> bcd = timecode_str_to_bcd(time_code_str='01:23:45:12')
    >>> print("0x{:08X}".format(bcd))
    0x12345612
    """
    temp_str = time_code_str.replace(':', '')
    if len(temp_str) != 8:
        raise TypeError('invalid time code str!')

    bcd = 0
    for idx in range(len(temp_str)):
        bcd += int(temp_str[idx]) << (8 - idx - 1) * 4

    return (int(bcd), int(0))


def _test_save_various_format(width=1024, height=768):
    grad_10 = tpg.gen_step_gradation(width=width, height=height,
                                     step_num=1025, bit_depth=10,
                                     color=(1.0, 1.0, 1.0), direction='h')
    grad_10 = grad_10 / np.max(grad_10)

    # attr_dpx = {"oiio:BitsPerSample": 12}
    # save_img_using_oiio(grad_10, 'test_dpx_12bit.dpx',
    #                     out_img_type_desc=oiio.UINT16, attr=attr_dpx)

    # timecode = '01:23:45:12'
    # attr_dpx = {"oiio:BitsPerSample": 10,
    #             'dpx:TimeCode': timecode,
    #             'dpx:UserBits': 0,
    #             'dpx:FrameRate': 24.0,
    #             'dpx:TemporalFrameRate': 24.0,
    #             'dpx:TimeOffset': 0.0,
    #             'dpx:BlackLevel': 64,
    #             'dpx:BlackGain': 0.0,
    #             'dpx:BreakPoint': 0.0,
    #             'dpx:WhiteLevel': 940}
    # save_img_using_oiio(grad_10, 'test_dpx_10bit.dpx',
    #                     out_img_type_desc=oiio.UINT16, attr=attr_dpx)

    # OpenEXR へのタイムコード埋め込みは失敗。どうすればいいの？
    timecode = '02:31:59:19'
    timecode_bcd = timecode_str_to_bcd(timecode)
    attr_openexr = {'smpte:TimeCode': [oiio.TypeDesc.TypeTimeCode, timecode_bcd]}
    save_img_using_oiio(grad_10, 'test_dpx_10bit.exr',
                        out_img_type_desc=oiio.UINT16, attr=attr_openexr)

    # attr_dpx = {"oiio:BitsPerSample": 16}
    # save_img_using_oiio(grad_10, 'test_dpx_16bit.dpx',
    #                     out_img_type_desc=oiio.UINT16, attr=attr_dpx)

    # save_img_using_oiio(grad_10, 'test_tiff_16bit.tiff',
    #                     out_img_type_desc=oiio.UINT16, attr=None)

    # save_img_using_oiio(grad_10, 'test_tiff_8bit.tiff',
    #                     out_img_type_desc=oiio.UINT8, attr=None)

    # save_img_using_oiio(grad_10, 'test_png_8bit.png',
    #                     out_img_type_desc=oiio.UINT8, attr=None)

    # save_img_using_oiio(grad_10, 'test_png_16bit.png',
    #                     out_img_type_desc=oiio.UINT16, attr=None)


def _test_load_various_format():
    # img, attr = load_img_using_oiio(fname='test_exr01087116.exr')
    img, attr = load_img_using_oiio(fname='test_dpx_10bit.exr')
    # img, attr = load_img_using_oiio(fname='test_dpx_10bit.dpx')
    # img, attr = load_img_using_oiio(fname='Untitled00120612.dpx')
    print(attr)


def test_10bit_error():
    # 1024x1024 のRampパターン作成
    inc10 = np.arange(1024)
    img_org = np.dstack([inc10, inc10, inc10])
    img_org = img_org * np.ones((1024, 1, 3))  # V方向に拡張
    img_normalized = img_org / np.max(img_org)

    # 保存
    fname = 'inc10.dpx'
    attr_dpx = {"oiio:BitsPerSample": 10}
    save_img_using_oiio(img_normalized, fname,
                        out_img_type_desc=oiio.UINT16, attr=attr_dpx)

    # 読み込み
    img_load, attr = load_img_using_oiio(fname)
    img_load_10bit = np.uint16(np.round(normalize_by_dtype(img_load) * 1023))

    # とりあえずプロット
    ax1 = pu.plot_1_graph()
    ax1.plot(img_load_10bit[0, :, 0])
    plt.show()

    # オリジナルデータとの差分確認
    diff = np.sum(np.abs(img_org - img_load_10bit))
    print(diff)

    # 隣接ピクセルとの差分確認
    line_data = img_load_10bit[0, :, 0]
    diff = line_data[1:] - line_data[:-1]
    print(np.sum(diff != 1))


def test_12bit_error():
    # 1024x1024 のRampパターン作成
    inc12 = np.arange(4096)
    img_org = np.dstack([inc12, inc12, inc12])
    img_org = img_org * np.ones((2160, 1, 3))  # V方向に拡張
    img_normalized = img_org / np.max(img_org)

    # 保存
    fname = 'inc12.dpx'
    attr_dpx = {"oiio:BitsPerSample": 12}
    save_img_using_oiio(img_normalized, fname,
                        out_img_type_desc=oiio.UINT16, attr=attr_dpx)

    # 読み込み
    img_load, attr = load_img_using_oiio(fname)
    img_load_12bit = np.uint16(np.round(normalize_by_dtype(img_load) * 4095))

    # とりあえずプロット
    ax1 = pu.plot_1_graph()
    ax1.plot(img_load_12bit[0, :, 0])
    plt.show()

    # オリジナルデータとの差分確認
    diff = np.sum(np.abs(img_org - img_load_12bit))
    print(diff)

    # 隣接ピクセルとの差分確認
    line_data = img_load_12bit[0, :, 0]
    diff = line_data[1:] - line_data[:-1]
    print(np.sum(diff != 1))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # _test_save_various_format()
    # _test_load_various_format()
    # test_10bit_error()
    test_12bit_error()

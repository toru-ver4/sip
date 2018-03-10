#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HDR用のテストパターンを作る
"""

import os
import cv2
import numpy as np
import test_pattern_generator as tpg
import colour
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imp
imp.reload(tpg)

SIDE_V_GRADATION_WIDTH = 0.03
SIDE_V_GRADATION_TEXT_WIDTH = 0.058
SIDE_V_GRADATION_TEXT_H_OFFFSET = 0.03
H_GRADATION_HEIGHT = 0.05


def gen_video_level_text_img(width=1024, height=768,
                             font_size=15, font_color=(0.5, 0.5, 0.5),
                             text_info=[["Video Level", "Brightness"]]):
    """
    テキスト情報が付与された画像データを生成する。

    Parameters
    ----------
    width : numeric
        width of the text image.
    height : numeric
        height of the text image.
    font_size : integer
        font size
    font_color : tuple of numeric
        font color. (red, green, blue)
    text_info : array_like
        array of [[video level, brightness], ...].

    Returns
    -------
    ndarray
        text info image.
    """
    fg_color = tuple([int(x * 0xFF) for x in font_color])
    v_offset = height / text_info.shape[0]
    st_pos_h = width * SIDE_V_GRADATION_TEXT_H_OFFFSET
    st_pos_v = 0
    txt_img = Image.new("RGB", (width, height), (0x00, 0x00, 0x00))
    draw = ImageDraw.Draw(txt_img)
    font = ImageFont.truetype("./fonts/NotoSansMonoCJKjp-Regular.otf",
                              font_size)
    for idx, text_pair in enumerate(text_info):
        pos = (st_pos_h, int(st_pos_v + v_offset * idx))
        if text_pair[1] < 999.99999:
            text_data = "{:>4.0f},{:>7.1f} nits".format(text_pair[0],
                                                        text_pair[1])
        else:
            text_data = "{:>4.0f},{:>6.0f}  nits".format(text_pair[0],
                                                         text_pair[1])
        draw.text(pos, text_data, font=font, fill=fg_color)

    txt_img = (np.asarray(txt_img) * 0x100).astype(np.uint16)

    return txt_img


def composite_pq_vertical_gray_scale(img):
    """
    execute the composition processing for the virtical pq gradation.

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    Returns
    -------
    ndarray
        a image with pq gray scale.

    Notes
    -----
    -

    Examples
    --------
    >>> img = np.zeros((1080, 1920, 3), np.dtype=uint8)
    >>> composite_pq_vertical_gray_scale(img)
    """
    # 基本情報作成
    # ------------------------------------------------------
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    scale_width = int(img_width * SIDE_V_GRADATION_WIDTH)
    scale_height = img_height - 2  # "-2" is for pixels of frame.
    text_width = int(img_width * SIDE_V_GRADATION_TEXT_WIDTH)
    text_height = img_height - 2  # "-2" is for pixels of frame.
    bit_depth = 10
    video_max = (2 ** bit_depth) - 1
    step_num = 65

    # PQ カーブのグラデーション作成
    # ------------------------------------------------------
    scale = tpg.gen_step_gradation(width=scale_width, height=scale_height,
                                   step_num=step_num, color=(1.0, 1.0, 1.0),
                                   direction='v', bit_depth=bit_depth)
    img[0+1:scale_height+1, 0+1:scale_width+1] = scale

    # ビデオレベルと明るさを表示
    # ------------------------------------------------------
    video_level = [x * (2 ** bit_depth) // (step_num - 1)
                   for x in range(step_num)]
    video_level[-1] -= 1  # 最終データは1多いので引いておく
    video_level_float = np.array(video_level) / video_max
    bright = colour.eotf(video_level_float, 'ITU-R BT.2100 PQ')
    text_info = np.dstack((video_level, bright)).reshape((bright.shape[0], 2))
    font_size = int(text_height / step_num / 96 * 72)
    txt_img = gen_video_level_text_img(width=text_width, height=text_height,
                                       font_size=font_size,
                                       text_info=text_info)
    img[0+1:text_height+1, scale_width:scale_width+text_width, :] = txt_img


def composite_hlg_vertical_gray_scale(img):
    """
    execute the composition processing for the virtical hlg gradation.

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    Returns
    -------
    ndarray
        a image with pq gray scale.

    Notes
    -----
    -

    Examples
    --------
    >>> img = np.zeros((1080, 1920, 3), np.dtype=uint8)
    >>> composite_hlg_vertical_gray_scale(img)
    """
    # 基本情報作成
    # ------------------------------------------------------
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    scale_width = int(img_width * SIDE_V_GRADATION_WIDTH)
    scale_height = img_height - 2  # "-2" is for pixels of frame.
    text_width = int(img_width * SIDE_V_GRADATION_TEXT_WIDTH)
    text_height = img_height - 2  # "-2" is for pixels of frame.
    bit_depth = 10
    video_max = (2 ** bit_depth) - 1
    step_num = 65

    # HLGのグラデーション作成
    # ------------------------------------------------------
    scale = tpg.gen_step_gradation(width=scale_width, height=scale_height,
                                   step_num=step_num, color=(1.0, 1.0, 1.0),
                                   direction='v', bit_depth=bit_depth)
    h_st = img_width - 1 - scale_width
    h_ed = -1
    img[0+1:scale_height+1, h_st:h_ed] = scale

    # ビデオレベルと明るさを表示
    # ------------------------------------------------------
    video_level = [x * (2 ** bit_depth) // (step_num - 1)
                   for x in range(step_num)]
    video_level[-1] -= 1  # 最終データは1多いので引いておく
    video_level_float = np.array(video_level) / video_max
    bright = colour.eotf(video_level_float, 'ITU-R BT.2100 HLG',
                         L_W=1000, gamma=1.2)
    text_info = np.dstack((video_level, bright)).reshape((bright.shape[0], 2))
    font_size = int(text_height / step_num / 96 * 72)
    txt_img = gen_video_level_text_img(width=text_width, height=text_height,
                                       font_size=font_size,
                                       text_info=text_info)
    h_st = img_width - (text_width + scale_width)
    h_ed = h_st + text_width
    img[0+1:text_height+1, h_st:h_ed, :] = txt_img


def composite_8bit_middle_gray_scale(img):
    """
    execute the composition processing for the horizontal 8bit gradation.

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    Returns
    -------
    ndarray
        a image with 8bit horizontal gray scale.

    Notes
    -----
    -

    Examples
    --------
    >>> img = np.zeros((1080, 1920, 3), np.dtype=uint8)
    >>> composite_8bit_middle_gray_scale(img)
    """
    img_width = img.shape[1]
    img_height = img.shape[0]

    # 
    if img_width <= 2048:
        grad_width = 2048
    else:
        grad_width = 4096
    grad_height = int(img_height * H_GRADATION_HEIGHT)

    grad_8 = tpg.gen_step_gradation(width=grad_width*2, height=grad_height,
                                    step_num=257, bit_depth=8,
                                    color=(1.0, 1.0, 1.0), direction='h')

    grad_10 = tpg.gen_step_gradation(width=grad_width*2, height=grad_height,
                                     step_num=257, bit_depth=8,
                                     color=(1.0, 1.0, 1.0), direction='h')


def composite_10bit_middle_gray_scale(img):
    """
    execute the composition processing for the horizontal 10bit gradation.

    Parameters
    ----------
    img : array_like
        image data. shape is must be (V_num, H_num, 3).

    Returns
    -------
    ndarray
        a image with 10bit horizontal gray scale.

    Notes
    -----
    -

    Examples
    --------
    >>> img = np.zeros((1080, 1920, 3), np.dtype=uint8)
    >>> composite_8bit_middle_gray_scale(img)
    """
    pass


def m_and_e_tp_rev5(width=1920, height=1080):

    # ベースの背景画像を作成
    # img = np.zeros((height, width, 3), dtype=np.uint16)
    img = np.ones((height, width, 3), dtype=np.uint16) * 0x4000

    # 外枠のフレーム作成
    tpg.draw_rectangle(img, (0, 0), (width-1, height-1), (1.0, 0.0, 0.0))

    # 左端にPQグレースケール
    composite_pq_vertical_gray_scale(img)

    # 右端にHLGグレースケール
    composite_hlg_vertical_gray_scale(img)

    # 真ん中上部に8btグレースケール(64～192)
    composite_8bit_middle_gray_scale(img)

    # 真ん中上部に10btグレースケール(256～768)
    composite_10bit_middle_gray_scale(img)

    # preview
    tpg.preview_image(img, 'rgb')

    # write to the file
    cv2.imwrite("test.tiff", img[:, :, ::-1])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    m_and_e_tp_rev5(1920, 1080)
    # m_and_e_tp_rev5(4096, 2160)

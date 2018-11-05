#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ITU-R. BT.2100 用の ColorChecker を作る。
"""

import os
import colour
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import cv2


""" Target Brightness [cd/m2] を選択 """
TARGET_BRIGHTNESS = 100  # ColorChecker のピーク輝度をどこに合わせるか


""" ColorChecker を選択 """
# COLOR_CHECKER_NAME = 'ColorChecker 1976'
COLOR_CHECKER_NAME = 'ColorChecker 2005'
# COLOR_CHECKER_NAME = 'BabelColor Average'


""" Chromatic Adaptation を選択 """
# CHROMATIC_ADAPTATION_TRANSFORM = 'Bradford'
CHROMATIC_ADAPTATION_TRANSFORM = 'CAT02'


""" Color Space を選択(Gamut, WhitePoint, XYZ_to_RGB_mtx で使用) """
COLOR_SPACE = colour.models.BT2020_COLOURSPACE
# COLOR_SPACE = colour.models.BT709_COLOURSPACE
# COLOR_SPACE = colour.models.ACES_PROXY_COLOURSPACE
# COLOR_SPACE = colour.models.S_GAMUT3_COLOURSPACE
# COLOR_SPACE = colour.models.S_GAMUT3_CINE_COLOURSPACE
# COLOR_SPACE = colour.models.V_GAMUT_COLOURSPACE


""" WhitePoint を選択 """
# WHITE_POINT_STR = 'D50'
# WHITE_POINT_STR = 'D55'
# WHITE_POINT_STR = 'D60'
# WHITE_POINT_STR = 'DCI-P3'
WHITE_POINT_STR = 'D65'

WHITE_POINT = colour.colorimetry.ILLUMINANTS['cie_2_1931'][WHITE_POINT_STR]

"""
OETF を選択

HDR の OETF は一番ミスりやすい箇所。
測定目的の場合は OOTF を考慮する必要がある。

HLG の場合、モニター側で EOTF と一緒に OOTF が掛かるため
OETF では OOTF の inverse も一緒に掛ける必要がある。

一方で ST2084 の場合はモニター側で OOTF は掛からないので
素直に OETF だけ適用すれば良い。


補足だが、以下の2つの関数は内部動作が異なる(OOTFの有無)。

* OETF = colour.models.oetf_ST2084
* OETF = colour.models.oetf_BT2100_PQ
"""
OETF_TYPE = 'HLG'
# OETF_TYPE = 'ST2084'
# OETF_TYPE = "sRGB"
# OETF_TYPE = "BT1886_Reverse"  # gamma = 1/2.4


""" Image Spec """
IMG_WIDTH = 3840
IMG_HEIGHT = 2160
COLOR_CHECKER_SIZE = 1 / 4.5  # [0:1] で記述
COLOR_CHECKER_PADDING = 0.02
COLOR_CHECKER_H_NUM = 6
COLOR_CHECKER_V_NUM = 4
IMG_MAX_LEVEL = 0xFFFF


""" ColorChecker Name """
COLOR_CHECKER_EACH_NAME = [
    "dark skin", "light skin", "blue sky", "foliage",
    "blue flower", "bluish green", "orange", "purplish blue",
    "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow",
    "magenta", "cyan", "white 9.5", "neutral 8",
    "neutral 6.5", "neutral 5", "neutral 3.5", "black 2"
]


def _add_text_info(img, st_pos, font_size=15, font_color=(0.5, 0.5, 0.5),
                   text="I'm a engineer"):
    """
    imgに対して指定座標からテキストを書く

    Parameters
    ----------
    img : ndarray
        image
    pos_st : array of numeric.
        position of the start. order is (H, V).
    font_size : integer
        font size
    font_color : tuple of numeric
        font color. (red, green, blue)
    text : strings
        text for write to the image.

    Returns
    -------
    -

    Examples
    --------
    >>> img = np.zeros((1080, 1920, 3), np.dtype=uint8)
    >>> st_pos = (300, 400)
    >>> _add_text_info(img, st_pos, text="I'm a HERO")
    """
    img_width = img.shape[1]
    img_height = img.shape[0]
    fg_color = tuple([int(x * 0xFF) for x in font_color])

    width = img_width // 3
    height = img_height // 20
    if st_pos[0] + width > img_width:
        print("text box is over(h).")
        width = img_width - st_pos[0]
    if st_pos[1] + height > img_height:
        print("text box is over(v).")
        height = img_height - st_pos[1]

    txt_img = Image.new("RGB", (width, height), (0x00, 0x00, 0x00))
    draw = ImageDraw.Draw(txt_img)
    font = ImageFont.truetype("./fonts/NotoSansMonoCJKjp-Regular.otf",
                              font_size)
    draw.text((0, 0), text, font=font, fill=fg_color)
    text_img = (np.asarray(txt_img) * 0x100).astype(np.uint16)
    text_img = text_img / 0xFFFF

    # temp_img = img[st_pos[1]:height, st_pos[0]:width]
    # temp_img = text_img
    st_pos_v = st_pos[1]
    ed_pos_v = st_pos[1] + height
    st_pos_h = st_pos[0]
    ed_pos_h = st_pos[0] + width

    # img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = text_img
    text_index = text_img > 0
    temp_img = img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h]
    temp_img[text_index] = text_img[text_index]

    img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = temp_img


def get_colorchecker_large_xyz_and_whitepoint(cc_name=COLOR_CHECKER_NAME):
    """
    ColorChecker の XYZ値を取得する

    Parameters
    ------------
    cc_name : strings
        color space name.

    Returns
    ------------
    array_like
        ColorChecker の XYZ値
    """
    colour_checker_param = colour.COLOURCHECKERS.get(cc_name)

    # 今回の処理では必要ないデータもあるので xyY と whitepoint だけ抽出
    # -------------------------------------------------------------
    _name, data, whitepoint = colour_checker_param
    temp_xyY = []
    for _index, label, xyY in data:
        temp_xyY.append(xyY)
    temp_xyY = np.array(temp_xyY)
    large_xyz = colour.models.xyY_to_XYZ(temp_xyY)

    return large_xyz, whitepoint


def get_linear_rgb_from_large_xyz(large_xyz, whitepoint,
                                  color_space=COLOR_SPACE):
    """
    XYZ値 から RGB値（Linear）を求める

    Parameters
    ------------
    large_xyz : array_like
        colorchecker の XYZ値
    whitepoint : array_like
        colorckecker の XYZ値の whitepoint
    color_space : RGB_Colourspace
        XYZ to RGB 変換の対象となる color space

    Returns
    ------------
    array_like
        [0:1] の Linear な RGBデータ
    """
    illuminant_XYZ = whitepoint   # ColorCheckerのオリジナルデータの白色点
    illuminant_RGB = WHITE_POINT  # RGBの白色点を設定
    chromatic_adaptation_transform = CHROMATIC_ADAPTATION_TRANSFORM
    large_xyz_to_rgb_matrix = color_space.XYZ_to_RGB_matrix
    rgb = colour.models.XYZ_to_RGB(large_xyz, illuminant_XYZ, illuminant_RGB,
                                   large_xyz_to_rgb_matrix,
                                   chromatic_adaptation_transform)

    # overflow, underflow check
    # -----------------------------
    rgb[rgb < 0.0] = 0.0
    rgb[rgb > 1.0] = 1.0

    return rgb


def oetf_bt1886(x):
    """
    BT.1886 の EOTF の Reverse を行う

    Parameters
    ------------
    x : array_like
        [0:1] の Linear Data

    Returns
    ------------
    array_like
        [0:1] の Gammaが掛かったデータ
    """
    return x ** (1/2.4)


def get_rgb_with_prime(rgb, bright=100):
    """
    Linear な RGB値に ガンマカーブを掛ける

    Parameters
    ------------
    rgb : array_like
        [0:1] の Linear Data

    Returns
    ------------
    array_like
        [0:1] の Gammaが掛かったデータ
    """

    if OETF_TYPE == 'HLG':
        oetf_func = colour.models.eotf_reverse_BT2100_HLG
    elif OETF_TYPE == 'ST2084':
        oetf_func = colour.models.oetf_ST2084
    elif OETF_TYPE == 'sRGB':
        oetf_func = colour.models.oetf_sRGB
    elif OETF_TYPE == 'BT1886_Reverse':
        oetf_func = oetf_bt1886
    else:
        oetf_func = None

    # [0:1] の RGB値を所望の輝度値[cd/m2] に変換。ただしHDRの場合のみ
    # 変換する理由は、oetf の関数の引数の単位が [cd/m2] だから。
    # ------------------------------------------------------------
    if OETF_TYPE == 'HLG' or OETF_TYPE == 'ST2084':
        rgb_bright = rgb * bright
    else:
        rgb_bright = rgb

    # OETF 適用
    # -----------------------------------------
    rgb_prime = oetf_func(rgb_bright)

    return rgb_prime


def preview_image(img, order='rgb', over_disp=False):
    """ OpenCV の機能を使って画像をプレビューする """
    if order == 'rgb':
        cv2.imshow('preview', img[:, :, ::-1])
    elif order == 'bgr':
        cv2.imshow('preview', img)
    else:
        raise ValueError("order parameter is invalid")

    if over_disp:
        cv2.resizeWindow('preview', )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_color_checker_image(rgb, bright):
    """
    以下の2種類の画像を生成＆保存＆Previewする。

    * 24枚の測定用 ColorChecker画像。画面中央にRGBパターン表示
    * 1枚の確認用画像。1枚の画面に24枚種類の ColorChecker を表示

    Parameters
    ------------
    rgb : array_like
        [0:1] の ガンマカーブ適用済みデータ

    """

    # 基本パラメータ算出
    # --------------------------------------
    h_num = 6
    v_num = 4
    img_height = IMG_HEIGHT
    img_width = IMG_WIDTH
    patch_st_h = int(IMG_WIDTH / 2.0
                     - (IMG_HEIGHT * COLOR_CHECKER_SIZE
                        * COLOR_CHECKER_H_NUM / 2.0
                        + (IMG_HEIGHT * COLOR_CHECKER_PADDING
                           * (COLOR_CHECKER_H_NUM / 2.0 - 0.5)) / 2.0))
    patch_st_v = int(IMG_HEIGHT / 2.0
                     - (IMG_HEIGHT * COLOR_CHECKER_SIZE
                        * COLOR_CHECKER_V_NUM / 2.0
                        + (IMG_HEIGHT * COLOR_CHECKER_PADDING
                           * (COLOR_CHECKER_V_NUM / 2.0 - 0.5)) / 2.0))
    patch_width = int(img_height * COLOR_CHECKER_SIZE)
    patch_height = patch_width
    patch_space = int(img_height * COLOR_CHECKER_PADDING)
    all_patch_file_str = "./img/ColorChecker_All_{:s}_{:s}_{:s}_{:d}nits.tiff"

    # 24ループで1枚の画像に24パッチを描画
    # -------------------------------------------------
    img_all_patch = np.zeros((img_height, img_width, 3))
    for idx in range(h_num * v_num):
        v_idx = idx // h_num
        h_idx = (idx % h_num)
        patch = np.ones((patch_height, patch_width, 3))
        patch[:, :] = rgb[idx]
        st_h = patch_st_h + (patch_width + patch_space) * h_idx
        st_v = patch_st_v + (patch_height + patch_space) * v_idx
        img_all_patch[st_v:st_v+patch_height, st_h:st_h+patch_width] = patch

    # テキスト付与
    # ------------
    text = "{:d}nits".format(bright)
    _add_text_info(img_all_patch, st_pos=(60, 60), font_size=70,
                   text=text, font_color=(0.75, 0.75, 0.75))

    # パッチのプレビューと保存
    # --------------------------------------------------
    preview_image(img_all_patch)
    file_name = all_patch_file_str.format(COLOR_SPACE._name,
                                          WHITE_POINT_STR, OETF_TYPE,
                                          bright)
    cv2.imwrite(file_name, _get_16bit_img(img_all_patch[:, :, ::-1]))


def _get_16bit_img(img):
    """
    16bit整数型に変換した画像データを得る

    Parameters
    ------------
    img : array_like
        [0:1] の浮動小数点の値

    Returns
    ------------
    array_like
        [0:65535] に正規化された 16bit整数型の値
    """
    return np.uint16(np.round(img * IMG_MAX_LEVEL))


def _get_10bit_img(img):
    """
    10bit整数型に変換した画像データを得る

    Parameters
    ------------
    img : array_like
        [0:1] の浮動小数点の値

    Returns
    ------------
    array_like
        [0:1023] に正規化された 10bit整数型の値
    """
    return np.uint16(np.round(img * 0x3FF))


def main_func():
    # 所望の設定に適したRGB値を算出
    # ---------------------------------
    large_xyz, whitepoint = get_colorchecker_large_xyz_and_whitepoint()
    rgb = get_linear_rgb_from_large_xyz(large_xyz, whitepoint)
    bright = 100
    for bright in range(100, 1100, 100):
        rgb_prime = get_rgb_with_prime(rgb, bright)

        # ColorChecker画像生成
        # ---------------------------------
        save_color_checker_image(rgb_prime, bright)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()

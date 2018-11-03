
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NUKEなどのコンポジットツールで LOGファイルを読み込み
HDR表示を試すためのテストパターン画像を作成する
"""

import os
import cv2
import OpenImageIO as oiio
import image_io as tyio
import transfer_functions as tf
import numpy as np
# import gamma_func as gm
import test_pattern_generator2 as tpg
import colour
from colour.characterisation import COLOURCHECKERS
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from scipy import linalg
import imp
imp.reload(tpg)

REVISION = 9


class LuminanceCodeValue:
    """
    Luminance と CodeValue の相互変換を行う
    """
    def __init__(self, name):
        """
        Parameters
        ----------
        img : name
            eotf/oetf name.

        Examples
        --------
        >>> lc = LuminanceCodeValue("gm24")
        >>> lc.l_to_cv(100)
        1.0
        >>> lc = LuminanceCodeValue("st2084")
        >>> lc.l_to_cv(100)
        0.01
        """
        # define gray Luminance
        gray_luminance = 20
        
        # get gray code value
        # gray = 0.18  # this value is writtten int the eotf specification.

        # [0:1] に正規化した際の gray の Linear空間での値を算出
        # gray_linear = eotf(eotf_name, gray)

        # max_luminance = 1 / gray_linear * gray_luminance

    def l_to_cv(self, value):
        pass

    def cv_to_l(self, value):
        pass


class TpgControl:
    """
    必要なパラメータの受け取り。部品作成。合成。プレビュー。ファイル吐き出し。
    """

    def __init__(self, resolution='3840x2160', transfer_function=tf.GAMMA24):
        self.bg_color = 0.75  # unit is nits
        self.fg_color = 100  # unit is nits
        self.transfer_function = transfer_function
        self.parse_resolution(resolution)
        self.bit_depth = 10
        self.img_max = (2 ** self.bit_depth) - 1

    def parse_resolution(self, resolution):
        if resolution == '1920x1080':
            self.img_width = 1920
            self.img_height = 1080
        elif resolution == '3840x2160':
            self.img_width = 3840
            self.img_height = 2160
        else:
            raise ValueError("Invalid resolution parameter.")

    def preview_iamge(self, order='rgb'):
        tpg.preview_image(self.img / 0x3FC, order)

    def draw_bg_color(self):
        """
        背景色を描く。
        nits で指定したビデオレベルを使用
        """
        code_value = tf.oetf_from_luminance(self.bg_color,
                                            self.transfer_function)
        code_value = round(code_value * self.img_max)
        self.img *= code_value

    def draw_outline(self):
        """
        外枠として1pxの直線を描く。
        nits で指定したビデオレベルを使用
        """
        code_value = tf.oetf_from_luminance(self.fg_color,
                                            self.transfer_function)
        code_value = round(code_value * self.img_max)

        st_h, st_v = (0, 0)
        ed_h, ed_v = (self.img_width - 1, self.img_height - 1)

        self.img[st_v, st_h:ed_h, :] = code_value
        self.img[ed_v, st_h:ed_h, :] = code_value
        self.img[st_v:ed_v, st_h, :] = code_value
        self.img[st_v:ed_v, ed_h, :] = code_value

    def make_image(self):
        self.img = np.ones((self.img_height, self.img_width, 3),
                           dtype=np.uint16)

        self.draw_bg_color()
        self.draw_outline()

    def save_image(self, fname):
        # メインのDPXファイル吐き出し
        attr = {"oiio:BitsPerSample": 10}
        writer = tyio.TyWriter(self.img / self.img_max, fname, attr)
        writer.write()

        # TIFFファイルも合わせて吐き出しておく
        root, ext = os.path.splitext(fname)
        fname_tiff = root + '.tiff'
        writer2 = tyio.TyWriter(self.img / self.img_max, fname_tiff)
        writer2.write(out_img_type_desc=oiio.UINT16)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    tpg_ctrl = TpgControl(resolution='1920x1080', transfer_function=tf.GAMMA24)
    tpg_ctrl.make_image()
    tpg_ctrl.preview_iamge()
    tpg_ctrl.save_image("./img/hoge.dpx")

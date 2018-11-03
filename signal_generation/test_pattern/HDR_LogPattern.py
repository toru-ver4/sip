
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

REVISION = 0
BIT_DEPTH = 10


class TpgIO:
    """
    TestPattern作成に特化した File IO の提供
    """

    def __init__(self, img=None, bit_depth=10):
        self.img = img
        self.bit_depth = BIT_DEPTH
        self.img_max = (2 ** self.bit_depth) - 1

    def save_dpx_image(self, fname):
        attr = {"oiio:BitsPerSample": self.bit_depth}
        writer = tyio.TyWriter(self.img / self.img_max, fname, attr)
        writer.write()

    def save_exr_image(self, fname):
        print("This Function is not implemented.")

    def save_tiff_image(self, fname):
        writer2 = tyio.TyWriter(self.img / self.img_max, fname)
        writer2.write(out_img_type_desc=oiio.UINT16)

    def save_image(self, fname):
        root, ext = os.path.splitext(fname)
        if ext == '.dpx':
            self.save_dpx_image(fname)
        elif ext == '.exr':
            self.save_exr_image(fname)

        # TIFFファイルも合わせて吐き出しておく
        self.save_tiff_image(root + '.tiff')

    def load_image(self, fname):
        """
        UIをどうするか、まだ決まっていない。
        """
        reader = tyio.TyReader(fname)
        self.load_img = reader.read()
        self.load_attr = reader.get_attr()
        self.load_img = self.load_img \
            / reader.get_max_value_from_dtype(self.load_img)
        img = np.uint16(np.round(self.load_img * self.img_max))
        print(img)

        return img


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
        io = TpgIO(self.img, BIT_DEPTH)
        io.save_image(fname)

    def load_image(self, fname):
        io = TpgIO(BIT_DEPTH)
        self.load_img = io.load_image(fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    tpg_ctrl = TpgControl(resolution='1920x1080', transfer_function=tf.GAMMA24)
    tpg_ctrl.make_image()
    # tpg_ctrl.preview_iamge()
    fname = "./img/hoge.dpx"
    tpg_ctrl.save_image(fname)
    tpg_ctrl.load_image(fname)

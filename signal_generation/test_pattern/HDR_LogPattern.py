
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NUKEなどのコンポジットツールで LOGファイルを読み込み
HDR表示を試すためのテストパターン画像を作成する
"""

import os
from TpgIO import TpgIO
from TpgDraw import TpgDraw
import transfer_functions as tf
import numpy as np
# import gamma_func as gm

REVISION = 0
BIT_DEPTH = 10


class TpgControl:
    """
    必要なパラメータの受け取り。各種命令の実行。
    """
    def __init__(self, resolution='3840x2160', transfer_function=tf.GAMMA24):
        self.bg_color = 0.75  # unit is nits
        self.fg_color = 100  # unit is nits
        self.transfer_function = transfer_function
        self.parse_resolution(resolution)
        self.bit_depth = 10
        self.draw_param = self.gen_keywords_for_draw()

    def parse_resolution(self, resolution):
        if resolution == '1920x1080':
            self.img_width = 1920
            self.img_height = 1080
        elif resolution == '3840x2160':
            self.img_width = 3840
            self.img_height = 2160
        else:
            raise ValueError("Invalid resolution parameter.")

    def gen_keywords_for_draw(self):
        """
        TpgDraw に渡すパラメータをまとめる
        """
        kwargs = {'bg_color': self.bg_color, 'fg_color': self.fg_color,
                  'img_width': self.img_width, 'img_height': self.img_height,
                  'bit_depth': self.bit_depth,
                  'transfer_function': self.transfer_function}

        return kwargs

    def draw_image(self):
        draw = TpgDraw(self.draw_param)
        self.img = draw.draw()

    def save_image(self, fname):
        io = TpgIO(self.img, BIT_DEPTH)
        io.save_image(fname)

    def load_image(self, fname):
        io = TpgIO(BIT_DEPTH)
        self.load_img = io.load_image(fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    tpg_ctrl = TpgControl(resolution='1920x1080', transfer_function=tf.GAMMA24)
    tpg_ctrl.draw_image()
    # tpg_ctrl.preview_iamge()
    fname = "./img/hoge.dpx"
    tpg_ctrl.save_image(fname)
    tpg_ctrl.load_image(fname)

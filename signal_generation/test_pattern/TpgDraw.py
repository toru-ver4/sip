#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# TpgDraw モジュール

## 概要
TestPattern作成時の各種描画関数をまとめたもの。
"""

import os
import numpy as np
import transfer_functions as tf
import test_pattern_generator2 as tpg
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import imp
imp.reload(tpg)


class TpgDraw:
    """
    テストパターン用の各種描画を行う。
    """
    def __init__(self, draw_param):
        # TpgControl から受け通るパラメータ
        self.bg_color = draw_param['bg_color']
        self.fg_color = draw_param['fg_color']
        self.img_width = draw_param['img_width']
        self.img_height = draw_param['img_height']
        self.bit_depth = draw_param['bit_depth']
        self.img_max = (2 ** self.bit_depth) - 1
        self.transfer_function = draw_param['transfer_function']

        # TpgDraw 内部パラメータ(細かい座標など)
        self.ramp_height_param = 0.075  # range is [0.0:1.0]
        self.ramp_st_pos_v_param = 0.5

    def preview_iamge(self, img, order='rgb'):
        tpg.preview_image(img, order)

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

    def draw_10bit_ramp(self):
        """
        10bitのRampパターンを描く
        """
        global g_cuurent_pos_v

        width = (self.img_height // 1080) * 1024
        height = int(self.img_height * self.ramp_height_param)
        ramp_st_pos_h = (self.img_width - width) // 2
        ramp_st_pos_v = int(self.img_height * self.ramp_st_pos_v_param)

        ramp_10bit = tpg.gen_step_gradation(width=width, height=height,
                                            step_num=1025,
                                            bit_depth=self.bit_depth,
                                            color=(1.0, 1.0, 1.0),
                                            direction='h')
        tpg.merge(self.img, ramp_10bit, pos=(ramp_st_pos_h, ramp_st_pos_v))

        self.preview_iamge(self.img / self.img_max)

    def draw(self):
        self.img = np.ones((self.img_height, self.img_width, 3),
                           dtype=np.uint16)
        self.draw_bg_color()
        self.draw_outline()
        self.draw_10bit_ramp()

        return self.img


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


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
import colour
# import gamma_func as gm

REVISION = 0
BIT_DEPTH = 10

BT709_CS = colour.models.BT709_COLOURSPACE
BT2020_CS = colour.models.BT2020_COLOURSPACE
V_GAMUT_CS = colour.models.V_GAMUT_COLOURSPACE
ALEXA_WIDE_GAMUT_CS = colour.models.ALEXA_WIDE_GAMUT_COLOURSPACE
S_GAMUT3_CINE_CS = colour.models.S_GAMUT3_CINE_COLOURSPACE
S_GAMUT3_CS = colour.models.S_GAMUT3_COLOURSPACE

PARAM_LIST = [{'tf': tf.GAMMA24, 'cs': BT709_CS, 'wp': 'D65'},
              {'tf': tf.GAMMA24, 'cs': BT2020_CS, 'wp': 'D65'},
              {'tf': tf.HLG, 'cs': BT2020_CS, 'wp': 'D65'},
              {'tf': tf.ST2084, 'cs': BT2020_CS, 'wp': 'D65'},
              {'tf': tf.SLOG3, 'cs': S_GAMUT3_CS, 'wp': 'D65'},
              {'tf': tf.SLOG3, 'cs': S_GAMUT3_CINE_CS, 'wp': 'D65'}]


class TpgControl:
    """
    必要なパラメータの受け取り。各種命令の実行。
    """
    def __init__(self, resolution='3840x2160', transfer_function=tf.GAMMA24,
                 color_space=BT709_CS, white_point="D65"):
        """
        white_point は 次のいずれか。'D50', 'D55', 'D60', 'D65', 'DCI-P3'
        """
        self.bg_color = 0.75  # unit is nits
        self.fg_color = 50  # unit is nits
        self.transfer_function = transfer_function
        self.parse_resolution(resolution)
        self.bit_depth = 10
        self.color_space = color_space
        self.white_point = white_point
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
                  'transfer_function': self.transfer_function,
                  'color_space': self.color_space,
                  'white_point': self.white_point}

        return kwargs

    def draw_image(self, preview=False):
        draw = TpgDraw(self.draw_param, preview)
        self.img = draw.draw()

    def save_image(self, fname):
        io = TpgIO(self.img, BIT_DEPTH)
        io.save_image(fname)

    def load_image(self, fname):
        io = TpgIO(BIT_DEPTH)
        self.load_img = io.load_image(fname)


def main_func():
    tf_list = [tf.GAMMA24, tf.HLG, tf.ST2084, tf.SLOG3]
    resolution_list = ['1920x1080', '3840x2160']

    for param in PARAM_LIST:
        transfer_function = param['tf']
        color_space = param['cs']
        white_point = param['wp']
        for resolution in resolution_list:
            tpg_ctrl = TpgControl(resolution=resolution,
                                  transfer_function=transfer_function,
                                  color_space=color_space,
                                  white_point=white_point)
            tpg_ctrl.draw_image(preview=False)
            fname = "./img/{}_{}_{}_{}.dpx".format(transfer_function,
                                                   color_space.name,
                                                   white_point,
                                                   resolution)
            tpg_ctrl.save_image(fname)
            # tpg_ctrl.load_image(fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()

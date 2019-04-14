
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

REVISION = 2
BIT_DEPTH = 10


class DciP3ColorSpace:
    """
    DCI-P3 D65 の定義
    """
    def __init__(self):
        self.name = "DCI-P3"


BT709_CS = colour.models.BT709_COLOURSPACE
BT2020_CS = colour.models.BT2020_COLOURSPACE
V_GAMUT_CS = colour.models.V_GAMUT_COLOURSPACE
ALEXA_WIDE_GAMUT_CS = colour.models.ALEXA_WIDE_GAMUT_COLOURSPACE
S_GAMUT3_CINE_CS = colour.models.S_GAMUT3_CINE_COLOURSPACE
S_GAMUT3_CS = colour.models.S_GAMUT3_COLOURSPACE
V_LOG_CS = colour.models.V_GAMUT_COLOURSPACE
ALEXA_WIDE_GAMUT_CS = colour.models.ALEXA_WIDE_GAMUT_COLOURSPACE
RED_WIDE_GAMUT_RGB_CS = colour.models.RED_WIDE_GAMUT_RGB_COLOURSPACE
DCI_P3_CS = DciP3ColorSpace()
SRGB_CS = colour.models.sRGB_COLOURSPACE

PARAM_LIST = [{'tf': tf.GAMMA24, 'cs': BT709_CS, 'wp': 'D65'},
              {'tf': tf.GAMMA24, 'cs': BT2020_CS, 'wp': 'D65'},
              {'tf': tf.HLG, 'cs': BT2020_CS, 'wp': 'D65'},
              {'tf': tf.ST2084, 'cs': BT2020_CS, 'wp': 'D65'},
              {'tf': tf.ST2084, 'cs': DCI_P3_CS, 'wp': 'D65'},
              {'tf': tf.SLOG3, 'cs': S_GAMUT3_CS, 'wp': 'D65'},
              {'tf': tf.VLOG, 'cs': V_LOG_CS, 'wp': 'D65'},
              {'tf': tf.LOGC, 'cs': ALEXA_WIDE_GAMUT_CS, 'wp': 'D65'},
              {'tf': tf.LOGC, 'cs': BT2020_CS, 'wp': 'D65'},
              {'tf': tf.LOG3G10, 'cs': RED_WIDE_GAMUT_RGB_CS, 'wp': 'D65'},
              {'tf': tf.LOG3G12, 'cs': RED_WIDE_GAMUT_RGB_CS, 'wp': 'D65'},
              {'tf': tf.LOG3G10, 'cs': BT2020_CS, 'wp': 'D65'},
              {'tf': tf.LOG3G12, 'cs': BT2020_CS, 'wp': 'D65'}]

# PARAM_LIST = [{'tf': tf.ST2084, 'cs': BT709_CS, 'wp': 'D65'},
#               {'tf': tf.SLOG3, 'cs': BT709_CS, 'wp': 'D65'},
#               {'tf': tf.VLOG, 'cs': BT709_CS, 'wp': 'D65'},
#               {'tf': tf.LOGC, 'cs': BT709_CS, 'wp': 'D65'},
#               {'tf': tf.LOG3G10, 'cs': BT709_CS, 'wp': 'D65'},
#               {'tf': tf.DLOG, 'cs': BT709_CS, 'wp': 'D65'},
#               {'tf': tf.FLOG, 'cs': BT709_CS, 'wp': 'D65'},
#               {'tf': tf.NLOG, 'cs': BT709_CS, 'wp': 'D65'},
#               {'tf': tf.GAMMA24, 'cs': BT709_CS, 'wp': 'D65'}]

# PARAM_LIST = [{'tf': tf.GAMMA24, 'cs': BT709_CS, 'wp': 'D65'}]


class TpgControl:
    """
    必要なパラメータの受け取り。各種命令の実行。
    """
    def __init__(self, resolution='3840x2160', transfer_function=tf.GAMMA24,
                 color_space=BT709_CS, white_point="D65",
                 revision=REVISION):
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
        self.revision = revision
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
                  'white_point': self.white_point,
                  'revision': self.revision}

        return kwargs

    def draw_image_type1(self, preview=False):
        draw = TpgDraw(self.draw_param, preview)
        self.img = draw.draw_tpg_type1()

    def draw_image_type2(self, preview=False):
        draw = TpgDraw(self.draw_param, preview)
        self.img = draw.draw_tpg_type2()

    def save_image(self, fname, transfer_function):
        io = TpgIO(self.img, BIT_DEPTH, transfer_function)
        io.save_image(fname)

    def load_image(self, fname):
        io = TpgIO(BIT_DEPTH)
        self.load_img = io.load_image(fname)


def main_func():
    resolution_list = ['1920x1080', '3840x2160']
    # resolution_list = ['1920x1080']

    for param in PARAM_LIST:
        transfer_function = param['tf']
        color_space = param['cs']
        white_point = param['wp']
        for resolution in resolution_list:
            tpg_ctrl = TpgControl(resolution=resolution,
                                  transfer_function=transfer_function,
                                  color_space=color_space,
                                  white_point=white_point,
                                  revision=REVISION)
            tpg_ctrl.draw_image_type1(preview=False)
            fname_str = "./img/{}_{}_{}_{}_rev{:02d}_type1.dpx"
            fname = fname_str.format(transfer_function,
                                     color_space.name,
                                     white_point,
                                     resolution,
                                     REVISION)
            tpg_ctrl.save_image(fname, transfer_function)
            fname_exr = "./img/{}_{}_{}_{}_rev{:02d}_type1.exr"
            fname = fname_exr.format(transfer_function,
                                     color_space.name,
                                     white_point,
                                     resolution,
                                     REVISION)
            tpg_ctrl.save_image(fname, transfer_function)

            # tpg_ctrl.draw_image_type2(preview=False)
            # fname_str = "./img/{}_{}_rev{:02d}_type2.exr"
            # fname = fname_str.format(transfer_function,
            #                          resolution,
            #                          REVISION)
            # tpg_ctrl.save_image(fname, transfer_function)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()

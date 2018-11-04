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

        self.convert_to_8bit_coef = 2 ** (self.bit_depth - 8)
        self.convert_from_10bit_coef = 2 ** (16 - self.bit_depth)

        # TpgDraw 内部パラメータ(細かい座標など)
        self.ramp_height_coef = 0.075  # range is [0.0:1.0]
        self.ramp_st_pos_h_coef = 0.4  # range is [0.0:1.0]
        self.ramp_st_pos_v_coef = 0.2  # range is [0.0:1.0]
        self.checker_8bit_st_pos_v_coef = 0.31  # range is [0.0:1.0]
        self.checker_10bit_st_pos_v_coef = 0.42  # range is [0.0:1.0]
        self.each_spec_text_size_coef = 0.02  # range is [0.0:1.0]
        self.outline_text_size_coef = 0.02  # range is [0.0:1.0]
        self.step_bar_width_coef = 0.95
        self.step_bar_height_coef = 0.2
        self.step_bar_st_pos_v_coef = 0.75
        self.step_bar_text_width = 0.3

        self.set_fg_code_value()
        self.set_bg_code_value()

    def set_bg_code_value(self):
        code_value = tf.oetf_from_luminance(self.bg_color,
                                            self.transfer_function)
        code_value = int(round(code_value * self.img_max))
        self.bg_code_value = code_value

    def get_bg_code_value(self):
        return self.bg_code_value

    def set_fg_code_value(self):
        code_value = tf.oetf_from_luminance(self.fg_color,
                                            self.transfer_function)
        code_value = int(round(code_value * self.img_max))
        self.fg_code_value = code_value

    def get_fg_code_value(self):
        return self.fg_code_value

    def preview_iamge(self, img, order='rgb'):
        tpg.preview_image(img, order)

    def draw_bg_color(self):
        """
        背景色を描く。
        nits で指定したビデオレベルを使用
        """
        code_value = self.get_bg_code_value()
        self.img *= code_value

    def draw_outline(self):
        """
        外枠として1pxの直線を描く。
        nits で指定したビデオレベルを使用
        """
        code_value = self.get_fg_code_value()

        st_h, st_v = (0, 0)
        ed_h, ed_v = (self.img_width - 1, self.img_height - 1)

        self.img[st_v, st_h:ed_h, :] = code_value
        self.img[ed_v, st_h:ed_h, :] = code_value
        self.img[st_v:ed_v, st_h, :] = code_value
        self.img[st_v:ed_v, ed_h, :] = code_value

    def get_each_spec_text_height_and_size(self):
        """
        各パーツ説明のテキストの高さ(px)とフォントサイズを吐く
        """
        font_size = self.img_height * self.each_spec_text_size_coef
        text_height = font_size / 72 * 96 * 1.1

        return int(text_height), int(font_size)

    def get_color_bar_text_font_size(self, text_height):
        """
        カラーバー横に表示する階調＋輝度のフォントサイズを取得する
        """
        font_size = int(text_height / 96 * 72)
        return font_size

    def get_text_st_pos_for_over_info(self, tp_pos, text_height):
        return (tp_pos[0], tp_pos[1] - text_height)

    def get_fg_color_for_pillow(self):
        """
        Pillow 用 に 8bit精度のFG COLORを算出する
        """
        text_video_level_8bit\
            = int(self.fg_code_value / self.convert_to_8bit_coef)
        fg_color = tuple([text_video_level_8bit for x in range(3)])
        return fg_color

    def convert_from_pillow_to_numpy(self, img):
        img = np.uint16(np.asarray(img)) * self.convert_to_8bit_coef

        return img

    def merge_text(self, txt_img, pos):
        """
        テキストを合成する作業の最後の部分。
        pos は テキストの (st_pos_h, st_pos_v) 。

        ## 個人的実装メモ
        今回はちゃんとアルファチャンネルを使った合成をしたかったが、
        PILは8bit, それ以外は 10～16bit により BG_COLOR に差が出るので断念。
        """
        st_pos_v = pos[1]
        ed_pos_v = pos[1] + txt_img.shape[0]
        st_pos_h = pos[0]
        ed_pos_h = pos[0] + txt_img.shape[1]

        # かなり汚い実装。0x00 で無いピクセルのインデックスを抽出し、
        # そのピクセルのみを元の画像に上書きするという処理をしている。
        text_index = txt_img > 0
        temp_img = self.img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h]
        temp_img[text_index] = txt_img[text_index]
        self.img[st_pos_v:ed_pos_v, st_pos_h:ed_pos_h] = temp_img

    def merge_each_spec_text(self, pos, font_size, text_img_size, text):
        """
        各パーツの説明テキストを合成。
        pos は テキストの (st_pos_h, st_pos_v) 。
        text_img_size = (size_h, size_v)

        ## 個人的実装メモ
        今回はちゃんとアルファチャンネルを使った合成をしたかったが、
        PILは8bit, それ以外は 10～16bit により BG_COLOR に差が出るので断念。
        """
        # テキストイメージ作成
        text_width = text_img_size[0]
        text_height = text_img_size[1]
        fg_color = self.get_fg_color_for_pillow()
        bg_coor = (0x00, 0x00, 0x00)
        txt_img = Image.new("RGB", (text_width, text_height), bg_coor)
        draw = ImageDraw.Draw(txt_img)
        font = ImageFont.truetype("./fonts/NotoSansMonoCJKjp-Regular.otf",
                                  font_size)
        draw.text((0, 0), text, font=font, fill=fg_color)
        txt_img = self.convert_from_pillow_to_numpy(txt_img)

        self.merge_text(txt_img, pos)

    def draw_10bit_ramp(self):
        """
        10bitのRampパターンを描く
        """
        global g_cuurent_pos_v

        # パラメータ計算
        width = (self.img_height // 1080) * 1024
        height = int(self.img_height * self.ramp_height_coef)
        ramp_st_pos_h = self.get_ramp_st_pos_h(width)
        ramp_st_pos_v = int(self.img_height * self.ramp_st_pos_v_coef)
        ramp_pos = (ramp_st_pos_h, ramp_st_pos_v)
        text_height, font_size = self.get_each_spec_text_height_and_size()
        text = "▼ 10bit gray ramp from 0 to 1023 level."
        text_pos = self.get_text_st_pos_for_over_info(ramp_pos, text_height)

        # ramp パターン作成
        ramp_10bit = tpg.gen_step_gradation(width=width, height=height,
                                            step_num=1025,
                                            bit_depth=self.bit_depth,
                                            color=(1.0, 1.0, 1.0),
                                            direction='h')
        tpg.merge(self.img, ramp_10bit, pos=ramp_pos)
        self.merge_each_spec_text(text_pos, font_size,
                                  (width, text_height), text)

    def get_bit_depth_checker_grad_width(self):
        """
        画面中央のグラデーション(256～768)の幅を求める
        """
        if self.img_height == 1080:
            width = 2048
        elif self.img_height == 2160:
            width = 4096
        else:
            raise ValueError('invalid img_height')

        return width

    def get_bit_depth_checker_grad_st_ed(self):
        """
        8bit/10bit チェック用のグラデーションは長めに作ってあるので
        最後にトリミングが必要となる。
        トリミングポイントの st, ed を返す
        """
        grad_width = self.get_bit_depth_checker_grad_width()
        grad_st_h = grad_width // 4
        grad_ed_h = grad_st_h + (grad_width // 2)

        return grad_st_h, grad_ed_h

    def get_ramp_st_pos_h(self, width):
        return int(self.img_width * self.ramp_st_pos_h_coef)

    def draw_8bit_10bit_checker(self, bit_depth='8bit', pos_v_coef=0.5):
        """
        256～768 の 8bit/10bitのRampパターンを描く
        """
        # パラメータ計算
        width = self.get_bit_depth_checker_grad_width()
        height = int(self.img_height * self.ramp_height_coef)
        grad_st_pos_h, grad_ed_pos_h = self.get_bit_depth_checker_grad_st_ed()
        width_after_trim = grad_ed_pos_h - grad_st_pos_h
        ramp_st_pos_h = self.get_ramp_st_pos_h(width_after_trim)
        ramp_st_pos_v = int(self.img_height * pos_v_coef)
        ramp_pos = (ramp_st_pos_h, ramp_st_pos_v)
        text_height, font_size = self.get_each_spec_text_height_and_size()
        text = "▼ " + bit_depth + " gray ramp from 256 to 768 level."
        text_pos = self.get_text_st_pos_for_over_info(ramp_pos, text_height)

        # ramp パターン作成
        step_num = 257 if bit_depth == '8bit' else 1025
        ramp = tpg.gen_step_gradation(width=width, height=height,
                                      step_num=step_num,
                                      bit_depth=self.bit_depth,
                                      color=(1.0, 1.0, 1.0),
                                      direction='h')
        tpg.merge(self.img, ramp[:, grad_st_pos_h:grad_ed_pos_h],
                  pos=ramp_pos)
        self.merge_each_spec_text(text_pos, font_size,
                                  (width_after_trim, text_height), text)

    def get_color_bar_st_pos_h(self, width):
        return (self.img_width - width) // 2

    def draw_wrgbmyc_color_bar(self):
        """
        階段状のカラーバーをプロットする
        """
        scale_step = 65
        color_list = [(1, 1, 1), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                      (1, 0, 1), (1, 1, 0), (0, 1, 1)]
        width = int(self.img_width * self.step_bar_width_coef)
        height = int(self.img_height * self.step_bar_height_coef)
        color_bar_st_pos_h = self.get_color_bar_st_pos_h(width)
        color_bar_st_pos_v = int(self.img_height * self.step_bar_st_pos_v_coef)
        st_pos = (color_bar_st_pos_h, color_bar_st_pos_v)

        bar_height_list = tpg.equal_devision(height, len(color_list))
        bar_img_list = []
        for color, bar_height in zip(color_list, bar_height_list):
            color_bar = tpg.gen_step_gradation(width=width, height=bar_height,
                                               step_num=scale_step,
                                               bit_depth=self.bit_depth,
                                               color=color, direction='h')
            bar_img_list.append(color_bar)
        color_bar = np.vstack(bar_img_list)
        tpg.merge(self.img, color_bar, st_pos)

        # ここからテキスト。あらかじめV方向で作っておき、最後に回転させる
        txt_img = self.get_video_level_text_img(scale_step, width)
        text_pos = self.get_text_st_pos_for_over_info(st_pos, txt_img.shape[0])
        self.merge_text(txt_img, text_pos)

    def get_video_level_text_img(self, scale_step, width):
        """
        ステップカラーに付与する VideoLevel & Luminance 情報。
        最初は縦向きで作って、最後に横向きにする
        """
        fg_color = self.get_fg_color_for_pillow()
        text_height_list = tpg.equal_devision(width, scale_step)
        font_size = self.get_color_bar_text_font_size(width / scale_step)
        video_level = np.linspace(0, 2 ** self.bit_depth, scale_step)
        video_level[-1] -= 1
        video_level_float = video_level / self.img_max
        bright_list = tf.eotf_to_luminance(video_level_float,
                                           self.transfer_function)
        text_width = int(self.step_bar_text_width * self.img_height)
        txt_img = Image.new("RGB", (text_width, width), (0x00, 0x00, 0x00))
        draw = ImageDraw.Draw(txt_img)
        font = ImageFont.truetype("./fonts/NotoSansMonoCJKjp-Regular.otf",
                                  font_size)
        st_pos_h = 0
        st_pos_v = 0
        for idx in range(scale_step):
            pos = (st_pos_h, st_pos_v)
            if bright_list[idx] < 999.99999:
                text_data = " {:>4.0f},{:>7.1f} nit".format(video_level[idx],
                                                            bright_list[idx])
            else:
                text_data = " {:>4.0f},{:>6.0f} nit".format(video_level[idx],
                                                            bright_list[idx])
            draw.text(pos, text_data, font=font, fill=fg_color)
            st_pos_v += text_height_list[idx]

        txt_img = self.convert_from_pillow_to_numpy(txt_img)
        txt_img = np.rot90(txt_img)

        return txt_img

    def draw(self):
        self.img = np.ones((self.img_height, self.img_width, 3),
                           dtype=np.uint16)
        self.draw_bg_color()
        self.draw_outline()
        self.draw_10bit_ramp()
        self.draw_8bit_10bit_checker('8bit', self.checker_8bit_st_pos_v_coef)
        self.draw_8bit_10bit_checker('10bit', self.checker_10bit_st_pos_v_coef)
        self.draw_wrgbmyc_color_bar()

        self.preview_iamge(self.img / self.img_max)

        return self.img


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

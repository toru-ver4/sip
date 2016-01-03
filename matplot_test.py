#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
Matplotの動作テスト
"""

import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from numpy.random import randn

#Picture_file_name='tabako.jpg' 
Picture_file_name='hiasshuku.tiff' 

if __name__ == '__main__':

    # 描画用のWindow？を準備
    fig = plt.figure()

    # fig の中に複数のグラフを定義
    # rgb_hist = fig.add_subplot(4,1,1) # 引数はそれぞれ 横数、縦数、index 
    # r_hist = fig.add_subplot(4,1,2)
    # g_hist = fig.add_subplot(4,1,3)
    # b_hist = fig.add_subplot(4,1,4)

    # add_axes を使って細かく位置を調整
    left_margin = 0.1
    bottom_margin = 0.05
    right_margin = 0.01
    top_margin = 0.05
    plot_width = 1.0 - (left_margin + right_margin)
    plot_height = ( 1.0 - (top_margin + bottom_margin) ) / 5.0
    rgb_hist = fig.add_axes((left_margin, bottom_margin + plot_height * 3, plot_width, plot_height * 2))
    r_hist = fig.add_axes(  (left_margin, bottom_margin + plot_height * 2, plot_width, plot_height * 1), sharey=rgb_hist)
    g_hist = fig.add_axes(  (left_margin, bottom_margin + plot_height * 1, plot_width, plot_height * 1), sharey=rgb_hist)
    b_hist = fig.add_axes(  (left_margin, bottom_margin + plot_height * 0, plot_width, plot_height * 1), sharey=rgb_hist)

    # x軸の目盛レンジ設定
    rgb_hist.set_xlim(0, 256)
    r_hist.set_xlim(0, 256)
    g_hist.set_xlim(0, 256)
    b_hist.set_xlim(0, 256)

    # 軸の非表示設定
    rgb_hist.tick_params(labelbottom="off")
    r_hist.tick_params(labelbottom="off", labelleft="off")
    g_hist.tick_params(labelbottom="off", labelleft="off")
    b_hist.tick_params(labelleft="off")

    # ヒストグラムデータを準備
    img = cv2.imread(Picture_file_name)
    img_b, img_g, img_r = np.dsplit(img, 3)

    # ヒストグラム作成
    hist_bins = 64
    hist_alpha = 0.5
    rgb_hist.hist(img.flatten(), bins=hist_bins, color='k', alpha=hist_alpha)
    r_hist.hist(img_r.flatten(), bins=hist_bins, color='r', alpha=hist_alpha)
    g_hist.hist(img_g.flatten(), bins=hist_bins, color='g', alpha=hist_alpha)
    b_hist.hist(img_b.flatten(), bins=hist_bins, color='b', alpha=hist_alpha)

    plt.show()



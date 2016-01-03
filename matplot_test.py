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
    rgb_hist = fig.add_subplot(4,1,1) # 引数はそれぞれ 横数、縦数、index 
    r_hist = fig.add_subplot(4,1,2)
    g_hist = fig.add_subplot(4,1,3)
    b_hist = fig.add_subplot(4,1,4)

    # ヒストグラムデータを準備
    img = cv2.imread(Picture_file_name)
    img_b, img_g, img_r = np.dsplit(img, 3)

    # ヒストグラム作成
    hist_bins = 256
    hist_alpha = 0.5
    rgb_hist.hist(img.flatten(), bins=hist_bins, color='k', alpha=hist_alpha)
    r_hist.hist(img_r.flatten(), bins=hist_bins, color='r', alpha=hist_alpha)
    g_hist.hist(img_g.flatten(), bins=hist_bins, color='g', alpha=hist_alpha)
    b_hist.hist(img_b.flatten(), bins=hist_bins, color='b', alpha=hist_alpha)

    plt.show()


    




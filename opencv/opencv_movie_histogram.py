#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
OpenCV の動画編集テスト。
"""

import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

File_name='bbb.dpx' 
normalized_val_uint16 = 65535

def calc_hist(in_data, out):
    for idx in in_data.flatten():
        out[idx] += 1
    
    return out


if __name__ == '__main__':

    capture = cv2.VideoCapture("hanayamata.mp4")
    wait_time = int(1/29.97 * 1000)

    ret, img_pre = capture.read()
    if(ret != True):
        print("source open error!")
        sys.exit(0)


    counter = 0
    fig = plt.figure()
    left_margin = 0.1
    bottom_margin = 0.05
    right_margin = 0.01
    top_margin = 0.05
    plot_width = 1.0 - (left_margin + right_margin)
    plot_height = ( 1.0 - (top_margin + bottom_margin) ) / 5.0
    ax1 = fig.add_axes((left_margin, bottom_margin + plot_height * 3, plot_width, plot_height * 2))
    ax2 = fig.add_axes(  (left_margin, bottom_margin + plot_height * 2, plot_width, plot_height * 1), sharey=ax1)
    ax3 = fig.add_axes(  (left_margin, bottom_margin + plot_height * 1, plot_width, plot_height * 1), sharey=ax1)
    ax4 = fig.add_axes(  (left_margin, bottom_margin + plot_height * 0, plot_width, plot_height * 1), sharey=ax1)

    # x軸の目盛レンジ設定
    ax1.set_xlim(0, 256)
    ax2.set_xlim(0, 256)
    ax3.set_xlim(0, 256)
    ax4.set_xlim(0, 256)

    # 軸の非表示設定
    ax1.tick_params(labelbottom="off")
    ax2.tick_params(labelbottom="off", labelleft="off")
    ax3.tick_params(labelbottom="off", labelleft="off")
    ax4.tick_params(labelleft="off")

    hist_rgb = np.zeros(256, dtype=np.int)
    hist_r = np.zeros(256, dtype=np.int)
    hist_g = np.zeros(256, dtype=np.int)
    hist_b = np.zeros(256, dtype=np.int)

    count = 0
    while True:
        ret, img_now = capture.read()
        if(ret != True):
            break
        if (count % 96) == 0:
            img_now = cv2.resize(img_now, (960, 540))
            print(count // 24)
            b, g, r = [x.flatten() for x in np.dsplit(img_now, 3)]
            hist_rgb = hist_rgb + np.histogram(img_now.flatten(), np.arange(257, dtype=np.int))[0]
            hist_r = hist_r + np.histogram(r, np.arange(257, dtype=np.int))[0]
            hist_g = hist_g + np.histogram(g, np.arange(257, dtype=np.int))[0]
            hist_b = hist_b + np.histogram(b, np.arange(257, dtype=np.int))[0]

        count += 1
        

    # ヒストグラム作成
    plt.yscale('log')
    ax1.plot(hist_rgb, '-', color='k')
    ax2.plot(hist_r, '-', color='r')
    ax3.plot(hist_g, '-', color='g')
    ax4.plot(hist_b, '-', color='b')
    plt.show()

    

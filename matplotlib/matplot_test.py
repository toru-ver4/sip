#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
Matplotの動作テスト
"""

import os
import sys
import time
import pandas
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from numpy.random import randn

#Picture_file_name='tabako.jpg' 
Picture_file_name = 'hiasshuku.tiff' 
Csv_file_name = 'xyz_list.csv'

Rec_709_area  = [[0.640, 0.300, 0.150, 0.640], [0.330, 0.600, 0.060, 0.330]]
Rec_2020_area = [[0.708, 0.170, 0.131, 0.708], [0.292, 0.797, 0.046, 0.292]]

Resize_resolution = (320, 180)

def Get_xyz_Color_Matching_func(csv_file):
    """csv形式で書かれた xyz等色関数を readする"""
    csv = np.array(pandas.read_csv(csv_file, header=None)).transpose()
    # ret_mtx = [0] * 4
    # ret_mtx = [ [] for x in ret_mtx ]
    # ret_mtx[0] = csv['wlen'])
    # ret_mtx[1] = np.csv['val_x']
    # ret_mtx[2] = csv['val_y']
    # ret_mtx[3] = csv['val_z']

    return csv


def Calc_Spectrum_xy_Chromaticity(xyz_mtx):
    """
    xyzの等色関数からxy色度を覓める
      xyz_mtx[0] : 波長
      xyz_mtx[1] : x
      xyz_mtx[2] : y
      xyz_mtx[2] : z
    """
    sum_xyz = (xyz_mtx[1] + xyz_mtx[2] + xyz_mtx[3])
    x = xyz_mtx[1] / sum_xyz
    y = xyz_mtx[2] / sum_xyz

    # 先頭に右端のデータ付加して、プロットがグルっと一周するようにする。
    max_idx = np.argmax(x)
    x = np.append(np.array(x[max_idx]), x)
    y = np.append(np.array(y[max_idx]), y)

    return xyz_mtx[0], x, y

def RGB_to_XYZ(img, mat=None):
    """RGBをXYZに変換する。mat が None の場合は cvtColor で XYZ変換する。
       その場合、色域は Rec.709、色温度は D65 に固定となる。"""
    if mat != None:
        b, g, r = np.dsplit(img, 3)

        # 行列計算
        ret_X = r * mat[0][0] + g * mat[0][1] + b * mat[0][2]
        ret_Y = r * mat[1][0] + g * mat[1][1] + b * mat[1][2]
        ret_Z = r * mat[2][0] + g * mat[2][1] + b * mat[2][2]

        # XYZ結合
        ret_img = np.dstack( (ret_X, ret_Y, ret_Z) )
        
    else:
        ret_img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

    return ret_img

def RGB_to_xy(img, mat=None):
    """RGB から xy色度を算出。戻り値は x, y の配列"""
    # 正規化
    normalize_val = (2 ** (8 * img.itemsize)) - 1
    img = np.float32(img / normalize_val)
    
    img_XYZ = RGB_to_XYZ(img, mat)
    X, Y, Z = np.dsplit(img_XYZ, 3)
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    
    return x, y    

if __name__ == '__main__':

    print("kidou")
    xyz_mtx = Get_xyz_Color_Matching_func(Csv_file_name)
    wave_len, chroma_x, chroma_y = Calc_Spectrum_xy_Chromaticity(xyz_mtx)

    # 動画ファイルを開く
    capture = cv2.VideoCapture("nichijo_op.mp4")
    ret, img_RGB = capture.read()
#    img_RGB = cv2.imread(Picture_file_name)
    
    # 処理負荷軽減のためにResize
    img_RGB_resize = cv2.resize(img_RGB, Resize_resolution)

    # xy に変換
    img_x, img_y = RGB_to_xy(img_RGB_resize)

    print("kidou2")
    # 描画用のWindow？を準備
    fig = plt.figure(figsize=(10,10)) # fgsize は inch で指定
    fig.patch.set_facecolor('white')

    # 描画領域の設定
    ax1 = fig.add_axes( (0.1, 0.1, 0.85, 0.85) )
    ax1.tick_params(axis='both', which='major', labelsize=20)

    # 軸の範囲を設定
    ax1.set_xlim(0, 0.8)
    ax1.set_ylim(0, 0.9)

    # 軸ラベルの設定
    plt.xlabel("$x$", fontsize=25)
    plt.ylabel("$y$", fontsize=25)

    # 描画領域の色を設定
    ax1.patch.set_facecolor('black')
    ax1.patch.set_alpha(0.15)

    # 描画
    lines, = ax1.plot(img_x.flatten(), img_y.flatten(), '.')

    ax1.plot(chroma_x, chroma_y, '-', color='k', label="CIE1931")
    ax1.plot(Rec_709_area[0], Rec_709_area[1], '--', color='r', label="Rec709")
    ax1.plot(Rec_2020_area[0], Rec_2020_area[1], '--', color='g', label="Rec2020" )

    # 判例の描画
    plt.legend() 

    # 補助線の描画
    plt.grid()

    plt.pause(.001)

    print("kidou3")
    while True:
        # 1フレーム取得
        ret, img_RGB = capture.read()
        
        # 処理負荷軽減のためにResize
        img_RGB_resize = cv2.resize(img_RGB, Resize_resolution)
        img_x, img_y = RGB_to_xy(img_RGB_resize)

        lines.set_data(img_x.flatten(), img_y.flatten())

        plt.pause(.01)

        cv2.imshow("cam view", img_RGB)
        cv2.waitKey(1)
        print("kidou4")
    
    sys.exit(1)


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



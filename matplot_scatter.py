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
Picture_file_name = '20141004232349bf4.jpg'
Csv_file_name = 'xyz_list.csv'

Rec_709_area  = [[0.640, 0.300, 0.150, 0.640], [0.330, 0.600, 0.060, 0.330]]
Rec_2020_area = [[0.708, 0.170, 0.131, 0.708], [0.292, 0.797, 0.046, 0.292]]
DCI_P3_area   = [[0.680, 0.265, 0.150, 0.680], [0.320, 0.690, 0.060, 0.320]]

Resize_resolution = (360, 180)

class ScatterPlot():
    def __init__(self):

        # CIE1931のxy色度を算出
        xyz_mtx = Get_xyz_Color_Matching_func(Csv_file_name)
        wave_len, chroma_x, chroma_y = Calc_Spectrum_xy_Chromaticity(xyz_mtx)

        
        # 描画用のWindow？を準備
        self.fig = plt.figure(figsize=(10,10)) # fgsize は inch で指定
        self.fig.patch.set_facecolor('white')

        # 描画領域の設定
        self.ax1 = self.fig.add_axes( (0.1, 0.1, 0.85, 0.85) )
        self.ax1.tick_params(axis='both', which='major', labelsize=20)
        
        # 軸の範囲を設定
        self.ax1.set_xlim(0, 0.8)
        self.ax1.set_ylim(0, 0.9)
        
        # 軸ラベルの設定
        plt.xlabel("$x$", fontsize=25)
        plt.ylabel("$y$", fontsize=25)
        
        # 描画領域の色を設定
        self.ax1.patch.set_facecolor('black')
        self.ax1.patch.set_alpha(0.15)

        # 各領域をプロット
        self.ax1.plot(chroma_x, chroma_y, '-', color='k', label="CIE1931")
        self.ax1.plot(Rec_709_area[0],  Rec_709_area[1],  '--', color='r', label="Rec709")
        self.ax1.plot(Rec_2020_area[0], Rec_2020_area[1], '--', color='g', label="Rec2020" )
        self.ax1.plot(DCI_P3_area[0],   DCI_P3_area[1],   '--', color='c', label="DCI-P3" )

        # 判例の描画
        plt.legend() 
        
        # 補助線の描画
        plt.grid()

    
    def set_data(self, x_data, y_data, color_data=None):
        self.ax1.scatter(x_data, y_data, marker='o', c=color_data, s=50, alpha=0.2, edgecolors='face')

    def show(self):
        t0 = time.time()
        
        plt.show()

        t1 = time.time()
        print(t1-t0)
    def save(self, name="hoge.png"):
        plt.savefig(name, bbox_inches='tight')


def RGB_to_Scatter_RGB(img_RGB):

    # 1.0で正規化
    normalize_val = (2 ** (8 * img_RGB.itemsize)) - 1
    img_RGB_normalized = np.float32(img_RGB / normalize_val)

    # 明度を固定したいので、HSV空間でVの値を強制的に変える
    img_HSV = cv2.cvtColor(img_RGB_normalized, cv2.COLOR_BGR2HSV)
    img_HSV[:,:,2] = 1.0
    img_bright = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR)

    # RGB値を(R,G,B)のタプルで表現
    t0 = time.time()
    color_array = img_bright.reshape(img_bright.shape[0] * img_bright.shape[1], 3)
    t1 = time.time()
    print(t1-t0)
    # RGB値を #XXXXXX 形式で出力
    # color_array = []
    # width = img_bright.shape[1]
    # for v_idx, h_val in enumerate(img_bright):
    #     for h_idx, v_val in enumerate(h_val):
    #         color_str = "#%02x%02x%02x" % (v_val[0], v_val[1], v_val[2])
    #         color_array.append(color_str)

    return color_array, img_bright

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

    # 動画ファイルを開く
    img_RGB = cv2.imread(Picture_file_name)
    
    # 処理負荷軽減のためにResize
#    img_RGB_resize = cv2.resize(img_RGB, Resize_resolution)
    img_RGB_resize = cv2.resize(img_RGB, (img_RGB.shape[1]//2, img_RGB.shape[0]//2))

    # xy に変換
    img_x, img_y = RGB_to_xy(img_RGB_resize)

    # 散布図の色指定用の配列を作成
    scatter_color, hsv_img = RGB_to_Scatter_RGB(img_RGB_resize)

    img_vcat  = cv2.vconcat([img_RGB_resize, np.uint8(hsv_img * 255)])

    cv2.imshow("get HS", img_vcat)
    cv2.waitKey(1)


    # ScatterPlotインスタンス作成
    my_plt_obj = ScatterPlot()

    # データを設定
    my_plt_obj.set_data(img_x, img_y, color_data=scatter_color)

    # 描画
    my_plt_obj.show()

    
    
    sys.exit(1)



#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
Matplotの動作テスト
"""

import os
import sys
import time
import datetime
import pandas
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from numpy.random import randn

Picture_file_name = 'hiasshuku.tiff' 
Movie_file_name   = 'nichijo_op.mp4'
Csv_file_name     = 'xyz_list.csv'
Picture_mode_str  = '-p'
Movie_mode_str    = '-m'
Plot_mode = Picture_mode_str

Rec_709_area  = [[0.640, 0.300, 0.150, 0.640], [0.330, 0.600, 0.060, 0.330]]
Rec_2020_area = [[0.708, 0.170, 0.131, 0.708], [0.292, 0.797, 0.046, 0.292]]
DCI_P3_area   = [[0.680, 0.265, 0.150, 0.680], [0.320, 0.690, 0.060, 0.320]]

Resize_resolution = (360, 180)

class ScatterPlot():
    """
    xy色度図の描画に特化したクラス。
    汎用性は欠片も考慮してないので、このクラスを継承することは推奨しない。
    """
    
    def __init__(self):
        """凡例やラベルの設定はinitで済ませておく。"""

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

        # プロット用変数の初期化
        self.alpha = 0.5
        self.marker = 'o'
        self.marker_size = 50
        self.edgecolors = 'face'

        # 判例の描画
        plt.legend() 
        
        # 補助線の描画
        plt.grid()
    
    def set_data(self, x_data, y_data, color_data=None):
        """散布図のデータの描画。戻り値の point_obj はアニメーション時に使う"""
        self.point_obj = self.ax1.scatter(x_data, 
                                          y_data, 
                                          marker=self.marker, 
                                          c=color_data, 
                                          s=self.marker_size, 
                                          alpha=self.alpha, 
                                          edgecolors=self.edgecolors)

    def update_data(self, x_data, y_data, color_data=None):
        """散布図のデータの再描画。set_offsets() に合わせてデータの並べ替えをしている"""
        set_data = np.dstack((x_data, y_data))
        self.point_obj.set_offsets(set_data)
        self.point_obj.set_color(color_data)
        
    def show(self):
        plt.show()

    def show_seq(self, delay=0.01):
        """アニメーション用の表示"""
        plt.pause(delay)

    def save(self, name=None):
        """グラフの保存。上書きしないようにファイル名に日付情報を入れている"""
        if name == None:
            name = 'scatter_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.png'
        plt.savefig(name, bbox_inches='tight')

def usage():
    print('Usage: %s [options [filename]]' % sys.argv[0])
    print('  -h, --help   show this help message and exit')
    print('  -p,          plot the pictire file')
    print('  -m,          plot the movie file')
    print('  filename    a filename')
    print('Example:')
    print('$ %s -m fate_op.mp4' % sys.argv[0])
    print('$ %s -p gochiusa_img.tiff' % sys.argv[0])

def Get_Args():
    # グローバル変数を書き換えるので global 宣言しとく
    global Plot_mode
    global Picture_file_name
    global Movie_file_name

    if len(sys.argv) > 1:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            usage()
        Plot_mode = sys.argv[1]

    if len(sys.argv) > 2:
        if sys.argv[1] == Picture_mode_str:
            Picture_file_name = sys.argv[2]
        elif sys.argv[1] == Movie_mode_str:
            Movie_file_name = sys.argv[2]
        else:
            pass
    print(Picture_file_name)
    print(Movie_file_name)
    print(Plot_mode)
        
def RGB_to_Scatter_RGB(img_RGB):
    """
    散布図の各点の着色用データを算出する
    元画像のRGB値をそのまま使うと明度の低い部分が暗くなってしまうので、
    HSV空間で V=1.0 としてから逆変換した RGB値を着色用のデータとして使う。
    """

    # 1.0で正規化
    normalize_val = (2 ** (8 * img_RGB.itemsize)) - 1
    img_RGB_normalized = np.float32(img_RGB / normalize_val)

    # 明度を固定したいので、HSV空間でVの値を強制的に変える
    img_HSV = cv2.cvtColor(img_RGB_normalized, cv2.COLOR_BGR2HSV)
    img_HSV[:,:,2] = 1.0
    img_bright = cv2.cvtColor(img_HSV, cv2.COLOR_HSV2RGB)

    # RGB値を(R,G,B)のタプルで表現
    t0 = time.time()
    color_array = img_bright.reshape(img_bright.shape[0] * img_bright.shape[1], 3)
    t1 = time.time()

    return color_array

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
    
    Get_Args()
    
    if Plot_mode == Picture_mode_str:
        
        # 静止画ファイルを開く
        img_RGB = cv2.imread(Picture_file_name)
        
        # 静止画を表示
        cv2.imshow(Picture_file_name, img_RGB)
        cv2.waitKey(1)
    
        # 処理負荷軽減のためにResize
        img_RGB_resize = cv2.resize(img_RGB, Resize_resolution)
        
        # xy に変換
        img_x, img_y = RGB_to_xy(img_RGB_resize)

        # 散布図の色指定用の配列を作成
        scatter_color = RGB_to_Scatter_RGB(img_RGB_resize)

        # ScatterPlotインスタンス作成
        my_plt_obj = ScatterPlot()

        # データを設定
        my_plt_obj.set_data(img_x, img_y, color_data=scatter_color)

        # 描画 & 保存
        my_plt_obj.save()
        my_plt_obj.show()

    elif Plot_mode == Movie_mode_str:
        # 動画ファイルを開く
        capture = cv2.VideoCapture(Movie_file_name)
        ret, img_RGB = capture.read()
    
        # ------------+
        # 1回目の処理 |
        # ------------+

        # 静止画を表示
        cv2.imshow(Movie_file_name, img_RGB)
        cv2.waitKey(1)

        # 処理負荷軽減のためにResize
        # =====================================================
        # Resize のアルゴリズムを変えられる？間引き的なヤツにしたい
        # =====================================================
        img_RGB_resize = cv2.resize(img_RGB, Resize_resolution)

        # xy に変換
        img_x, img_y = RGB_to_xy(img_RGB_resize)

        # 散布図の色指定用の配列を作成
        scatter_color = RGB_to_Scatter_RGB(img_RGB_resize)

        # ScatterPlotインスタンス作成
        my_plt_obj = ScatterPlot()

        # データを設定
        my_plt_obj.set_data(img_x, img_y, color_data=scatter_color)
        my_plt_obj.show_seq(0.01)

        # ----------------+
        # 2回目移行の処理 |
        # ----------------+
        while True:
            ret, img_RGB = capture.read()
            if ret != True:
                break

            # 静止画を表示
            cv2.imshow(Movie_file_name, img_RGB)
            cv2.waitKey(1)

            img_RGB_resize = cv2.resize(img_RGB, Resize_resolution)
            
            # xy に変換
            img_x, img_y = RGB_to_xy(img_RGB_resize)
            
            # 散布図の色指定用の配列を作成
            scatter_color = RGB_to_Scatter_RGB(img_RGB_resize)

            
            # データを更新
            my_plt_obj.update_data(img_x, img_y, color_data=scatter_color)
            my_plt_obj.show_seq(0.01)

    else:
        print('invalid mode select')

    # exit
    cv2.destroyAllWindows()




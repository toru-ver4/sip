#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
helloqt.py
PyQt5 で Hello, world!
"""

import os
import sys
import cv2
import numpy as np
from pgmagick import Image

File_name='bbb.dpx' 
normalized_val_uint16 = 65535

def gamma_func(val):
    return val * 0.5

def conv_dpx10_to_tiff16(name):
    root, ext = os.path.splitext(name)
    out_name = root + ".tif"
    cmd = "convert -depth 10" + " " + name + " " + "-depth 16" + " " + out_name
    print(cmd)
    os.system(cmd)

    return out_name

def view_limited_black(img):
    threshold = 64/1023 
    img = img * (img < threshold)
    img = img * (1 / threshold)

    return img

def view_superwhite(img):
    threshold = 512/1023 
    img = img * (img > threshold)

    return img

if __name__ == '__main__':

    # 拡張子チェック
    root, ext = os.path.splitext(File_name)
    print(root, ext)
    if ext == ".dpx":
        # dpx形式のファイルをtiff(16bit)に変換
        tiff_image_name = conv_dpx10_to_tiff16(File_name)
    elif ext == ".tif" or ext == ".tiff":
        tiff_image_name = File_name
    else:
        print("please set tiff or dpx image.")
        sys.exit(1)

    # tiffのデータをread
    img = cv2.imread(tiff_image_name, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)

    # 0..1 の範囲に正規化
    normalized_val = 2**(4 * img.dtype.num) - 1
    img             = img/normalized_val

    # 画像を半分にリサイズ
    image_width  = img.shape[1]//2
    image_height = img.shape[0]//2
    img_resize_half = cv2.resize(img, (image_width, image_height))

    # 半分、64未満、940以上の画像を抽出
    img_half_level  = gamma_func(img_resize_half)
    img_black_area  = view_limited_black(img_resize_half)
    img_super_white = view_superwhite(img_resize_half)
    
    # 各画像を結合して1つの画像にする
    img_vcat1 = cv2.vconcat([img_resize_half, img_half_level])
    img_vcat2 = cv2.vconcat([img_black_area, img_super_white])
    img_hcat  = cv2.hconcat([img_vcat1, img_vcat2])

    # 画像のプレビュー
    cv2.imshow('bbb.tif', img_hcat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 出力用に 0..1 → 0..65535 の変換を実施
    out_img = img_hcat * normalized_val_uint16
    out_img = np.uint16(out_img)

    # 保存
    cv2.imwrite('out.tiff', out_img)
    

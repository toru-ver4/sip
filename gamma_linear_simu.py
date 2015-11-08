#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
パワポ資料用にリニアで保存したイメージの
画像ファイルを作るよ！
"""

import os
import sys
import cv2
import numpy as np
from pgmagick import Image

File_name='001.JPG' 
normalized_val_uint16 = 65535
normalized_val_uint8 = 255

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

    # tiffのデータをread
    img = cv2.imread(File_name, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)

    # 0..1 の範囲に正規化
    normalized_val = 2**(4 * img.dtype.num) - 1
    img             = img/normalized_val

    img = cv2.pow(img, 3.5)
    img = img * normalized_val
    img = np.uint8(img)
    img = img/normalized_val
    img = cv2.pow(img, 1/3.5)

    # 画像のプレビュー
    cv2.imshow('bbb.tif', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 出力用に 0..1 → 0..65535 の変換を実施
    out_img = img * normalized_val_uint8
    out_img = np.uint8(out_img)

    # 保存
    cv2.imwrite('out.jpg', out_img)
    

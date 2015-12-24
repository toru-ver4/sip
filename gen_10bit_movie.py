#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
乱数動画を生成する。
静止画は 16bit で作って、エンコードの際に 16bit --> 10bit に落とす。
ffmpeg.exe は予め同一フォルダに用意しておくこと。
"""

import os
import sys
import cv2
import numpy as np

normalized_val_uint16 = 65535
image_width  = 1920
image_height = 1080
color_num    = 3
fps          = 60
movie_time   = 3


if __name__ == '__main__':

    # 乱数画像を生成
    for idx in range(fps * movie_time):
        image = np.random.rand(image_height, image_width, color_num)
        image = image * normalized_val_uint16
        image = np.uint16(image)
        idx_str = str(idx).zfill(4)
        save_str = "random_" + idx_str + ".tiff"
        cv2.imwrite(save_str, image)
        
        
    # ffmpeg で encode。ffmpeg.exe を同じフォルダに入れといてね！
    cmd = "ffmpeg -r 60 -i random_%4d.tiff -c:v prores_ks -profile:v 4 -pix_fmt yuv444p10le out.mov"
    print(cmd)
    os.system(cmd)


    

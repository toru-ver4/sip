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

File_name='bbb.dpx' 
normalized_val_uint16 = 65535


if __name__ == '__main__':

#    capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture("nichijo_op.mp4")
    wait_time = int(1/29.97 * 1000)

    while True:
        ret, img = capture.read()
        cv2.imshow("cam view", img)
        if cv2.waitKey(wait_time) >= 0:
            break

    cv2.destroyAllWindows()
    sys.exit(1)
    # 画像のプレビュー
    cv2.imshow('bbb.tif', img_hcat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 出力用に 0..1 → 0..65535 の変換を実施
    out_img = img_hcat * normalized_val_uint16
    out_img = np.uint16(out_img)

    # 保存
    cv2.imwrite('out.tiff', out_img)
    

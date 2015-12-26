#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
OpenCV の動画編集テスト。
"""

import os
import sys
import cv2
import numpy as np

File_name='bbb.dpx' 
normalized_val_uint16 = 65535
target_frame_no = 100


if __name__ == '__main__':

    capture = cv2.VideoCapture(0) # from web camera
#    capture = cv2.VideoCapture("nichijo_op.mp4") # from video file
    
    for idx in range(target_frame_no - 1):
        ret, img = capture.read()

    ret, img = capture.read()
    cv2.imshow("cam view", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存
    cv2.imwrite('out.tiff', img)
    sys.exit(1)

    

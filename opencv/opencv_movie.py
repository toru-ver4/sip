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

    ret, img_pre = capture.read()
    if(ret != True):
        print("source open error!")
        sys.exit(0)

    while True:
        ret, img_now = capture.read()
        if(ret != True):
            break
        img_edit = (img_pre * 0.4) + (img_now * 0.6)
        img_edit = np.uint8(img_edit)
        img_view = cv2.hconcat([img_now, img_edit])
        cv2.imshow("cam view", img_view)

        img_pre = img_edit.copy()
        if cv2.waitKey(1) >= 0:
            break

    cv2.destroyAllWindows()
    

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
        ret, img_org = capture.read()
        if(ret != True):
            break
        edge_info = cv2.Canny(img_org, 40.0, 200.0)
        img_edit = cv2.cvtColor(edge_info, cv2.COLOR_GRAY2RGB)
#        img_edit = cv2.cvtColor(img_org, cv2.COLOR_RGB2YUV)

        img_view = cv2.hconcat([img_org, img_edit])
        cv2.imshow("cam view", img_view)
#        if cv2.waitKey(wait_time) >= 0:
        if cv2.waitKey(20) >= 0:
            break

    cv2.destroyAllWindows()
    

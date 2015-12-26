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

normalized_val_uint16 = 65535

# Bradford's XYZ2LMS
XYZ_to_LMS_mat = [ [ 0.8951000,  0.2664000, -0.1614000 ],
                   [ -0.7502000, 1.7135000,  0.0367000 ],
                   [ 0.0389000,  -0.0685000, 1.0296000 ] ]
LMS_to_XYZ_mat = [ [ 0.9869929,  -0.1470543, 0.1599627 ],
                   [ 0.4323053,   0.5183603, 0.0492912 ],
                   [ -0.0085287,  0.0400428, 0.9684867 ] ]
XYZ_to_LMS_mat = np.float32(XYZ_to_LMS_mat)
LMS_to_XYZ_mat = np.float32(LMS_to_XYZ_mat)

# ref : http://www.filmlight.ltd.uk/pdf/whitepapers/FL-TL-TN-0417-StdColourSpaces.pdf
#   name              x        y
color_temp_4000k = [0.3820, 0.3792]
color_temp_4500k = [0.3620, 0.3656]
color_temp_5000k = [0.3460, 0.3532]
color_temp_5500k = [0.3330, 0.3421]
color_temp_6000K = [0.3224, 0.3324]
color_temp_6500K = [0.3137, 0.3239]
color_temp_D40   = [0.3823, 0.3838]
color_temp_D45   = [0.3621, 0.3709]
color_temp_D50   = [0.3457, 0.3587]
color_temp_D55   = [0.3325, 0.3476]
color_temp_D60   = [0.3217, 0.3378]
color_temp_D65   = [0.3128, 0.3292]
color_temp_D70   = [0.3054, 0.3216]

# set target
src_white_xy = color_temp_6500K
dst_white_xy = color_temp_5000k

src_white_xyz = src_white_xy + [( 1 - ( src_white_xy[0] + src_white_xy[1] ) )]
dst_white_xyz = dst_white_xy + [( 1 - ( dst_white_xy[0] + dst_white_xy[1] ) )]

def xyz_to_XYZ(xyz):
    ret = [0.0] * 3
    ret[0] = xyz[0] / xyz[1]
    ret[1] = 1.0
    ret[2] = xyz[2] / xyz[1]
    
    return np.float32(ret)

def XYZ_to_LMS(XYZ):
    ret = np.dot(XYZ_to_LMS_mat, XYZ)

    return ret

def calc_LMS_MAT(s_LMS, d_LMS):
    divided_LMS = d_LMS / s_LMS
    ret = np.diag(divided_LMS)
    
    return ret

def calc_XYZ_to_XYZ(MA, MA_inv, LMS):
    fst_result = LMS.dot(MA)
    ret = MA_inv.dot(fst_result)

    return ret

if __name__ == '__main__':
    src_XYZ = xyz_to_XYZ(src_white_xyz)
    dst_XYZ = xyz_to_XYZ(dst_white_xyz)
    src_LMS = XYZ_to_LMS(src_XYZ)
    dst_LMS = XYZ_to_LMS(dst_XYZ)
    LMS_MAT = calc_LMS_MAT(src_LMS, dst_LMS)
    XYZ_to_XYZ_mat = calc_XYZ_to_XYZ(XYZ_to_LMS_mat, LMS_to_XYZ_mat, LMS_MAT)
    print(src_XYZ)
    print(dst_XYZ)
    print(src_LMS)
    print(dst_LMS)
    print(LMS_MAT)
    print(XYZ_to_XYZ_mat)

#    capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture("nichijo_op.mp4")

    while True:
        ret, img_src = capture.read()
        if(ret != True):
            break
        img_dst = cv2.cvtColor(img_src, cv2.COLOR_RGB2XYZ)
        img_dst = XYZ_to_XYZ_mat * img_dst
        img_dst = cv2.cvtColor(img_dst, cv2.COLOR_XYZ2RGB)
        img_dst = np.uint8(img_dst)
        img_view = cv2.hconcat([img_src, img_dst])
        cv2.imshow("cam view", img_view)

        if cv2.waitKey(1) >= 0:
            break

    cv2.destroyAllWindows()
    

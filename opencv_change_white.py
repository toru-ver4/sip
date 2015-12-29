#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
OpenCV の動画編集テスト。
色温度変換を試す。
"""

import os
import sys
import time
import cv2
import numpy as np

normalized_val_uint16 = 65535

# Bradford's XYZ2LMS <http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html>
XYZ_to_LMS_mat = [ [ 0.8951000,  0.2664000, -0.1614000 ],
                   [ -0.7502000, 1.7135000,  0.0367000 ],
                   [ 0.0389000,  -0.0685000, 1.0296000 ] ]
LMS_to_XYZ_mat = [ [ 0.9869929,  -0.1470543, 0.1599627 ],
                   [ 0.4323053,   0.5183603, 0.0492912 ],
                   [ -0.0085287,  0.0400428, 0.9684867 ] ]
RGB_to_XYZ_mat = [ [ 0.412391,  0.357584,  0.180481 ],
                   [ 0.212639,  0.715169,  0.072192 ],
                   [ 0.019331,  0.119195,  0.950532 ] ]
XYZ_to_RGB_mat = [ [ 3.240970, -1.537383, -0.498611 ],
                   [-0.969244,  1.875968,  0.041555 ],
                   [ 0.055630, -0.203977,  1.056972 ] ]
XYZ_to_LMS_mat = np.float32(XYZ_to_LMS_mat)
LMS_to_XYZ_mat = np.float32(LMS_to_XYZ_mat)
RGB_to_XYZ_mat = np.float32(RGB_to_XYZ_mat)
XYZ_to_RGB_mat = np.float32(XYZ_to_RGB_mat)

# ref : http://www.filmlight.ltd.uk/pdf/whitepapers/FL-TL-TN-0417-StdColourSpaces.pdf
#   name              x        y
color_temp_4000k = [0.3820, 0.3792]
color_temp_4500k = [0.3620, 0.3656]
color_temp_5000k = [0.3460, 0.3532]
color_temp_5500k = [0.3330, 0.3421]
color_temp_6000k = [0.3224, 0.3324]
color_temp_6500k = [0.3137, 0.3239]
color_temp_D40   = [0.3823, 0.3838]
color_temp_D45   = [0.3621, 0.3709]
color_temp_D50   = [0.3457, 0.3587]
color_temp_D55   = [0.3325, 0.3476]
color_temp_D60   = [0.3217, 0.3378]
color_temp_D65   = [0.3128, 0.3292]
color_temp_D70   = [0.3054, 0.3216]

# set target
src_white_xy = color_temp_6500k
dst_white_xy = color_temp_5000k

src_white_xyz = src_white_xy + [( 1 - ( src_white_xy[0] + src_white_xy[1] ) )]
dst_white_xyz = dst_white_xy + [( 1 - ( dst_white_xy[0] + dst_white_xy[1] ) )]

def xyz_to_XYZ(xyz):
    """xyz から XYZ を計算。Y=1.0 で正規化してる。"""
    ret = [0.0] * 3
    ret[0] = xyz[0] / xyz[1]
    ret[1] = 1.0
    ret[2] = xyz[2] / xyz[1]
    
    return np.float32(ret)

def XYZ_to_LMS(XYZ):
    """XYZから人間？の刺激値に変換"""
    ret = np.dot(XYZ_to_LMS_mat, XYZ)

    return ret

def calc_LMS_MAT(s_LMS, d_LMS):
    """LMS_MATを計算する"""
    divided_LMS = d_LMS / s_LMS
    ret = np.diag(divided_LMS)
    
    return ret

def calc_XYZ_to_XYZ(MA, MA_inv, LMS):
    """色温度変換の XYZ_to_XYZをLMSから計算"""
    fst_result = LMS.dot(MA)
    ret = MA_inv.dot(fst_result)

    return ret

def normalize_RGB_to_RGB_mat(mat):
    """そもそもオーバーフローしないように正規化を行う"""
    sum_array = [ np.sum(row) for row in mat ]
    ret_mat = mat / np.max(sum_array)

    return ret_mat

def multiply_3x3_mat(src, mat):
    """RGBの各ピクセルに対して3x3の行列演算を行う"""

    # 正規化用の係数を調査
    normalize_val = (2 ** (8 * src.itemsize)) - 1

    # 0 .. 1 に正規化して RGB分離
    b, g, r = np.dsplit(src / normalize_val, 3)

    # 行列計算
    ret_r = r * mat[0][0] + g * mat[0][1] + b * mat[0][2]
    ret_g = r * mat[1][0] + g * mat[1][1] + b * mat[1][2]
    ret_b = r * mat[2][0] + g * mat[2][1] + b * mat[2][2]

    # オーバーフロー確認(実は Matrixの係数を調整しているので不要)
    ret_r = cv2.min(ret_r, 1.0)
    ret_g = cv2.min(ret_g, 1.0)
    ret_b = cv2.min(ret_b, 1.0)

    # アンダーフロー確認(実は Matrixの係数を調整しているので不要)
    ret_r = cv2.max(ret_r, 0.0)
    ret_g = cv2.max(ret_g, 0.0)
    ret_b = cv2.max(ret_b, 0.0)

    # RGB結合
    ret_mat = np.dstack( (ret_b, ret_g, ret_r) )

    # 0 .. 255 に正規化
    ret_mat *= normalize_val

    return np.uint8(ret_mat)

if __name__ == '__main__':
    
    # http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html を参考に…
    src_XYZ = xyz_to_XYZ(src_white_xyz)
    dst_XYZ = xyz_to_XYZ(dst_white_xyz)
    src_LMS = XYZ_to_LMS(src_XYZ)
    dst_LMS = XYZ_to_LMS(dst_XYZ)
    LMS_MAT = calc_LMS_MAT(src_LMS, dst_LMS)
    XYZ_to_XYZ_mat = calc_XYZ_to_XYZ(XYZ_to_LMS_mat, LMS_to_XYZ_mat, LMS_MAT)
    RGB_to_RGB_mat = np.dot( XYZ_to_RGB_mat, np.dot(XYZ_to_XYZ_mat, RGB_to_XYZ_mat) )
    RGB_to_RGB_mat = normalize_RGB_to_RGB_mat(RGB_to_RGB_mat)
    print(XYZ_to_XYZ_mat)
    print(RGB_to_RGB_mat)

#    capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture("nichijo_op.mp4")

    while True:
        # 1フレーム抜き出す
        ret, img_src = capture.read()
        if(ret != True):
            break

        # 1.0 に正規化して RGB に分離(BGRの順序なことに注意)
        b_array, g_array, r_array = np.dsplit(img_src, 3)

        # 色温度変換
        img_dst = multiply_3x3_mat(img_src, RGB_to_RGB_mat)

        # src と dst を１つにまとめる
        img_view = cv2.hconcat([img_src, img_dst])

        # 表示
        cv2.imshow("cam view", img_view)

        if cv2.waitKey(1) >= 0:
            break

    cv2.destroyAllWindows()
    

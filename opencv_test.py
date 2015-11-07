#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
helloqt.py
PyQt5 で Hello, world!
"""

File_name='bbb.dpx' 
import os
import sys
import cv2
import numpy as np
from pgmagick import Image

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

if __name__ == '__main__':

    tiff_image_name = conv_dpx10_to_tiff16(File_name)

    img = cv2.imread(tiff_image_name, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)

    normalized_val = 2**(4 * img.dtype.num) - 1
    print("normalized_val is", normalized_val)

    normalized_img = img/normalized_val

    img = gamma_func(normalized_img)
    # cv2.imshow('bbb.tif', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    img = img * normalized_val_uint16
    img = np.uint16(img)

    cv2.imwrite('tabako_half.tiff', img)
    

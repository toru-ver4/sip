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

    tiff_image_name = conv_dpx10_to_tiff16(File_name)

    img = cv2.imread(tiff_image_name, cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)

    normalized_val = 2**(4 * img.dtype.num) - 1
    print("normalized_val is", normalized_val)

    img             = img/normalized_val
    print(img.shape)
    image_width  = img.shape[1]//2
    image_height = img.shape[0]//2
    print(image_width, image_height)
    img_resize_half = cv2.resize(img, (image_width, image_height))
    img_half_level  = gamma_func(img_resize_half)
    img_black_area  = view_limited_black(img_resize_half)
    img_super_white = view_superwhite(img_resize_half)
    
    # concatenate image
    img_vcat1 = cv2.vconcat([img_resize_half, img_half_level])
    img_vcat2 = cv2.vconcat([img_black_area, img_super_white])
    img_hcat  = cv2.hconcat([img_vcat1, img_vcat2])

    cv2.namedWindow("fullscreen", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("fullscreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('bbb.tif', img_hcat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out_img = img_hcat * normalized_val_uint16
    out_img = np.uint16(out_img)

    cv2.imwrite('tabako_half.tiff', out_img)
    

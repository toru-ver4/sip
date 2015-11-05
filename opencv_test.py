#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
helloqt.py
PyQt5 で Hello, world!
"""

FileName='aaa.tif' 
import os
import sys
import cv2
import numpy as np

def gamma_func(val):
    return val * 0.5


if __name__ == '__main__':

    img = cv2.imread('tabako.jpg', cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
    print(img[0])

    img = img/255

    for hhh, line in enumerate(img):
        for vvv, pixel in enumerate(line):
            for rgb, val in enumerate(pixel):
                img[hhh][vvv][rgb] = gamma_func(val)

    print(img[0])
    img = img * 65535
    img = np.uint16(img)

    cv2.imwrite('tabako_half.tiff', img)
    

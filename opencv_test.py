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

if __name__ == '__main__':

    img = cv2.imread('tabako.jpg', cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
    print(img[0][0])
#    cv2.imshow('aaa', img)

#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

#    cv2.imwrite('bbb.tif', img)

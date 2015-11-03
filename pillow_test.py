#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
helloqt.py
PyQt5 で Hello, world!
"""
 
import sys
from PIL import Image

if __name__ == '__main__':
    img = Image.open('tabako.jpg', 'r')
    pixels = img.load()
    width, height = img.size

    for xIdx in range(width):
        for yIdx in range(height):
            rrr = round(pixels[xIdx,yIdx][0] * 0.5)
            ggg = round(pixels[xIdx,yIdx][1] * 0.5)
            bbb = round(pixels[xIdx,yIdx][2] * 0.5)
            pixels[xIdx,yIdx] = (rrr, ggg, bbb)
            
    img.save('tabako_half.jpg')
    

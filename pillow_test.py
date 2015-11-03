#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
helloqt.py
PyQt5 で Hello, world!
"""
 
import sys
from PIL import Image

if __name__ == '__main__':
    img = Image.open('mikuru.jpg', 'r')
    out = img.point(lambda i:i * 0.5)
    out.save('half.jpg', 'JPEG')
    

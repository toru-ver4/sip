#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
oiio の量子化誤差確認
"""

import os
import numpy as np
import OpenImageIO as oiio


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(dir(oiio))

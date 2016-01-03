#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
Matplotの動作テスト
"""

import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from numpy.random import randn

Picture_file_name='tabako.jpg' 

if __name__ == '__main__':

    # 描画用のWindow？を準備
    fig = plt.figure()

    # fig の中に複数のグラフを定義
    ax1 = fig.add_subplot(2,2,1) # 引数はそれぞれ 横数、縦数、index 
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)

    # ax に対して描画
    ax1.hist(randn(300), bins=20, color='k', alpha=0.3)
    ax2.plot(randn(50).cumsum(), 'k--' )

    plt.show()
    




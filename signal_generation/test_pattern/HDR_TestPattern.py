#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HDR用のテストパターンを作る
"""

import os
import cv2
import numpy as np
import test_pattern_generator as tpg
import colour
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imp
imp.reload(tpg)


def m_and_e_tp_rev5(width=3840, height=2160):

    # ベースの背景画像を作成
    # img = np.zeros((height, width, 3), dtype=np.uint16)
    img = np.ones((height, width, 3), dtype=np.uint16) * 0x4000

    # 外枠のフレーム作成
    tpg.draw_rectangle(img, (0, 0), (width-1, height-1), (1.0, 0.0, 0.0))
    # tpg.draw_rectangle(img, (50, 100), (640, 480), (1.0, 0.0, 0.0))

    tpg.preview_image(img, 'rgb')
    cv2.imwrite("test.tiff", img[:, :, ::-1])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    m_and_e_tp_rev5(1920, 1080)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ICC Profile絡みの動作検証
"""

import os
import numpy as np
import test_pattern_generator2 as tpg
import TyImageIO as tyio
import OpenImageIO as oiio


def resolve_bt709_checker():
    """
    Davinci Resolve で Linear の
    0.0, 0.18, 0.20, 0.90, 1.00 が
    BT.709 では、どの Code Value にマッピングされるか確認する。
    """
    img = np.zeros((1080, 1920, 3), dtype=np.float64)
    base_patch = np.ones((300, 300, 3), dtype=np.float64)

    p018 = base_patch * 0.18
    p020 = base_patch * 0.20
    p090 = base_patch * 0.90
    p100 = base_patch * 1.00

    tpg.merge(img, p018, (300, 300))
    tpg.merge(img, p020, (700, 300))
    tpg.merge(img, p090, (300, 700))
    tpg.merge(img, p100, (700, 700))

    # tpg.preview_image(img)
    writer = tyio.TyWriter(img, "hoge.exr")
    writer.write(out_img_type_desc=oiio.FLOAT)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    resolve_bt709_checker()

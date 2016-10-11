#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
iPhone7のGamut検証用データ作成

"""

import os
import sys
import cv2
import numpy as np
from PIL import ImageCms
from PIL import Image


def gen_color_bar():
    width = 1920
    height = 1080
    r_img = np.zeros((height, width, 3))
    o_img = np.zeros((height, width, 3))
    g_img = np.zeros((height, width, 3))
    b_img = np.zeros((height, width, 3))
    w_img = np.ones((height, width, 3))

    # 各色作る。OpenCV は RGB でなく BGR なのに注意
    # -------------------------------------------
    r_img[:, :, 2] = 1
    o_img[:, :, 2] = 1
    o_img[:, :, 0] = 0.25
    o_img[:, :, 1] = 0.5
    g_img[:, :, 1] = 1
    b_img[:, :, 0] = 1

    # cv2.imshow('preview', r_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # uint8 に変換
    # -------------------------------------------
    r_img = np.uint8(np.round(r_img * 0xFF))
    o_img = np.uint8(np.round(o_img * 0xFF))
    g_img = np.uint8(np.round(g_img * 0xFF))
    b_img = np.uint8(np.round(b_img * 0xFF))
    w_img = np.uint8(np.round(w_img * 0xFF))

    # 保存
    # -------------------------------------------
    cv2.imwrite("./picture/r.png", r_img)
    cv2.imwrite("./picture/o.png", o_img)
    cv2.imwrite("./picture/g.png", g_img)
    cv2.imwrite("./picture/b.png", b_img)
    cv2.imwrite("./picture/w.png", w_img)


def gen_saturation_color_bar():
    width = 1920
    height = 1080 // 3

    gradation = np.arange(width - 1, 0 - 1, -1) / (width - 1)
    static = np.ones(width)

    r = np.dstack((static, gradation, gradation))
    g = np.dstack((gradation, static, gradation))
    b = np.dstack((gradation, gradation, static))

    r_array = [r] * height
    g_array = [g] * height
    b_array = [b] * height

    r_img = np.vstack(r_array)
    g_img = np.vstack(g_array)
    b_img = np.vstack(b_array)

    img = np.vstack([r_img, g_img, b_img])

    # cv2.imshow('preview', img[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img = np.uint8(np.round(img * 0xFF))
    img = img & 0xF0
    cv2.imwrite("./picture/saturation.png", img[:, :, ::-1])


def open_profile():
    filename = "./picture/DCI-P3.icc"
    profile = ImageCms.getOpenProfile(filename)
    print(profile.tobytes())
    # print(profile.profile.red_colorant)


def add_profile():
    in_file_list = ["./picture/r.png", "./picture/o.png",
                    "./picture/g.png", "./picture/b.png",
                    "./picture/w.png", "./picture/saturation.png"]
    out_file_list = ["./picture/r_dci.png", "./picture/o_dci.png",
                     "./picture/g_dci.png", "./picture/b_dci.png",
                     "./picture/w_dci.png", "./picture/saturation_dci.png"]
    out_file_list2 = ["./picture/r_sRGB.png", "./picture/o_sRGB.png",
                      "./picture/g_sRGB_png", "./picture/b_sRGB.png",
                      "./picture/w_sRGB.png", "./picture/saturation_sRGB.png"]

    dci_profile_name = "./picture/DCI-P3_modify.icc"
    dci_profile = ImageCms.getOpenProfile(dci_profile_name)
    sRGB_profile = ImageCms.getOpenProfile(dci_profile_name)
    sRGB_profile_ref = ImageCms.createProfile('sRGB')
    sRGB_profile.profile = sRGB_profile_ref

    for idx, in_file in enumerate(in_file_list):
        img = Image.open(in_file)
        img.save(out_file_list[idx], icc_profile=dci_profile.tobytes())
        img.save(out_file_list2[idx], icc_profile=sRGB_profile.tobytes())


def extract_profile():
    filename = './picture/Webkit-logo-P3.png'
    img = Image.open(filename)
    out_file_name = './picture/dci_profile.icc'

    with open(out_file_name, 'wb') as f:
        f.write(img.info['icc_profile'])


if __name__ == '__main__':
    # gen_color_bar()
    # gen_saturation_color_bar()
    # add_profile()
    extract_profile()

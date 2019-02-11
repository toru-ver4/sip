#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ノイズの耐性を確認する
"""

import os
import cv2
import numpy as np
import TyImageIO as tyio
from colour import RGB_to_YCbCr, YCbCr_to_RGB
import test_pattern_generator2 as tpg


def get_nominal_peak(img):
    try:
        peak = np.iinfo(img.dtype).max
    except ValueError:
        peak = 1.0
    return peak


def normalize_img_by_dtype(img):
    peak = get_nominal_peak(img)
    return np.float64(img) / peak


def read_linear_src_img():
    """
    sourceの画像を読み込む。更に正規化して[0:1]にする。
    """
    file_name = "./src_img/src_img.exr"
    reader = tyio.TyReader(file_name)
    img = reader.read()
    img = normalize_img_by_dtype(img)

    return img


def video_camera_emulation(img, gamma=2.4, out_bit_depth=10):
    """
    Linear Lihgt を受け取って、ガンマ補正して、RGB2YCbCr 変換する。
    """
    non_linear_img = img ** (1/gamma)
    ycbcr = RGB_to_YCbCr(non_linear_img, out_bits=out_bit_depth,
                         out_legal=True, out_int=True)

    # crycb = np.dstack((ycbcr[..., 2], ycbcr[..., 0], ycbcr[..., 1]))
    # tpg.preview_image(crycb / 1023, order='rgb')

    return ycbcr


def noise_addition(img, bit_depth=10, noise_rate=0.01, seed=0):
    """
    ガウシアンノイズを画像に加える
    """
    img_max_val = (2 ** bit_depth) - 1
    sigma = noise_rate * img_max_val
    np.random.seed(seed)
    noise = np.random.normal(0.0, sigma, img.shape)
    noise = np.round(noise).astype(np.int32)
    noised_img = np.int32(img) + noise
    noised_img = np.clip(noised_img, a_min=0, a_max=img_max_val)

    return noised_img


def display_eulation(img, gamma=2.4, in_bit_depth=10):
    """
    YCbCr2RGB 変換してInverseガンマ補正して Linear Light を復元する。
    """
    non_linear_img = YCbCr_to_RGB(img, in_bits=in_bit_depth, in_int=True,
                                  out_legal=False, out_int=False)
    non_linear_img = np.clip(non_linear_img, a_min=0.0, a_max=1.0)

    # tpg.preview_image(non_linear_img, 'rgb')
    linear_img = non_linear_img ** gamma

    return linear_img


def save_img(img, fname="./out_still/hoge.tiff"):
    attr = {'Compression': 'none'}
    writer = tyio.TyWriter(img, fname, attr)
    writer.write()


def save_noised_img(out_name, gamma=2.4, noise_rate=0.05, seed=0):
    src_linear_img = read_linear_src_img()
    encoded_img = video_camera_emulation(img=src_linear_img, gamma=gamma)
    encoded_noise_img = noise_addition(encoded_img, bit_depth=10,
                                       noise_rate=noise_rate, seed=seed)
    dst_linear_img = display_eulation(img=encoded_noise_img, gamma=gamma)
    save_img(dst_linear_img ** (1/2.4), out_name)


def noised_movie(gamma=2.4, noise_rate=0.05):
    fps = 60
    sec = 3
    frame = fps * sec

    for idx in range(frame):
        fname = "./out_sequential_still/gamma_{:02d}_rate_{:.02f}_{:06d}.tiff"
        fname = fname.format(int(gamma*10), noise_rate, idx)
        print(fname)
        save_noised_img(fname, gamma, noise_rate, seed=idx)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    noise = 0.05
    # save_noised_img("./out_still/gamm24.tiff", gamma=2.4, seed=0,
    #                 noise_rate=noise)
    # save_noised_img("./out_still/gamm10.tiff", gamma=1.0, seed=0,
    #                 noise_rate=noise)
    noised_movie(gamma=2.4, noise_rate=0.05)
    noised_movie(gamma=1.0, noise_rate=0.05)

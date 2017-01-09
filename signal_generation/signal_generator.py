#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
捨てるには惜しい。きちんと管理するには微妙。
そんな出来損ないのコード群。マニュアルなんて無いよっ！
"""

import os
import sys
import shutil
import subprocess
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imp
import test_pattern_generator as tpg
imp.reload(tpg)


def copy_movie_seq_file():
    """ffmpegに静止画を食わせるために連番ファイルを作る"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    in_name = "./ffmpeg_tiff/source.tiff"
    for idx in range(60):
        out_name = "./ffmpeg_tiff/img/hdr_img_{:08d}.tiff".format(idx)
        shutil.copyfile(in_name, out_name)


def brightness_limitter_test_pattern():
    """TVのブライトネスリミットの掛かり具合を調べるパターンを作る"""
    sec = 5
    fps = 60
    frame = sec * fps
    x = np.arange(frame) / (frame - 1)
    x = np.concatenate((x, x[::-1], x, x[::-1]))
    x = x ** 2.0
    counter = 0
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for size in x:
        img = tpg.gen_youtube_hdr_test_pattern(high_bit_num=6,
                                               window_size=float(size))
        out_name = "./ffmpeg_tiff/img/hdr_img_{:08d}.tiff".format(counter)
        cv2.imwrite(out_name, img)
        counter += 1


def encode_hdr_movie():
    """YouTubeにアップ出来る形式で動画を作る"""
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/ffmpeg_tiff")

    ext_cmd = ['ffmpeg', '-r', '24', '-i', 'img/hdr_img_%8d.tiff',
               '-i', 'bgm.wav', '-ar', '48000', '-ac', '2', '-c:a', 'aac',
               '-b:a', '384k',
               '-r', '24', '-vcodec', 'prores_ks', '-profile:v', '3',
               '-pix_fmt', 'yuv422p10le',
               '-b:v', '85000k', '-shortest', '-y', 'out.mkv']
    p = subprocess.Popen(ext_cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, universal_newlines=True)
    # while True:
    #     line = p.stdout.readline().rstrip()
    #     if not line:
    #         break
    #     print(line.decode())
    for line in p.stdout:
        print(line.rstrip())

    p.wait()

    ext_cmd = ['mkvmerge.exe',
               '-o', 'hdr_movie.mkv',
               '--colour-matrix', '0:9',
               '--colour-range', '0:1',
               '--colour-transfer-characteristics', '0:16',
               '--colour-primaries', '0:9',
               '--max-content-light', '0:1000',
               '--max-frame-light', '0:300',
               '--max-luminance', '0:1000',
               '--min-luminance', '0:0.01',
               '--chromaticity-coordinates',
               '0:0.68,0.32,0.265,0.690,0.15,0.06',
               '--white-colour-coordinates', '0:0.3127,0.3290',
               'out.mkv']
    p = subprocess.Popen(ext_cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, universal_newlines=True)

    for line in p.stdout:
        print(line.rstrip())

    p.wait()


if __name__ == '__main__':
    # 静止画のHDR確認動画を生成
    # -------------------------
    copy_movie_seq_file()
    encode_hdr_movie()

    # 白ベタWindow がサイズを変えるHDR確認動画を生成
    # -------------------------------------------
    # brightness_limitter_test_pattern()
    # encode_hdr_movie()

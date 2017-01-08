#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
捨てるには惜しい。きちんと管理するには微妙。
そんな出来損ないのコード群。マニュアルなんて無いよっ！
"""

import os
import shutil
import subprocess


def copy_movie_seq_file():
    """ffmpegに静止画を食わせるために連番ファイルを作る"""
    os.chdir(os.path.dirname(__file__))
    in_name = "./ffmpeg_tiff/source.tiff"
    for idx in range(60):
        out_name = "./ffmpeg_tiff/img/hdr_img_{:08d}.tiff".format(idx)
        shutil.copyfile(in_name, out_name)


def encode_hdr_movie():
    """YouTubeにアップ出来る形式で動画を作る"""
    os.chdir(os.path.dirname(__file__) + "/ffmpeg_tiff")

    ext_cmd = ['ffmpeg', '-r', '3', '-i', 'img/hdr_img_%8d.tiff',
               '-i', 'bgm.wav', '-ar', '48000', '-ac', '2', '-c:a', 'aac',
               '-b:a', '384k',
               '-r', '24', '-vcodec', 'prores_ks', '-profile:v', '3',
               '-pix_fmt', 'yuv422p10le',
               '-b:v', '50000k', '-shortest', '-y', 'out.mkv']
    p = subprocess.Popen(ext_cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    # out, err = p.communicate()
    for line in p.stdout:
        print(line.decode())

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
                         stderr=subprocess.STDOUT)

    for line in p.stdout:
        print(line.decode())

    p.wait()


if __name__ == '__main__':
    copy_movie_seq_file()
    encode_hdr_movie()

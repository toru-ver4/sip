#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
画面キャプチャしたりしなかったり。
"""

import os 
import sys
import subprocess


def capture_desktiop():
    command_list = 'ffmpeg -f gdigrab -draw_mouse 1 -show_region 1 ' \
                   + '-framerate 30 -video_size 1280x720 '\
                   + '-offset_x 0 -offset_y 60 '\
                   + '-i desktop -c:v libx264 -crf 24 -pix_fmt yuv420p out.mp4'
    try:
        subprocess.check_output(command_list.split(" "))
    except subprocess.CalledProcessError as e:
        print('error! "{:s}" is failed. ret = {}'.format(" ".join(e.cmd), e.returncode))


def streaming_desktop():
    command_list = 'ffmpeg -f gdigrab -draw_mouse 1 -show_region 1 ' \
                   + '-framerate 30 -video_size 1280x720 '\
                   + '-offset_x 0 -offset_y 60 '\
                   + '-i desktop -c:v libx264 -crf 24 -f:v mpegts udp://192.168.100.168:10722'
    try:
        subprocess.check_output(command_list.split(" "))
    except subprocess.CalledProcessError as e:
        print('error! "{:s}" is failed. ret = {}'.format(" ".join(e.cmd), e.returncode))
    

if __name__ == '__main__':
    # capture_desktiop()
    streaming_desktop()

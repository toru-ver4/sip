#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ITEの画像をいい感じにコンバートする
あらかじめ、srcフォルダに以下の3ファイルをDLしておくこと。

* http://www.ite.or.jp/contents/chart/uhdtv/u10_Ship_4K.r
* http://www.ite.or.jp/contents/chart/uhdtv/u10_Ship_4K.g
* http://www.ite.or.jp/contents/chart/uhdtv/u10_Ship_4K.b
"""

import os
import sys
import struct
import numpy as np
import cv2


def preview_image(img, order='rgb', over_disp=False):
    """ OpenCV の機能を使って画像をプレビューする """
    if order == 'rgb':
        cv2.imshow('preview', img[:, :, ::-1])
    elif order == 'bgr':
        cv2.imshow('preview', img)
    else:
        raise ValueError("order parameter is invalid")

    if over_disp:
        cv2.resizeWindow('preview', )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_12bit_raw_mono_data(filename):
    """
    単色の 12bit rawデータを読み込む。
    仕様は http://www.ite.or.jp/contents/chart/uhdtv/manual.pdf の
    図2-1 を参照。

    一応、おもてなしとして解像度に合わせた2次元配列にして返す。

    Parameters
    ------------
    filename : strings
        filename.

    Returns
    ------------
    array_like
        (height, width, 1) の16bit画像データ。
        有効データは[0:11]。上位4bitは空っぽ。
    """

    # ファイルサイズから解像度を推定＆パラメータ設定
    # -------------------------------------------
    file_size = os.path.getsize(filename)
    if int(file_size / 2 / 3840) == 2160:
        width = 3840
        height = 2160
    elif int(file_size / 2 / 7680) == 4320:
        width = 7680
        height = 4320
    else:
        print("File Size ERRROR!")
        sys.exit(1)

    unpack_str = ">" + str(width * height) + "H"

    # 一気に全データをRead
    # --------------------------------------
    raw_data = open(filename, 'rb').read()

    # 我々にも解釈できるように16bitの配列としてデコード
    # ---------------------------------------------
    data = struct.unpack(unpack_str, raw_data)

    # width, height を使って2次元(3次元)配列化
    # ---------------------------------------------
    data = np.array(data, dtype=np.uint16).reshape(height, width, 1)

    return data


def get_12bit_raw_color_data(filename_prefix):
    """
    単色で存在する 12bit rawデータを結合してカラーデータを作る。
    仕様は http://www.ite.or.jp/contents/chart/uhdtv/manual.pdf の
    図2-1 を参照。

    一応、おもてなしとして解像度に合わせた3次元配列するが、
    データは下位12bit詰めであり、上位4bitは空であることに注意すること。

    Parameters
    ------------
    filename : strings
        filename prefix.

    Returns
    ------------
    array_like
        (height, width, 1) の16bit画像データ。
        有効データは[0:11]。上位4bitは空っぽ。
    """

    img = []
    extensions = ["r", "g", "b"]

    # R, G, B 各色のデータを img配列に入れる
    # --------------------------------------
    for ext in extensions:
        filename = filename_prefix + "." + ext
        img.append(get_12bit_raw_mono_data(filename))

    # 各色のデータをカラーデータとして結合
    # ----------------------------------------
    img = np.dstack((img[0], img[1], img[2]))

    # 画像のプレビュー
    # -----------------------------------------
    # preview_image(img * (2 ** 4))

    return img


def save_12bit_raw_color_data(filename_prefix):
    img = get_12bit_raw_color_data(filename_prefix)
    out_name = filename_prefix + ".tiff"

    # 12bit のデータを 16bit に変換
    out_img = img * (2 ** 4)

    # OpenCVのライブラリを使って保存
    # OpenCV は "RGB" ではなく "BGR" の順序なので
    # 並べ替えたデータを渡す
    # -----------------------------
    cv2.imwrite(out_name, out_img[:, :, ::-1])


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # get_12bit_raw_color_data("./src/u10_Ship_4K")
    save_12bit_raw_color_data("./src/u10_Ship_4K")

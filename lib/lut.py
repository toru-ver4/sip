#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## 概要

* 1DLUT/3DLUT の Write/Read をするモジュール
* 異なる形式への変換もできるよ！

## 仕様

### サポート形式

1DLUT では以下の形式をサポートする

* .cube (Adobe)
* .spi1d


3DLUT では以下の形式をサポートする

* .3dl (Lustre)
* .cube (Adobe)
* .spi3d (SPI)

### データの保持について

ライブラリ内部では以下の仕様でデータを持つ

### 1DLUT

* dtype: double
* array: [idx][3]

### 3DLUT

* dtype: double
* array: [idx][3]
  * idx は R -> G -> B の順に増加する
  * cube形式と同じ順序

## 使い方

本ソースコードのmainを参照。

* 3DLUTの新規作成
* 3DLUTの形式変換
* 1DLUTの新規作成
* 1DLUTの形式変換

## references
[3dl](http://download.autodesk.com/us/systemdocs/help/2009/lustre_ext1/index.html?url=WSc4e151a45a3b785a24c3d9a411df9298473-7ffd.htm,topicNumber=d0e8061)
[cube](http://wwwimages.adobe.com/www.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf)

"""

import os
import numpy as np


def get_3d_grid_cube_format(grid_num=4):
    """
    # 概要
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), ...
    みたいな配列を返す。
    CUBE形式の3DLUTを作成する時に便利。
    """

    base = np.linspace(0, 1, grid_num)
    ones_x = np.ones((grid_num, grid_num, 1))
    ones_y = np.ones((grid_num, 1, grid_num))
    ones_z = np.ones((1, grid_num, grid_num))
    r_3d = base[np.newaxis, np.newaxis, :] * ones_x
    g_3d = base[np.newaxis, :, np.newaxis] * ones_y
    b_3d = base[:, np.newaxis, np.newaxis] * ones_z
    r_3d = r_3d.flatten()
    g_3d = g_3d.flatten()
    b_3d = b_3d.flatten()

    grid = np.dstack((r_3d, g_3d, b_3d))
    grid = grid.reshape((grid_num ** 3, 3))

    return grid


def _convert_3dlut_from_cube_to_3dl(lut, grid_num):
    """
    cube形式(R -> G -> B 順で増加) のデータを
    3dl形式(B -> G -> R) に変換

    Parameters
    ----------
    lut : array_like
        3DLUT data with cube format.
    grid_num : int
        grid number of the 3dlut.

    Returns
    -------
    array_like
        3DLUT data with 3dl format.
    """

    out_lut = lut.reshape((grid_num, grid_num, grid_num, 3), order="F")
    out_lut = out_lut.reshape((grid_num ** 3, 3))

    return out_lut


def _convert_3dlut_from_3dl_to_cube(lut, grid_num):
    """
    3dl形式(B -> G -> R) のデータを
    cube形式(R -> G -> B 順で増加) に変換

    Parameters
    ----------
    lut : array_like
        3DLUT data with 3dl format.
    grid_num : int
        grid number of the 3dlut.

    Returns
    -------
    array_like
        3DLUT data with cube format.
    """

    out_lut = _convert_3dlut_from_cube_to_3dl(lut, grid_num)

    return out_lut


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    g_num = 3
    lut = get_3d_grid_cube_format(grid_num=g_num) * g_num
    dl_lut = _convert_3dlut_from_cube_to_3dl(lut, g_num)
    lut222 = _convert_3dlut_from_3dl_to_cube(dl_lut, g_num)


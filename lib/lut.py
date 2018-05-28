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

AUTHOR_INFORMATION = "# This 3DLUT data was created by TY-LUT creation tool"


def get_3d_grid_cube_format(grid_num=4):
    """
    .cubeフォーマットに準じた3DLUTの格子データを出力する。
    grid_num=3 の場の例を以下に示す。

    ```
    [[ 0.   0.   0. ]
     [ 0.5  0.   0. ]
     [ 1.   0.   0. ]
     [ 0.   0.5  0. ]
     [ 0.5  0.5  0. ]
     [ 1.   0.5  0. ]
     [ 0.   1.   0. ]
     [ 0.5  1.   0. ]
     [ 1.   1.   0. ]
     [ 0.   0.   0.5]

     中略

     [ 1.   0.5  1. ]
     [ 0.   1.   1. ]
     [ 0.5  1.   1. ]
     [ 1.   1.   1. ]]
     ```

    Parameters
    ----------
    grid_num : int
        grid number of the 3dlut.

    Returns
    -------
    array_like
        3DLUT grid data with cube format.
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


def save_3dlut(lut, grid_num, filename="./data/lut_sample/cube.cube",
               title=None, min=0.0, max=1.0):
    """
    3DLUTデータをファイルに保存する。
    形式の判定はファイル名の拡張子で行う。

    Parameters
    ----------
    filename : str
        file name.
    lut : array_like
        3dlut data.
    grid_num : int
        grid number.
    title : str
        title of the 3dlut data. It is for header information.
    min : int or float
        minimum value of the 3dlut
    max : int or float
        maximu value of the 3dlut
    """

    root, ext = os.path.splitext(filename)

    if ext == ".cube":
        print("write cube")
        save_3dlut_cube_format(lut, grid_num, filename=filename,
                               title="cube_test", min=min, max=max)
    elif ext == ".3dl":
        print("3dl")
        pass
    elif ext == ".spi3d":
        save_3dlut_spi_format(lut, grid_num, filename=filename,
                              title="spi3d_test", min=min, max=max)
        print("spi3d")
        pass
    else:
        raise IOError('extension "{:s}" is not supported.'.format(ext))


def save_3dlut_cube_format(lut, grid_num, filename,
                           title=None, min=0.0, max=1.0):
    """
    CUBE形式で3DLUTデータをファイルに保存する。

    Parameters
    ----------
    filename : str
        file name.
    lut : array_like
        3dlut data.
    grid_num : int
        grid number.
    title : str
        title of the 3dlut data. It is for header information.
    min : int or float
        minimum value of the 3dlut
    max : int or float
        maximu value of the 3dlut
    """

    # ヘッダ情報の作成
    # ------------------------
    header = ""
    header += AUTHOR_INFORMATION + '\n'

    if title:
        header += 'TITLE "{:s}"\n'.format(title)
    header += 'DOMAIN_MIN {0:} {0:} {0:}\n'.format(min)
    header += 'DOMAIN_MAX {0:} {0:} {0:}\n'.format(max)
    header += 'LUT_3D_SIZE {:}\n'.format(grid_num)
    header += '\n'

    # ファイルにデータを書き込む
    # ------------------------
    out_str = '{:.10f} {:.10f} {:.10f}\n'
    with open(filename, "w") as file:
        file.write(header)
        for line in lut:
            file.write(out_str.format(line[0], line[1], line[2]))


def save_3dlut_spi_format(lut, grid_num, filename,
                          title=None, min=0.0, max=1.0):
    """
    spi3d形式で3DLUTデータをファイルに保存する。

    Parameters
    ----------
    filename : str
        file name.
    lut : array_like
        3dlut data.
    grid_num : int
        grid number.
    title : str
        title of the 3dlut data. It is for header information.
    min : int or float
        minimum value of the 3dlut
    max : int or float
        maximu value of the 3dlut
    """

    # ヘッダ情報の作成
    # ------------------------
    header = ""
    header += 'SPILUT 1.0\n'
    header += '{:d} {:d}\n'.format(3, 3)  # 数値の意味は不明
    header += '{0:d} {0:d} {0:d}\n'.format(grid_num)

    # ファイルにデータを書き込む
    # ------------------------
    line_index = 0
    out_str = '{:d} {:d} {:d} {:.10f} {:.10f} {:.10f}\n'
    with open(filename, "w") as file:
        file.write(header)
        for line in lut:
            r_idx, g_idx, b_idx\
                = _get_rgb_index_for_spi3d_output(line_index, grid_num)
            file.write(out_str.format(r_idx, g_idx, b_idx,
                                      line[0], line[1], line[2]))
            line_index += 1


def _get_rgb_index_for_spi3d_output(line_index, grid_num):
    """
    3DLUT Data の行番号から、当該行に該当する r_idx, g_idx, b_idx を
    算出する。

    Parameters
    ----------
    line_index : int
        line number.
    grid_num : int
        grid number.

    Returns
    -------
    int, int, int
        grid index of each color.
    """

    r_idx = (line_index // (grid_num ** 2)) % grid_num
    g_idx = (line_index // (grid_num ** 1)) % grid_num
    b_idx = (line_index // (grid_num ** 0)) % grid_num

    return r_idx, g_idx, b_idx


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    g_num = 3
    lut = get_3d_grid_cube_format(grid_num=g_num)
    save_3dlut(lut, g_num, filename="./data/lut_sample/hoge.fuga.cube", min=-0.1, max=1.0)
    save_3dlut(lut, g_num, filename="./data/lut_sample/hoge.fuga.3dl")
    save_3dlut(lut, g_num, filename="./data/lut_sample/hoge.fuga.spi3d")
    save_3dlut(lut, g_num, filename="./data/lut_sample/hoge.fuga.spi1d")

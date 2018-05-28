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
import math

AUTHOR_INFORMATION = "This 3DLUT data was created by TY-LUT creation tool"
LUT_BIT_DEPTH_3DL = 16


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


def _is_float_expression(s):
    """
    copied from
    https://qiita.com/agnelloxy/items/137cbc8651ff4931258f
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


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
        save_3dlut_cube_format(lut, grid_num, filename=filename,
                               title="cube_test", min=min, max=max)
    elif ext == ".3dl":
        save_3dlut_3dl_format(lut, grid_num, filename=filename,
                              title="cube_test", min=min, max=max)
    elif ext == ".spi3d":
        save_3dlut_spi_format(lut, grid_num, filename=filename,
                              title="spi3d_test", min=min, max=max)
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
    header += '# ' + AUTHOR_INFORMATION + '\n'

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


def load_3dlut_cube_format(filename):
    """
    CUBE形式の3DLUTデータをファイルから読み込む。

    Parameters
    ----------
    filename : str
        file name.

    Returns
    -------
    lut : array_like
        3DLUT data with cube format.
    grid_num : int
        grid number.
    title : str
        title of the 3dlut data.
    min : double
        minium value of the 3dlut data.
    max : double
        maximum value of the 3dlut data.
    """

    # ヘッダ情報を読みつつ、データ開始位置を探る
    # --------------------------------------
    data_start_idx = 0
    title = None
    min = 0.0
    max = 1.0
    grid_num = None
    with open(filename, "r") as file:
        for line_idx, line in enumerate(file):
            line = line.rstrip()
            if line == '':  # 空行は飛ばす
                continue
            key_value = line.split()[0]
            if key_value == 'TITLE':
                title = line.split()[1]
            if key_value == 'DOMAIN_MIN':
                min = float(line.split()[1])
            if key_value == 'DOMAIN_MAX':
                max = float(line.split()[1])
            if key_value == 'LUT_3D_SIZE':
                grid_num = int(line.split()[1])
            if _is_float_expression(line.split()[0]):
                data_start_idx = line_idx
                break

    # 3DLUTデータを読む
    # --------------------------------------
    lut = np.loadtxt(filename, delimiter=' ', skiprows=data_start_idx)

    # 得られたデータを返す
    # --------------------------------------
    return lut, grid_num, title, min, max


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

    # 3dl形式へLUTデータの並べ替えをする
    # --------------------------------
    out_lut = _convert_3dlut_from_cube_to_3dl(lut, grid_num)

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
        for line in out_lut:
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


def save_3dlut_3dl_format(lut, grid_num, filename,
                          title=None, min=0.0, max=1.0):
    """
    3DL形式で3DLUTデータをファイルに保存する。

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

    # 3dl形式へLUTデータの並べ替えをする
    # --------------------------------
    out_lut = _convert_3dlut_from_cube_to_3dl(lut, grid_num)

    # ヘッダ情報の作成
    # ------------------------
    header = ""
    exponent = round(math.log2(grid_num - 1))
    bit_depth = LUT_BIT_DEPTH_3DL

    if title:
        header += '# TITLE "{:s}"\n'.format(title)
    header += '# ' + AUTHOR_INFORMATION + '\n'
    header += '\n'
    header += '3DMESH\n'
    header += 'Mesh {:d} {:d}\n'.format(exponent, bit_depth)
    header += '\n'

    # データを出力bit精度に変換
    # ------------------------
    out_lut = np.uint32(np.round(out_lut * ((2 ** bit_depth) - 1)))

    # ファイルにデータを書き込む
    # ------------------------
    out_str = '{:d} {:d} {:d}\n'
    with open(filename, "w") as file:
        file.write(header)
        for line in out_lut:
            file.write(out_str.format(line[0], line[1], line[2]))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    g_num = 3
    sample_arri_cube = "./data/lut_sample/AlexaV3_EI0800_LogC2Video_Rec709_LL_aftereffects3d.cube"
    lut = get_3d_grid_cube_format(grid_num=g_num)
    save_3dlut(lut, g_num, filename="./data/lut_sample/hoge.fuga.cube",
               min=-0.1, max=1.0)
    # load_3dlut_cube_format("./data/lut_sample/hoge.fuga.cube")
    lut, grid_num, title, min, max = load_3dlut_cube_format(sample_arri_cube)
    print(lut.shape)
    print(grid_num, title, min, max)
    # save_3dlut(lut, g_num, filename="./data/lut_sample/hoge.fuga.3dl")
    # save_3dlut(lut, g_num, filename="./data/lut_sample/hoge.fuga.spi3d")
    # save_3dlut(lut, g_num, filename="./data/lut_sample/hoge.fuga.spi1d")

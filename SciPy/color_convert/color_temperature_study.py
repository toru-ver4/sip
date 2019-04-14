import os
import sys
import cv2
import numpy as np
from scipy import linalg
from scipy import integrate
import matplotlib.pyplot as plt


const_sRGB_xy = [[0.64, 0.33],
                 [0.30, 0.60],
                 [0.15, 0.06],
                 [0.3127, 0.3290]]

const_ntsc_xy = [[0.67, 0.33],
                 [0.21, 0.71],
                 [0.14, 0.08],
                 [0.310, 0.316]]

const_rec601_xy = const_ntsc_xy

const_rec709_xy = const_sRGB_xy

const_rec2020_xy = [[0.708, 0.292],
                    [0.170, 0.797],
                    [0.131, 0.046],
                    [0.3127, 0.3290]]

const_sRGB_xyz = [[0.64, 0.33, 0.03],
                  [0.30, 0.60, 0.10],
                  [0.15, 0.06, 0.79],
                  [0.3127, 0.3290, 0.3583]]

const_xyz_to_lms = [[0.8951000, 0.2664000, -0.1614000],
                    [-0.7502000, 1.7135000, 0.0367000],
                    [0.0389000, -0.0685000, 1.0296000]]

const_rgb_to_large_xyz = [[2.7689, 1.7517, 1.1302],
                          [1.0000, 4.5907, 0.0601],
                          [0.0000, 0.0565, 5.5943]]

const_d65_xy = [0.31271, 0.32902]
const_d50_xy = [0.34567, 0.35850]

const_rec601_y_coef = [0.2990, 0.5870, 0.1140]
const_rec709_y_coef = [0.2126, 0.7152, 0.0722]


def xy_to_xyz_internal(xy):
    rz = 1 - (xy[0][0] + xy[0][1])
    gz = 1 - (xy[1][0] + xy[1][1])
    bz = 1 - (xy[2][0] + xy[2][1])
    wz = 1 - (xy[3][0] + xy[3][1])

    xyz = [[xy[0][0], xy[0][1], rz],
           [xy[1][0], xy[1][1], gz],
           [xy[2][0], xy[2][1], bz],
           [xy[3][0], xy[3][1], wz]]

    return xyz


def get_rgb_to_xyz_matrix(gamut=const_sRGB_xy):

    # まずは xyz 座標を準備
    # ------------------------------------------------
    if np.array(gamut).shape == (4, 2):
        gamut = xy_to_xyz_internal(gamut)
    elif np.array(gamut).shape == (4, 3):
        pass
    else:
        print("============ Fatal Error ============")
        print("invalid xy gamut parameter.")
        print("=====================================")
        sys.exit(1)

    gamut_mtx = np.array(gamut)

    # 白色点の XYZ を算出。Y=1 となるように調整
    # ------------------------------------------------
    large_xyz = [gamut_mtx[3][0]/gamut_mtx[3][1],
                 gamut_mtx[3][1]/gamut_mtx[3][1],
                 gamut_mtx[3][2]/gamut_mtx[3][1]]
    large_xyz = np.array(large_xyz)

    # Sr, Sg, Sb を算出
    # ------------------------------------------------
    s = linalg.inv(gamut_mtx[0:3]).T.dot(large_xyz)

    # RGB2XYZ 行列を算出
    # ------------------------------------------------
    s_matrix = [[s[0], 0.0,  0.0],
                [0.0,  s[1], 0.0],
                [0.0,  0.0,  s[2]]]
    s_matrix = np.array(s_matrix)
    rgb2xyz_mtx = gamut_mtx[0:3].T.dot(s_matrix)

    return rgb2xyz_mtx


def get_yuv_trans_coef(gamut=const_sRGB_xy):
    """
    # 概要
    YUV変換の係数を色域から算出する
    # 参考URL：http://www.ite.or.jp/study/musen/tips/tip07.html
    """
    gamut = np.array(xy_to_xyz(gamut))

    # y=1 で正規化した行列を用意
    # ------------------------------------------------
    y_is_1_xyz_of_rgb = np.array([x / x[1] for x in gamut[0:3]]).T
    y_is_1_xyz_of_w = gamut[3] / gamut[3][1]

    y_coef = linalg.inv(y_is_1_xyz_of_rgb).dot(y_is_1_xyz_of_w)

    return y_coef


def xy_to_xyz(gamut):
    """
    # 概要
    xy座標だった場合はxyzに変換して出力する。
    """
    if np.array(gamut).shape == (4, 2):
        gamut = xy_to_xyz_internal(gamut)
    elif np.array(gamut).shape == (4, 3):
        pass
    else:
        raise ValueError

    return gamut


def large_xyz_to_small_xyz(x, y, z):

    xyz_sum = x + y + z
    small_x = x / xyz_sum
    small_y = y / xyz_sum
    small_z = z / xyz_sum

    return small_x, small_y, small_z


def kelvin_to_rgb(t=6500):
    """
    # 概要
    色温度からRGB刺激値を算出する
    # 参考URL
    http://cafe.mis.ous.ac.jp/sawami/%E9%BB%92%E4%BD%93%E8%BC%BB%E5%B0%84.PDF
    """

    h = 6.6260755E-34
    k = 1.380658E-23
    c = 2.99792458E+08
    bunbo = 8 * np.pi * h * c
    r = bunbo / ((0.700E-06 ** 5) * (np.exp(h*c/(k*t*700E-09))-1))
    g = bunbo / ((0.546E-06 ** 5) * (np.exp(h*c/(k*t*546E-09))-1))
    b = bunbo / ((0.436E-06 ** 5) * (np.exp(h*c/(k*t*436E-09))-1))

    return r, g, b


def kelvin_to_xy_chromaticity(t=6500):
    rgb = np.array(kelvin_to_rgb(t=t))
    print(rgb)
    large_xyz = np.array(const_rgb_to_large_xyz).dot(rgb)
    print(large_xyz)
    x, y, z = large_xyz_to_small_xyz(*large_xyz)
    print(x, y, z)


def sekibun_test():
    os.chdir(os.path.dirname(__file__))
    data = np.loadtxt(fname="./data/lin2012xyz2e_fine_7sf.csv",
                      delimiter=',', skiprows=2).T[1:]
    print(data.shape)
    result = [integrate.simps(x) for x in data]
    print(result)

    x = np.arange(11)
    print(integrate.simps(x))


def get_large_xyz_t():
    """
    # 概要
    
    """

if __name__ == '__main__':
    # kelvin_to_xy_chromaticity(t=10000)
    # kelvin_to_xy_chromaticity(t=6500)
    # kelvin_to_xy_chromaticity(t=3000)
    sekibun_test()

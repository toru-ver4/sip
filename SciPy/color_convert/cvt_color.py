import os
import sys
import numpy as np
from scipy import linalg

const_sRGB_xy = [[0.64, 0.33],
                 [0.30, 0.60],
                 [0.15, 0.06],
                 [0.3127, 0.3290]]

const_sRGB_xyz = [[0.64, 0.33, 0.03],
                  [0.30, 0.60, 0.10],
                  [0.15, 0.06, 0.79],
                  [0.3127, 0.3290, 0.3583]]


def xy_to_xyz(xy):
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
        gamut = xy_to_xyz(gamut)
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


def inv_test():
    mtx = [[0.2126, 0.7152, 0.0722],
           [-0.114572, -0.385428, 0.5],
           [0.5, -0.454153, -0.045847]]
    mtx = np.array(mtx)
    inv_mtx = linalg.inv(mtx)
    print(mtx)
    print(inv_mtx)


if __name__ == '__main__':
    get_rgb_to_xyz_matrix(gamut=const_sRGB_xy)

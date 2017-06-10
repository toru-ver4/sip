import os
import imp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import common
import plot_utility as pu
import color_convert as ccv

imp.reload(ccv)
imp.reload(pu)


def cross_test():
    """
    # 概要
    外積の動作確認
    """
    r = np.array(ccv.const_rec2020_xy[0])
    g = np.array(ccv.const_rec2020_xy[1])
    b = np.array(ccv.const_rec2020_xy[2])
    # w = np.array([0.3127, 0.3290])
    w = np.array([0.15, 0.7])

    rg = g - r
    gw = w - g
    gb = b - g
    bw = w - b
    br = r - b
    rw = w - r

    r_result = np.cross(rg, gw)
    g_result = np.cross(gb, bw)
    b_result = np.cross(br, rw)
    print(r_result, g_result, b_result)

    print(r, g, b, w)


def is_inside_gamut(xy, gamut=np.array(ccv.const_rec2020_xy)):
    """
    # 概要
    xy座標が gamut 内部にあるか判別する
    # In/Out
    xy は (N, 2) の numpy 配列であること
    # 参考
    http://www.sousakuba.com/Programming/gs_hittest_point_triangle.html
    """
    # parameter check
    # -----------------------------------------
    if not common.is_small_xy_array_shape(xy):
        print('parameer "xy" is invalid.')
        return False

    if not common.is_small_xy_array_shape(gamut):
        print('parameer "gamut" is invalid.')
        return False

    if gamut.shape[1] != 2:
        print('parameer "gamut" is invalid.')
        return False

    # calc vector
    # -----------------------------------------
    rg = gamut[1] - gamut[0]
    gw = xy - gamut[1]
    gb = gamut[2] - gamut[1]
    bw = xy - gamut[2]
    br = gamut[0] - gamut[2]
    rw = xy - gamut[0]

    r_result = np.cross(rg, gw)
    g_result = np.cross(gb, bw)
    b_result = np.cross(br, rw)

    print(r_result)
    print(g_result)
    print(b_result)

    print((r_result >= 0) & (g_result >= 0) & (b_result >= 0))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # cross_test()
    xy = np.array([[0.1, 0.3], [0.3, 0.3], [0.9, 0.3], [0.3, 0.5],
                   [0.708, 0.292], [0.170, 0.797], [0.131, 0.046],
                   [0.709, 0.292], [0.170, 0.798], [0.131, 0.045]])
    is_inside_gamut(xy=xy)

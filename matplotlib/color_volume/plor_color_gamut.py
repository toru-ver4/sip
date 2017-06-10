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


def get_xy_inside_gamut(gamut=ccv.const_rec2020_xy, div_num=110):
    """
    # 概要
    gamut の領域に入ってる xy値を得る
    # 詳細
    div_num は 各次元の分割数。
    div_num=100 の場合 100x100=10000 グリッドで計算する。
    """
    # 大元の xy データ作成
    # ----------------------------
    x = np.linspace(0, 1, div_num)
    x = x[np.newaxis, :]
    ones_x = np.ones((div_num, 1))
    y = np.linspace(0, 1, div_num)
    y = y[:, np.newaxis]
    ones_y = np.ones((1, div_num))
    x = x * ones_x
    y = y * ones_y
    xy = np.dstack((x, y))
    xy = xy.reshape((xy.shape[0] * xy.shape[1], xy.shape[2]))

    # 判定
    # ----------------------------
    ok_idx = ccv.is_inside_gamut(xy, gamut=ccv.const_rec2020_xy)
    xy = xy[ok_idx]

    ax1 = pu.plot_1_graph()
    ax1.plot(xy[:, 0], xy[:, 1], '.', markersize=10)
    plt.show()

    return xy


def get_max_rgb_from_xy(xy, gamut=ccv.const_rec2020_xy):
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # cross_test()
    xy = np.array([[0.1, 0.3], [0.3, 0.3], [0.9, 0.3], [0.3, 0.5],
                   [0.708, 0.292], [0.170, 0.797], [0.131, 0.046],
                   [0.709, 0.292], [0.170, 0.798], [0.131, 0.045]])
    ccv.is_inside_gamut(xy=xy)
    get_xy_inside_gamut(div_num=100)

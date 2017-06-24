import os
import imp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def get_xy_gamut(div_num=5):
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

    return xy


def get_xy_inside_gamut(gamut=ccv.const_rec2020_xy, div_num=110, plot=False):
    """
    # 概要
    gamut の領域に入ってる xy値を得る
    # 詳細
    div_num は 各次元の分割数。
    div_num=100 の場合 100x100=10000 グリッドで計算する。
    """
    # 大元の xy データ作成
    # ----------------------------
    xy = get_xy_gamut(div_num)

    # 判定
    # ----------------------------
    ok_idx = ccv.is_inside_gamut(xy, gamut=ccv.const_rec2020_xy)
    xy = xy[ok_idx]

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="Title",
                              graph_title_size=None,
                              xlabel="X Axis Label", ylabel="Y Axis Label",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=(0, 0.8),
                              ylim=(0, 0.9),
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=3,
                              prop_cycle=None)
        ax1.plot(xy[:, 0], xy[:, 1], '.', markersize=10)
        plt.show()

    return xy


def get_max_rgb_from_xy(xy, gamut=ccv.const_rec2020_xy,
                        white=ccv.const_d65_large_xyz, large_y=1.0):
    xyY = ccv.small_xy_to_xyY(xy=xy, large_y=large_y)
    rgb = ccv.xyY_to_RGB(xyY=xyY, gamut=gamut, white=white)

    n = np.max(rgb, axis=2)  # normalize val
    normalize_val = np.dstack((n, n, n))
    # outline_rgb = rgb
    outline_rgb = rgb / normalize_val

    return outline_rgb


def get_large_xyz_from_rgb(rgb, large_y_rate,
                           gamut=ccv.const_rec2020_xy,
                           white=ccv.const_d65_large_xyz):
    large_xyz = ccv.rgb_to_large_xyz(rgb=rgb, gamut=gamut, white=white)

    return large_xyz


def get_xyY_from_large_xyz(large_xyz):
    large_x, large_y, large_z = np.dsplit(large_xyz, 3)
    sum_xyz = large_x + large_y + large_z
    x = large_x / sum_xyz
    y = large_y / sum_xyz

    return np.dstack((x, y, large_y))


def plot_xyY(xyY, rgb=None, grid_num=16):
    x = xyY[:, :, 0].flatten()
    y = xyY[:, :, 1].flatten()
    large_y = xyY[:, :, 2].flatten()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    color_array = []
    for idx in range(x.shape[0]):
        if rgb is not None:
            val = np.uint8(np.round(rgb[0][idx] * 0xFF))
            color = "#{:02X}{:02X}{:02X}".format(val[0], val[1], val[2])
            color_array.append(color)
        else:
            color = "#000000"
            color_array.append(color)

    ax.scatter3D(x, y, large_y, c=color_array, edgecolors='face', alpha=0.5)
    # ax.plot_wireframe(x_2d, y_2d, large_y)
    # xyY = np.reshape(xyY, (grid_num, grid_num, 3))
    plt.show()


def plot_rgb_patch(rgb):
    rgb = np.uint8(rgb.copy() * 0xFF)
    v_num = 4
    h_num = 11
    plt.rcParams["font.size"] = 18
    f, axarr = plt.subplots(v_num, h_num, sharex='col', sharey='row',
                            figsize=(h_num * 5, v_num * 5))
    for idx in range(v_num * h_num):
        color = "#{:02X}{:02X}{:02X}".format(rgb[0][idx][0],
                                             rgb[0][idx][1],
                                             rgb[0][idx][2])
        h_idx = idx % h_num
        v_idx = idx // h_num
        axarr[v_idx, h_idx].add_patch(
            patches.Rectangle(
                (0, 0), 1.0, 1.0, edgecolor='#A0A0A0', facecolor=color
            )
        )
        axarr[v_idx, h_idx].spines['right'].set_color('None')
        axarr[v_idx, h_idx].spines['left'].set_color('None')
        axarr[v_idx, h_idx].spines['top'].set_color('None')
        axarr[v_idx, h_idx].spines['bottom'].set_color('None')
        axarr[v_idx, h_idx].tick_params(axis='x', which='both',
                                        top='off', bottom='off',
                                        labelbottom='off')
        axarr[v_idx, h_idx].tick_params(axis='y', which='both',
                                        left='off', right='off',
                                        labelleft='off')
    plt.show()


def get_3d_grid(grid_num=4):
    """
    # 概要
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), ...
    みたいな配列を返す。
    RGB全値が欲しい時とかに使えるよ！
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

    return np.dstack((b_3d, g_3d, r_3d))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # cross_test()
    gamut = ccv.const_rec2020_xy
    white = ccv.const_d65_large_xyz
    div_num = 32
    large_y = 100
    rgb = get_3d_grid(grid_num=div_num)
    large_xyz = ccv.rgb_to_large_xyz(rgb, gamut=gamut, white=white)
    large_y = large_xyz[:, :, 2]
    xy = ccv.large_xyz_to_small_xy(large_xyz)
    xy[:, 0, :] = 0
    xyY = np.dstack((xy, large_y))
    plot_xyY(xyY, rgb)

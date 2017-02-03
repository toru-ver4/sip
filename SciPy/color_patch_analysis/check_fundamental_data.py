import os
import imp
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import color_convert as ccv
import light as lit

imp.reload(ccv)
imp.reload(lit)


def show_color_patch_spectral_data():
    """
    # 概要
    RIT で公開されている color patch のスペクトルデータをプロットする
    """
    spectral_data = "./data/MacbethColorChecker_SpectralData.csv"
    data = np.loadtxt(spectral_data, delimiter=',', skiprows=3).T

    # plot
    # ----------------------------------
    v_num = 4
    h_num = 6
    plt.rcParams["font.size"] = 18
    f, axarr = plt.subplots(v_num, h_num, sharex='col', sharey='row',
                            figsize=(24, 16))
    for idx in range(24):
        h_idx = idx % h_num
        v_idx = idx // h_num
        axarr[v_idx, h_idx].grid()
        if v_idx == (v_num - 1):
            axarr[v_idx, h_idx].set_xlabel("wavelength [nm]")
        if h_idx == 0:
            axarr[v_idx, h_idx].set_ylabel("reflectance")
        axarr[v_idx, h_idx].set_xlim(380, 780)
        axarr[v_idx, h_idx].set_ylim(0, 1.0)
        axarr[v_idx, h_idx].set_xticks([400, 500, 600, 700])
        axarr[v_idx, h_idx].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axarr[v_idx, h_idx].plot(data[0], data[idx + 1])
    plt.show()


def make_color_patch_image():
    """
    # 概要
    ColorChcker の color patch を表示(sRGB色域想定)
    """
    # csv ファイルから xyY データを取得
    # --------------------------------
    color_patch_xyY_data = "./data/ColorCheckerDataGretag.csv"
    xyY_data = np.loadtxt(color_patch_xyY_data, delimiter=',', skiprows=1,
                          usecols=(2, 3, 4))
    xyY_data = xyY_data.reshape((1, xyY_data.shape[0], xyY_data.shape[1]))

    # XYZに変換。さらに白のYが1.0となるように正規化
    # と思ったが、なんかオーバーフローしたので 0.9掛けした
    # --------------------------------
    large_xyz_data = ccv.xyY_to_XYZ(xyY_data)
    large_xyz_data /= np.max(xyY_data)
    large_xyz_data *= np.max(xyY_data) / 100  # 0.9

    # XYZ to RGB 変換を実施
    # --------------------------------
    rgb_large_xyz_matrix = ccv.get_rgb_to_xyz_matrix(gamut=ccv.const_sRGB_xy)
    large_xyz_to_rgb_mtx = linalg.inv(rgb_large_xyz_matrix)
    rgb_data = ccv.color_cvt(large_xyz_data, large_xyz_to_rgb_mtx)

    # 範囲外の値のクリップ および 8bitで正規化
    # --------------------------------
    if (np.sum(rgb_data < 0) > 0) or (np.sum(rgb_data > 1) > 0):
        print("Caution! Overflow has occured.")
        rgb_data[rgb_data < 0] = 0
        rgb_data[rgb_data > 1] = 1
    rgb_data = (rgb_data ** (1/2.2))
    rgb_data = np.uint8(np.round(rgb_data * 0xFF))

    # plot
    # ----------------------------------
    v_num = 4
    h_num = 6
    plt.rcParams["font.size"] = 18
    f, axarr = plt.subplots(v_num, h_num, sharex='col', sharey='row',
                            figsize=(24, 16))
    for idx in range(24):
        color = "#{:02X}{:02X}{:02X}".format(rgb_data[0][idx][0],
                                             rgb_data[0][idx][1],
                                             rgb_data[0][idx][2])
        h_idx = idx % h_num
        v_idx = idx // h_num
        axarr[v_idx, h_idx].add_patch(
            patches.Rectangle(
                (0, 0), 1.0, 1.0, facecolor=color
            )
        )
    plt.show()


def plot_d_illuminant():
    t = np.arange(4000, 10000, 100, dtype=np.float64)
    wl, s = lit.get_d_illuminants_spectrum(t)

    # plot
    # ----------------------------------
    v_num = 6
    h_num = 10
    plt.rcParams["font.size"] = 16
    f, axarr = plt.subplots(v_num, h_num, sharex='col', sharey='row',
                            figsize=(30, 14))
    for idx in range(v_num * h_num):
        h_idx = idx % h_num
        v_idx = idx // h_num
        axarr[v_idx, h_idx].grid()
        if v_idx == (v_num - 1):
            axarr[v_idx, h_idx].set_xlabel("wavelength [nm]")
        if h_idx == 0:
            axarr[v_idx, h_idx].set_ylabel("Relative Power")
        axarr[v_idx, h_idx].set_xlim(300, 830)
        axarr[v_idx, h_idx].set_ylim(0, 170)
        axarr[v_idx, h_idx].set_xticks([360, 560, 760])
        axarr[v_idx, h_idx].set_yticks([0, 50, 100, 150])
        axarr[v_idx, h_idx].plot(wl, s[idx])
    plt.show()


def plot_color_patch_variance():
    """
    # 概要
    高感度ノイズを除去する前準備としてノイズの分布状況を知る
    """
    coord_file = "./data/color_patch_coordinate.csv"
    coord = np.loadtxt(coord_file, dtype=np.uint16, delimiter=',', skiprows=1, )
    print(coord.shape)
    img_file = "./figure/YAMADA_Z10-B.tif"
    img = cv2.imread(img_file,
                     cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

    # color patch の領域をくり抜く
    # ----------------------------
    color_data = []
    for idx in range(coord.shape[0]):
        pt1 = (coord[idx][1], coord[idx][2])
        pt2 = (coord[idx][5], coord[idx][6])
        cv2.rectangle(img, pt1, pt2, (255, 255, 255))
        color_data.append(img[coord[idx][2]:coord[idx][6],
                              coord[idx][1]:coord[idx][5], :])

    # histogram をプロット
    # ----------------------------
    v_num = 4
    h_num = 6
    plt.rcParams["font.size"] = 16
    f, axarr = plt.subplots(v_num, h_num, sharex='col', sharey='row',
                            figsize=(30, 10))
    for idx in range(v_num * h_num):
        h_idx = idx % h_num
        v_idx = idx // h_num
        if v_idx == (v_num - 1):
            axarr[v_idx, h_idx].set_xlabel("Video Level")
        if h_idx == 0:
            axarr[v_idx, h_idx].set_ylabel("Frequency")
        axarr[v_idx, h_idx].set_xticks([20000, 40000, 60000])
        p = color_data[idx]
        p_b, p_g, p_r = np.dsplit(p, 3)
        axarr[v_idx, h_idx].hist(p_r.flatten(), normed=True, bins=100,
                                 color='red', alpha=0.5)
        axarr[v_idx, h_idx].hist(p_g.flatten(), normed=True, bins=100,
                                 color='green', alpha=0.5)
        axarr[v_idx, h_idx].hist(p_b.flatten(), normed=True, bins=100,
                                 color='blue', alpha=0.5)
    plt.show()


def get_color_patch_average():
    """
    # 概要
    カラーパッチの平均の算出。一応外れ値除外もするよ！
    """
    # カラーパッチデータ取得
    coord_file = "./data/color_patch_coordinate.csv"
    coord = np.loadtxt(coord_file, dtype=np.uint16, delimiter=',', skiprows=1, )
    print(coord.shape)
    img_file = "./figure/YAMADA_Z10-B.tif"
    img = cv2.imread(img_file,
                     cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

    # color patch の領域をくり抜く
    # ----------------------------
    color_data = []
    for idx in range(coord.shape[0]):
        pt1 = (coord[idx][1], coord[idx][2])
        pt2 = (coord[idx][5], coord[idx][6])
        cv2.rectangle(img, pt1, pt2, (255, 255, 255))
        color_data.append(img[coord[idx][2]:coord[idx][6],
                              coord[idx][1]:coord[idx][5], :])

    # まずは平均と分散を求める
    # ------------------------
    vari_array = []
    ave_array = []
    h_num = 3
    v_num = 8
    plt.rcParams["font.size"] = 16
    f, axarr = plt.subplots(v_num, h_num, sharex='col', sharey='row',
                            figsize=(20, 20))
    for idx in range(v_num * h_num):
        h_idx = idx % h_num
        v_idx = idx // h_num
        if v_idx == (v_num - 1):
            axarr[v_idx, h_idx].set_xlabel("Video Level")
        if h_idx == 0:
            axarr[v_idx, h_idx].set_ylabel("Frequency")
        axarr[v_idx, h_idx].set_xticks([20000, 40000, 60000])
        p = color_data[idx]
        p_b, p_g, p_r = np.dsplit(p, 3)
        axarr[v_idx, h_idx].hist(p_r.flatten(), normed=False, bins=100,
                                 color='red', alpha=0.5)
        axarr[v_idx, h_idx].hist(p_g.flatten(), normed=False, bins=100,
                                 color='green', alpha=0.5)
        axarr[v_idx, h_idx].hist(p_b.flatten(), normed=False, bins=100,
                                 color='blue', alpha=0.5)
        rgb = [x for x in np.dsplit(color_data[idx], 3)]
        b_v, g_v, r_v = [np.sqrt(np.var(x)) for x in rgb]
        b_a, g_a, r_a = [np.average(x) for x in rgb]
        axarr[v_idx, h_idx].plot([r_a-r_v, r_a-r_v], [0.0, 60000], 'r-')
        axarr[v_idx, h_idx].plot([r_a+r_v, r_a+r_v], [0.0, 60000], 'r-')
        axarr[v_idx, h_idx].plot([g_a-g_v, g_a-g_v], [0.0, 60000], 'g-')
        axarr[v_idx, h_idx].plot([g_a+g_v, g_a+g_v], [0.0, 60000], 'g-')
        axarr[v_idx, h_idx].plot([b_a-b_v, b_a-b_v], [0.0, 60000], 'b-')
        axarr[v_idx, h_idx].plot([b_a+b_v, b_a+b_v], [0.0, 60000], 'b-')

    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # show_color_patch_spectral_data()
    # make_color_patch_image()
    # plot_d_illuminant()
    # plot_color_patch_variance()
    get_color_patch_average()
    # x = np.arange(9).reshape(3, 3, 1)
    # print(np.var(x))


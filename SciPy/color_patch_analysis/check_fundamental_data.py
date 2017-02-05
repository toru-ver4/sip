import os
import imp
import numpy as np
from scipy import linalg
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import color_convert as ccv
import light as lit

imp.reload(ccv)
imp.reload(lit)

const_lambda = np.arange(380, 785, 5)


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


def get_color_patch_average(plot=False):
    """
    # 概要
    カラーパッチの平均の算出。一応外れ値除外もするよ！
    """
    # カラーパッチデータ取得
    coord_file = "./data/color_patch_coordinate.csv"
    coord = np.loadtxt(coord_file, dtype=np.uint16, delimiter=',', skiprows=1, )
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
    ave_array = []
    h_num = 6
    v_num = 4
    for idx in range(v_num * h_num):
        bgr = [x for x in np.dsplit(color_data[idx], 3)]
        b, g, r = np.dsplit(color_data[idx], 3)
        b_v, g_v, r_v = [np.sqrt(np.var(x)) for x in bgr]
        b_a, g_a, r_a = [np.average(x) for x in bgr]
        mask_r = np.logical_and(r > (r_a - r_v), r < (r_a + r_v))
        mask_g = np.logical_and(g > (g_a - g_v), g < (g_a + g_v))
        mask_b = np.logical_and(b > (b_a - b_v), b < (b_a + b_v))
        new_r = r[mask_r]
        new_g = g[mask_g]
        new_b = b[mask_b]
        new_rgb = [np.average(new_r), np.average(new_g), np.average(new_b)]
        ave_array.append(new_rgb)

    if plot:
        ave_array = np.array(ave_array)
        ave_array = np.uint8(np.round((ave_array / 0xFFFF) * 0xFF))
        plt.rcParams["font.size"] = 18
        f, axarr = plt.subplots(v_num, h_num, sharex='col', sharey='row',
                                figsize=(24, 16))
        for idx in range(h_num * v_num):
            color = "#{:02X}{:02X}{:02X}".format(ave_array[idx][0],
                                                 ave_array[idx][1],
                                                 ave_array[idx][2])
            h_idx = idx % h_num
            v_idx = idx // h_num
            axarr[v_idx, h_idx].add_patch(
                patches.Rectangle(
                    (0, 0), 1.0, 1.0, facecolor=color
                )
            )
        plt.show()

    return ave_array


def get_normal_distribution(mu, sigma, x=const_lambda, plot=False):
    """
    # 概要
    正規分布を出力する関数を作る

    # 注意事項
    横軸は波長。範囲は 380..780nm の 5nm 刻みな。
    """
    exp_naka = -((x - mu) ** 2) / (2 * (sigma ** 2))
    y = (1/np.sqrt(2 * np.pi * (sigma ** 2))) * np.exp(exp_naka)

    if plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y)
        plt.show()

    return x, y


def plot_normal_distribution(mu_list, sigma_list):
    """
    # 概要
    シミュレーションをぶん回す正規分布一覧をプロット

    # 注意事項
    横軸は波長。範囲は 380..780nm の 5nm 刻みな。
    """

    h_num = len(sigma_list)
    v_num = len(mu_list)
    plt.rcParams["font.size"] = 16
    f, axarr = plt.subplots(v_num, h_num, sharex='col', sharey='row',
                            figsize=(20, 20))
    for m_idx, mu in enumerate(mu_list):
        for s_idx, sigma in enumerate(sigma_list):
            idx = m_idx * len(sigma_list) + s_idx
            x, y = get_normal_distribution(mu, sigma)
            h_idx = idx % h_num
            v_idx = idx // h_num
            axarr[v_idx, h_idx].grid()
            if v_idx == (v_num - 1):
                axarr[v_idx, h_idx].set_xlabel("wavelength [nm]")
            if h_idx == 0:
                axarr[v_idx, h_idx].set_ylabel("sensitivity")
            axarr[v_idx, h_idx].set_xlim(380, 780)
            axarr[v_idx, h_idx].set_ylim(0, 0.15)
            axarr[v_idx, h_idx].set_xticks([400, 500, 600, 700])
            axarr[v_idx, h_idx].set_yticks([0, 0.03, 0.06, 0.09, 0.12, 0.15])
            axarr[v_idx, h_idx].plot(x, y)

    plt.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # show_color_patch_spectral_data()
    # make_color_patch_image()
    # plot_d_illuminant()
    # plot_color_patch_variance()
    # get_color_patch_average(plot=True)
    # x = np.arange(9).reshape(3, 3, 1)
    # print(np.var(x))
    # get_normal_distribution(a=1.0, mu=590, sigma=30)
    mu_list = [500, 525, 550, 575, 600]
    sigma_list = [3, 5, 7, 10, 15, 20]
    plot_normal_distribution(mu_list, sigma_list)

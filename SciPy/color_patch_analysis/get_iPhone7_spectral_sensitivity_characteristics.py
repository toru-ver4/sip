import os
import imp
import numpy as np
from scipy import linalg
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import color_convert as ccv
import light as lit
import check_fundamental_data as cfd
import plot_utility as pu

imp.reload(ccv)
imp.reload(lit)
imp.reload(cfd)
imp.reload(pu)


def color_patch_rgb_to_large_xyz(rgb_val):
    """
    # 概要
    RGB の ColorPatch値 を XYZ値に変換する

    # 注意事項
    rgb_val は numpy(dtype=np.uint16, shape=(N, 3)) とする。
    """
    rgb_val = rgb_val.reshape((1, rgb_val.shape[0], rgb_val.shape[1]))
    # dtype から型の最大値を求めて正規化する
    # ------------------------------------
    try:
        img_max_value = np.iinfo(rgb_val.dtype).max
    except:
        img_max_value = 1.0
    rgb_val = np.float32(rgb_val/img_max_value)

    # rgb2XYZ変換
    # --------------------------------------
    rgb_linear = ccv.srgb_to_linear(rgb_val)
    mtx = ccv.get_rgb_to_xyz_matrix(gamut=ccv.const_sRGB_xy)
    large_xyz_val = (ccv.color_cvt(rgb_linear, mtx) * 100)

    return large_xyz_val


def d_illuminant_interpolation(plot=False):
    """
    # 概要
    D Illuminant の分光特性は 10nm 刻みなので、
    それを 5nm 刻みに線形補完する
    """
    # D Illuminant 算出 (10nm精度)
    # -----------------------------
    t = np.array([5000], dtype=np.float64)
    wl, s = lit.get_d_illuminants_spectrum(t)

    # Interpolation
    # -----------------------------
    # f_quadratic = interpolate.interp1d(wl, s, kind='quadratic')
    f_cubic = interpolate.interp1d(wl, s, kind='cubic')
    wl_new = np.arange(380, 785, 5, dtype=np.uint16)
    # s_quadratic = f_quadratic(wl_new)
    s_cubic = f_cubic(wl_new)

    if plot:
        ax1 = pu.plot_1_graph(fontsize=20,
                              figsize=(10, 8),
                              graph_title="cubic interpolation",
                              graph_title_size=None,
                              xlabel="Wavelength [nm]", ylabel="Intensity",
                              axis_label_size=None,
                              legend_size=17,
                              xlim=None,
                              ylim=None,
                              xtick=None,
                              ytick=None,
                              xtick_size=None, ytick_size=None,
                              linewidth=None)
        ax1.plot(wl, s, 'ro', linewidth=5, label="original")
        # ax1.plot(wl_new, s_quadratic, 'g-x', linewidth=2, label="quadratic")
        ax1.plot(wl_new, s_cubic, 'b-', linewidth=2, label="cubic")
        plt.legend(loc='lower right')
        plt.show()

    return wl_new, s_cubic


def intergrate_large_xyz():
    """
    # 概要
    XYZ を求めるための積分をがんばる
    """

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # rgb_val = np.uint16(np.round(cfd.get_color_patch_average()))
    # color_patch_rgb_to_large_xyz(rgb_val)
    # plt.rcParams['axes.grid'] = True
    d_illuminant_interpolation(plot=True)

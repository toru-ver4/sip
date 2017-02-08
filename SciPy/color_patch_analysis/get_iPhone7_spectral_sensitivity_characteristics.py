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
import check_fundamental_data as cfd

imp.reload(ccv)
imp.reload(lit)
imp.reload(cfd)


def color_patch_rgb_to_large_xyz(rgb_val):
    """
    # 概要
    RGB の ColorPatch値 を XYZ値に変換する

    # 注意事項
    rgb_val は numpy(dtype=np.uint16, shape=(N, 3)) とする。
    """
    rgb_val = rgb_val.reshape((1, rgb_val.shape[0], rgb_val.shape[1]))
    mtx = ccv.get_rgb_to_xyz_matrix(gamut=ccv.const_sRGB_xy)
    large_xyz_val = np.uint16(np.round(ccv.color_cvt(rgb_val, mtx)))
    print(large_xyz_val)
    ccv.srgb_to_linear(large_xyz_val)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    rgb_val = np.uint16(np.round(cfd.get_color_patch_average()))
    color_patch_rgb_to_large_xyz(rgb_val)

import os
import sys
import cv2
import numpy as np
from scipy import linalg
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


def get_white_point_conv_matrix(src=const_d65_xy, dst=const_d50_xy):
    """
    参考： http://w3.kcua.ac.jp/~fujiwara/infosci/colorspace/bradford.html
    """
    if len(src) == 2:
        src = [src[0], src[1], 1 - (src[0] + src[1])]
    if len(dst) == 2:
        dst = [dst[0], dst[1], 1 - (dst[0] + dst[1])]

    src = np.array(src)
    dst = np.array(dst)

    src = src / src[1]
    dst = dst / dst[1]

    # LMS値を求めよう
    # --------------------------------------
    ma = np.array(const_xyz_to_lms)
    ma_inv = linalg.inv(ma)

    src_LMS = ma.dot(src)
    dst_LMS = ma.dot(dst)

    # print(src, dst)
    # print(src_LMS, dst_LMS)

    # M行列を求めよう
    # --------------------------------------
    mtx = [[dst_LMS[0]/src_LMS[0], 0.0, 0.0],
           [0.0, dst_LMS[1]/src_LMS[1], 0.0],
           [0.0, 0.0, dst_LMS[2]/src_LMS[2]]]

    m_mtx = ma_inv.dot(mtx).dot(ma)

    # print(m_mtx)

    return m_mtx


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


def inv_test():
    mtx = [[0.2126, 0.7152, 0.0722],
           [-0.114572, -0.385428, 0.5],
           [0.5, -0.454153, -0.045847]]
    mtx = np.array(mtx)
    inv_mtx = linalg.inv(mtx)
    print(mtx)
    print(inv_mtx)


def change_img_white_point(filename='wp04_1920x1080.jpg'):
    """
    # 概要
    色温度変換を行う。

    # 注意事項
    現状だと D65 --> D50 しか変換できない。
    気が向いたら任意の色温度に変換できるように拡張しよう。
    """
    img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    m_mtx = get_white_point_conv_matrix()

    # XYZ 変換 Matrix と組み合わせて RGB 空間用の mtx を求める
    # ----------------------------------------------------
    rgb2xyz_mtx = get_rgb_to_xyz_matrix(gamut=const_sRGB_xy)
    xyz2rgb_mtx = linalg.inv(rgb2xyz_mtx)
    mtx = xyz2rgb_mtx.dot(m_mtx).dot(rgb2xyz_mtx)

    # 求めた mtx を使って画像を変換
    # ----------------------------------------------------
    img_out = color_cvt(img[:, :, ::-1], mtx)[:, :, ::-1]
    img_out = np.round(img_out).astype(img.dtype)

    img_max = np.iinfo(img.dtype).max
    cv2.imshow('bbb.tif', img_out/img_max)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ファイルに保存
    # ----------------------------------------------------
    root, ext = os.path.splitext(filename)
    out_filename = root + "_modify" + ext
    cv2.imwrite(out_filename, img_out)


def color_cvt(img, mtx):
    """
    # 概要
    img に対して mtx を適用する。
    # 注意事項
    例によって、RGBの並びを考えている。BGRの並びの場合は
    img[:, :, ::-1] してから関数をコールすること。
    """
    img_max = np.iinfo(img.dtype).max
    img_min = np.iinfo(img.dtype).min

    r, g, b = np.dsplit(img, 3)
    ro = r * mtx[0][0] + g * mtx[0][1] + b * mtx[0][2]
    go = r * mtx[1][0] + g * mtx[1][1] + b * mtx[1][2]
    bo = r * mtx[2][0] + g * mtx[2][1] + b * mtx[2][2]

    out_img = np.dstack((ro, go, bo))

    out_img[out_img < img_min] = img_min
    out_img[out_img > img_max] = img_max

    return out_img


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


if __name__ == '__main__':
    # get_rgb_to_xyz_matrix(gamut=const_sRGB_xy)
    # get_white_point_conv_matrix()
    # change_img_white_point()
    print(get_yuv_trans_coef(gamut=const_rec601_xy))
    print(get_yuv_trans_coef(gamut=const_rec709_xy))
    print(get_yuv_trans_coef(gamut=const_rec2020_xy))

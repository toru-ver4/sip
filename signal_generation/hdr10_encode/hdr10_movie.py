#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
動画を作ったりするよ
"""

import os
import color_convert as cc


def get_hdr10_info_for_x265(gamut='dci-p3'):
    """
    # 概要
    x265 の --master-display で指定する文字列は計算で出す必要がある。
    それを実行する。
    # 引数
    dci-p3 : 'rec709', 'rec2020', 'dci-p3', 'dci-d65' のいずれかを指定。
    """
    div_val_gamut = 0.00002
    div_val_luminance = 0.0001
    l_min = 0.001
    l_max = 1000

    if gamut == 'rec709':
        gamut = cc.const_rec709_xy
        white = cc.const_d65_xy
    elif gamut == 'rec2020':
        gamut = cc.const_rec2020_xy
        white = cc.const_d65_xy
    elif gamut == 'dci-p3':
        gamut = cc.const_dci_p3_xy
        white = cc.const_dci_white_xy
    elif gamut == 'dci-d65':
        gamut = cc.const_dci_p3_xy
        white = cc.const_d65_xy
    else:
        print("please specify gamut")

    r_str = "R({:.0f},{:.0f})".format(gamut[0][0] / div_val_gamut,
                                      gamut[0][1] / div_val_gamut)
    g_str = "G({:.0f},{:.0f})".format(gamut[1][0] / div_val_gamut,
                                      gamut[1][1] / div_val_gamut)
    b_str = "B({:.0f},{:.0f})".format(gamut[2][0] / div_val_gamut,
                                      gamut[2][1] / div_val_gamut)
    w_str = "WP({:.0f},{:.0f})".format(white[0] / div_val_gamut,
                                       white[1] / div_val_gamut)
    l_str = "L({:.0f},{:.0f})".format(l_max / div_val_luminance,
                                      l_min / div_val_luminance)

    return g_str+b_str+r_str+w_str+l_str


if __name__ == '__main__':
    print(get_hdr10_info_for_x265(gamut='rec709'))
    print(get_hdr10_info_for_x265(gamut='rec2020'))
    print(get_hdr10_info_for_x265(gamut='dci-p3'))
    print(get_hdr10_info_for_x265(gamut='dci-d65'))

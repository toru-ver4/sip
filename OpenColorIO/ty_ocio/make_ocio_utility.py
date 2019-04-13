#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
OpenColorIO の Config 作成で共通して使うものをまとめておく。
"""

import os

OCIO_CONFIG_NAME = "ty_config.ocio"

# copied from transfer_funcsions.py
GAMMA24 = 'Gamma 2.4'
ST2084 = 'SMPTE ST2084'
HLG = 'BT.2100 HLG'
LOGC = 'ARRI LOG_C'
VLOG_IRE = 'Panasonic VLog (IRE Base)'
VLOG = 'Panasonic VLog'
SLOG3 = "SONY S-Log3 (IRE Base)"
SLOG3_REF = "SONY S-Log3"
REDLOG = "RED REDLog"
LOG3G10 = "RED Log3G10"
LOG3G12 = "RED Log3G12"
NLOG = "Nikon N-Log"
DLOG = "DJI D-Log"
FLOG = "FUJIFILM F-Log"
LINEAR = "Linear"
SRGB = "sRGB"

# copied from color_space.py
BT709 = 'ITU-R BT.709'
BT2020 = 'ITU-R BT.2020'
ACES_AP0 = 'ACES2065-1'
ACES_AP1 = 'ACEScg'
S_GAMUT3 = 'S-Gamut3'
S_GAMUT3_CINE = 'S-Gamut3.Cine'
ALEXA_WIDE_GAMUT = 'ALEXA Wide Gamut'
V_GAMUT = 'V-Gamut'
CINEMA_GAMUT = 'Cinema Gamut'
RED_WIDE_GAMUT_RGB = 'REDWideGamutRGB'
DCI_P3 = 'DCI-P3'
SRGB = "sRGB"

ACES2065_CS = [ACES_AP0, LINEAR]
ACESCG_CS = [ACES_AP1, LINEAR]
SRGB_CS = [SRGB, SRGB]
BT709_CS = [BT709, GAMMA24]
BT1886_CS = [BT709, GAMMA24]
BT2020_CS = [BT2020, GAMMA24]
P3_ST2084_CS = [DCI_P3, ST2084]
BT2020_ST2084_CS = [BT2020, ST2084]
ALEXA_LOGC_CS = [ALEXA_WIDE_GAMUT, LOGC]
BT2020_LOGC_CS = [BT2020, LOGC]

REFERENCE_ROLE = ACES2065_CS

DUMMY_MATRIX = [1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0]


def get_colorspace_name(gamut_eotf_pair):
    temp = "gamut_{} - eotf_{}".format(gamut_eotf_pair[0], gamut_eotf_pair[1])
    return temp.replace('ITU-R ', "")


def get_display_name(gamut_eotf_pair):
    temp = "gamut_{} - eotf_{}".format(gamut_eotf_pair[0], gamut_eotf_pair[1])
    return temp.replace('ITU-R ', "")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

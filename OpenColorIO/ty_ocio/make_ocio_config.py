#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""

"""

import os
import PyOpenColorIO as OCIO

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

ROLE_ACES2065 = [ACES_AP0, LINEAR]
ROLE_ACESCG = [ACES_AP1, LINEAR]
ROLE_SRGB = [SRGB, SRGB]
ROLE_BT709 = [BT709, GAMMA24]
ROLE_P3_ST2084 = [DCI_P3, ST2084]
ROLE_BT2020_ST2084 = [BT2020, ST2084]

REFERENCE_ROLE = ROLE_ACES2065


class OcioConfigControl:
    """
    色々と頑張ってOCIOのコンフィグを作る。
    """
    def __init__(self):
        self.config_name = OCIO_CONFIG_NAME

    def get_role_name(self, gamut_eotf_pair):
        return "{}_{}".format(gamut_eotf_pair[0], gamut_eotf_pair[1])

    def set_role(self):
        self.config.setRole(OCIO.Constants.ROLE_REFERENCE, self.get_role_name(REFERENCE_ROLE))
        # self.config.setRole(OCIO.Constants.ROLE_COLOR_TIMING, "Cineon")
        # self.config.setRole(OCIO.Constants.ROLE_COMPOSITING_LOG, "Cineon")
        # self.config.setRole(OCIO.Constants.ROLE_DATA, "ACEScg")
        # self.config.setRole(OCIO.Constants.ROLE_DEFAULT, "raw")
        # self.config.setRole(OCIO.Constants.ROLE_COLOR_PICKING, "sRGB")
        # self.config.setRole(OCIO.Constants.ROLE_MATTE_PAINT, "sRGB")
        # self.config.setRole(OCIO.Constants.ROLE_TEXTURE_PAINT, "sRGB")

    def set_color_space(self):
        # self.set_simple_color_space('bt.709')  # 通常は同じメソッドで引数で内容切り替え
        # self.set_simple_color_space('bt.2020')
        # self.set_simple_color_space('st2084')
        # self.set_hoge_color_space()  # 面倒なのは専用メソッド用意
        cs = OCIO.ColorSpace(name=self.get_role_name(REFERENCE_ROLE))
        cs.setDescription("")
        cs.setBitDepth(OCIO.Constants.BIT_DEPTH_F32)
        cs.setAllocation(OCIO.Constants.ALLOCATION_LG2)
        cs.setAllocationVars([-8, 5, 0.00390625])
        self.config.addColorSpace(cs)

    def set_display(self):
        display = 'default'
        self.config.addDisplay(display, 'None', 'raw')
        self.config.addDisplay(display, 'sRGB', 'sRGB')
        self.config.addDisplay(display, 'rec709', 'rec709')
        self.config.setActiveDisplays('default')
        self.config.setActiveViews('sRGB')


    def flush_config(self):
        try:
            self.config.sanityCheck()
        except Exception, e:
            print e

        f = file(self.config_name, "w")
        f.write(self.config.serialize())
        f.close()
        print("wrote", self.config_name)

    def make_config(self):
        self.config = OCIO.Config()
        self.config.setSearchPath('luts')
        self.set_role()
        self.set_color_space()
        self.set_display()
        self.flush_config()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(dir(OCIO.ColorSpace))
    ocio_config = OcioConfigControl()
    ocio_config.make_config()

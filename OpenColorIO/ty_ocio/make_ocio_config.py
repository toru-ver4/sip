#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
内部空間は Display Reffered とする。
rec.709 とか End-to-End Gamma とか考えるな！BT.1886で行くんだよ！！

作戦としては、最初に Python3 コードに一式のLUTとMatrixを作らせる。
で、その情報を使って Python2 コードで config を作る流れになるはず。

2つのコードでのやり取りは json になるかな？
"""

import os
import PyOpenColorIO as OCIO
from make_ocio_utility import get_colorspace_name, get_colorspace_name
from make_ocio_utility import REFERENCE_ROLE, BT1886_CS, ALEXA_LOGC_CS
from make_ocio_utility import OCIO_CONFIG_NAME
import make_ocio_color_space as mocs


class OcioConfigControl:
    """
    色々と頑張ってOCIOのコンフィグを作る。
    """
    def __init__(self):
        self.config_name = OCIO_CONFIG_NAME

    def get_colorspace_name(self, gamut_eotf_pair):
        temp = "gamut_{} - eotf_{}".format(gamut_eotf_pair[0],
                                           gamut_eotf_pair[1])
        return temp.replace('ITU-R ', "")

    def get_colorspace_name(self, gamut_eotf_pair):
        temp = "gamut_{} - eotf_{}".format(gamut_eotf_pair[0],
                                           gamut_eotf_pair[1])
        return temp.replace('ITU-R ', "")

    def set_role(self):
        self.config.setRole(OCIO.Constants.ROLE_COLOR_TIMING, get_colorspace_name(ALEXA_LOGC_CS))
        self.config.setRole(OCIO.Constants.ROLE_COMPOSITING_LOG, get_colorspace_name(ALEXA_LOGC_CS))
        self.config.setRole(OCIO.Constants.ROLE_DATA, 'raw')
        self.config.setRole(OCIO.Constants.ROLE_DEFAULT, 'raw')
        self.config.setRole(OCIO.Constants.ROLE_COLOR_PICKING, get_colorspace_name(BT1886_CS))
        self.config.setRole(OCIO.Constants.ROLE_MATTE_PAINT, 'raw')
        self.config.setRole(OCIO.Constants.ROLE_REFERENCE, 'raw')
        self.config.setRole(OCIO.Constants.ROLE_SCENE_LINEAR, 'raw')
        self.config.setRole(OCIO.Constants.ROLE_TEXTURE_PAINT, 'raw')

    def set_color_space(self):
        self.config.addColorSpace(mocs.make_ref_color_space())
        self.config.addColorSpace(mocs.make_raw_color_space())
        self.config.addColorSpace(mocs.make_bt1886_color_space())
        self.config.addColorSpace(mocs.make_p3_st2084_color_space())
        self.config.addColorSpace(mocs.make_arri_logc_color_space())

    def set_display(self):
        display = 'default'
        # self.config.addDisplay(display, 'raw', 'raw')
        # self.config.addDisplay(display, 'sRGB', 'sRGB')
        self.config.addDisplay(display, get_colorspace_name(BT1886_CS),
                               get_colorspace_name(BT1886_CS))
        self.config.setActiveDisplays(display)
        self.config.setActiveViews(get_colorspace_name(BT1886_CS))

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
    # print(dir(OCIO.FileTransform))
    # print(dir(OCIO.GroupTransform))
    print(dir(OCIO.Constants))
    ocio_config = OcioConfigControl()
    ocio_config.make_config()

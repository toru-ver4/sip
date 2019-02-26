#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""

"""

import os
import PyOpenColorIO as OCIO

OCIO_CONFIG_NAME = "ty_config.ocio"


class OcioConfigControl:
    """
    色々と頑張ってOCIOのコンフィグを作る。
    """
    def __init__(self):
        self.config_name = OCIO_CONFIG_NAME

    def make_config(self):
        self.config = OCIO.Config()
        self.config.setSearchPath('luts')
        self.flush_config()

    def set_role(self):
        config.setRole(OCIO.Constants.ROLE_SCENE_LINEAR, "linear")
        config.setRole(OCIO.Constants.ROLE_REFERENCE, "linear")
        config.setRole(OCIO.Constants.ROLE_COLOR_TIMING, "Cineon")
        config.setRole(OCIO.Constants.ROLE_COMPOSITING_LOG, "Cineon")
        config.setRole(OCIO.Constants.ROLE_DATA, "raw")
        config.setRole(OCIO.Constants.ROLE_DEFAULT, "raw")
        config.setRole(OCIO.Constants.ROLE_COLOR_PICKING, "sRGB")
        config.setRole(OCIO.Constants.ROLE_MATTE_PAINT, "sRGB")
        config.setRole(OCIO.Constants.ROLE_TEXTURE_PAINT, "sRGB")


    def flush_config(self):
        try:
            self.config.sanityCheck()
        except Exception, e:
            print e

        f = file(self.config_name, "w")
        f.write(self.config.serialize())
        f.close()
        print("wrote", self.config_name)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ocio_config = OcioConfigControl()
    ocio_config.make_config()

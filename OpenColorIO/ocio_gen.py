#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2018 - Toru Yoshihara'
__license__ = ''
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 gmail.com'
__status__ = 'Production'

__major_version__ = '0'
__minor_version__ = '0'
__change_version__ = '1'
__version__ = '.'.join((__major_version__,
                        __minor_version__,
                        __change_version__))

import os
import sys

ACES_PY_FILES_DIR = "./aces_1.0.3/python"
ACES_CTL_DIR = "./transforms/ctl"
OCIO_CONFIG_DIR = "./ocio_config"

# ACES_CTL_DIR = "../../transforms/ctl"
# OCIO_CONFIG_DIR = "../../ocio_config"

current_dir = os.path.dirname(os.path.abspath(__file__))
aces_py_dir = os.path.join(current_dir, ACES_PY_FILES_DIR)
sys.path.append(aces_py_dir)
# sys.path.append("./")


def ocio_gen_main():
    from aces_ocio.generate_config import generate_config
    generate_config(aces_ctl_directory=ACES_CTL_DIR,
                    config_directory=OCIO_CONFIG_DIR,
                    lut_resolution_1d=4096,
                    lut_resolution_3d=65,
                    bake_secondary_luts=True)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ocio_gen_main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
config.ocio 向けの LUT を作る。
"""

import os
import transfer_functions as tf
import lut as tylut
import numpy as np


def gen_eotf_lut_for_ocio(eotf_name=tf.ST2084, sample_num=4096,
                          out_dir_name='./luts'):
    """
    ocio向けの EOTF 1DLUT を作る。
    Matrix とかは別途用意してね！

    Parameters
    ----------
    eotf_name : strings
        A name of the eotf.
        select from **transfer_functions** module.
    sample_num : int
        sample number.

    Returns
    -------
        None.
    """
    x = np.linspace(0, 1, sample_num)
    y = tf.eotf_to_luminance(x, eotf_name) / tf.REF_WHITE_LUMINANCE
    fname_base = "{}/{}_to_Linear.spi1d"
    fname = fname_base.format(out_dir_name, eotf_name.replace(" ", "_"))
    tylut.save_1dlut_spi_format(lut=y, filename=fname, min=0.0, max=1.0)
    print("{} was generated.".format(fname))


def make_all_1dluts():
    gen_eotf_lut_for_ocio(tf.ST2084, 4096, "./luts")
    gen_eotf_lut_for_ocio(tf.GAMMA24, 4096, "./luts")
    gen_eotf_lut_for_ocio(tf.LOGC, 4096, "./luts")
    gen_eotf_lut_for_ocio(tf.VLOG, 4096, "./luts")
    gen_eotf_lut_for_ocio(tf.SLOG3, 4096, "./luts")
    gen_eotf_lut_for_ocio(tf.LOG3G10, 4096, "./luts")
    gen_eotf_lut_for_ocio(tf.LOG3G12, 4096, "./luts")
    gen_eotf_lut_for_ocio(tf.NLOG, 4096, "./luts")
    gen_eotf_lut_for_ocio(tf.DLOG, 4096, "./luts")
    gen_eotf_lut_for_ocio(tf.FLOG, 4096, "./luts")
    gen_eotf_lut_for_ocio(tf.SRGB, 4096, "./luts")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_all_1dluts()

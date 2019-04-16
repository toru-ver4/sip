#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
config.ocio 向けの LUT を作る。
"""

import os
import transfer_functions as tf
import lut as tylut
import numpy as np
import color_space as cs


def gen_sp_eotf_lut_default(sample_num=8, out_dir_name='./luts',
                            min=0, max=1.0):
    """
    OCIO特殊検証用の EOTF 1DLUT を作る。
    """
    x = np.linspace(0, 1, sample_num)
    y = ((x ** 2.4) * (max - min) + min)
    fname_base = "{}/experiment_min_{}_max_{}.spi1d"
    fname = fname_base.format(out_dir_name, min, max)
    fname = fname.replace("-", 'minus')
    tylut.save_1dlut_spi_format(lut=y, filename=fname, min=min, max=max)
    print("{} was generated.".format(fname))


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
    gen_sp_eotf_lut_default(sample_num=8, min=0, max=1)
    gen_sp_eotf_lut_default(sample_num=8, min=-1, max=2)
    gen_sp_eotf_lut_default(sample_num=8, min=-1, max=0.5)


def print_matrix_name(src=cs.ACES_AP0, dst=cs.BT709):
    name = "{}_TO_{}_MTX".format(src, dst)
    temp = name.replace('ITU-R ', "")
    temp = temp.replace("-", "_")
    temp = temp.replace(" ", "_")
    temp = temp.replace('.', "")
    print(temp)


def make_and_print_each_matrix(src, dst):
    print_matrix_name(src, dst)
    print(cs.ocio_matrix_transform_mtx(src, dst))
    print_matrix_name(dst, src)
    print(cs.ocio_matrix_transform_mtx(dst, src))
    print("")


def make_all_matrixes():
    make_and_print_each_matrix(cs.ACES_AP0, cs.SRTB)
    make_and_print_each_matrix(cs.ACES_AP0, cs.BT709)
    make_and_print_each_matrix(cs.ACES_AP0, cs.BT2020)
    make_and_print_each_matrix(cs.ACES_AP0, cs.DCI_P3)
    make_and_print_each_matrix(cs.ACES_AP0, cs.S_GAMUT3)
    make_and_print_each_matrix(cs.ACES_AP0, cs.S_GAMUT3_CINE)
    make_and_print_each_matrix(cs.ACES_AP0, cs.ALEXA_WIDE_GAMUT)
    make_and_print_each_matrix(cs.ACES_AP0, cs.ACES_AP1)
    make_and_print_each_matrix(cs.ACES_AP0, cs.V_GAMUT)
    make_and_print_each_matrix(cs.ACES_AP0, cs.RED_WIDE_GAMUT_RGB)


def make_exp_3dlut():
    pass


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_all_1dluts()
    make_all_matrixes()

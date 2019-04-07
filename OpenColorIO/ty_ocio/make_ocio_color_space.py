#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
OCIO用のColorSpaceを作る。
"""

import os
import PyOpenColorIO as OCIO
from make_ocio_utility import get_colorspace_name
from make_ocio_utility import BT1886_CS, REFERENCE_ROLE, P3_ST2084_CS
from make_ocio_utility import AP0_TO_BT709_MTX, BT709_TO_AP0_MTX,\
    P3_TO_AP0_MTX, AP0_TO_P3_MTX


DIRECTION_OPS = {
    'forward': OCIO.Constants.TRANSFORM_DIR_FORWARD,
    'inverse': OCIO.Constants.TRANSFORM_DIR_INVERSE}
INTERPOLATION_OPS = {
    'linear': OCIO.Constants.INTERP_LINEAR,
    'tetrahedral': OCIO.Constants.INTERP_TETRAHEDRAL}
COLOR_SPACE_DIRECTION = {
    'to_reference': OCIO.Constants.COLORSPACE_DIR_TO_REFERENCE,
    'from_reference': OCIO.Constants.COLORSPACE_DIR_FROM_REFERENCE}


def make_ref_color_space():
    cs = OCIO.ColorSpace(name=get_colorspace_name(REFERENCE_ROLE))
    cs.setDescription("")
    cs.setBitDepth(OCIO.Constants.BIT_DEPTH_F32)
    cs.setAllocation(OCIO.Constants.ALLOCATION_LG2)
    cs.setAllocationVars([-8, 5, 0.00390625])

    return cs


def make_raw_color_space():
    cs = OCIO.ColorSpace(name='raw')
    cs.setDescription("")
    cs.setBitDepth(OCIO.Constants.BIT_DEPTH_F32)
    cs.setAllocation(OCIO.Constants.ALLOCATION_UNIFORM)
    cs.setAllocationVars([0, 1])

    return cs


def make_bt1886_color_space():
    cs = OCIO.ColorSpace(name=get_colorspace_name(BT1886_CS))
    cs.setDescription("")
    cs.setBitDepth(OCIO.Constants.BIT_DEPTH_F32)
    cs.setAllocation(OCIO.Constants.ALLOCATION_UNIFORM)
    cs.setAllocationVars([0, 1])

    # to reference
    file_to_ref = OCIO.FileTransform('bt1886.spi1d',
                                     direction=DIRECTION_OPS['forward'],
                                     interpolation=INTERPOLATION_OPS['linear'])
    matrix_to_ref = OCIO.MatrixTransform(matrix=BT709_TO_AP0_MTX,
                                         direction=DIRECTION_OPS['forward'])
    group_to_ref = OCIO.GroupTransform([file_to_ref, matrix_to_ref])
    cs.setTransform(group_to_ref, COLOR_SPACE_DIRECTION['to_reference'])

    # from reference
    file_from_ref = OCIO.FileTransform('bt1886.spi1d',
                                       direction=DIRECTION_OPS['inverse'],
                                       interpolation=INTERPOLATION_OPS['linear'])
    matrix_from_ref = OCIO.MatrixTransform(matrix=AP0_TO_BT709_MTX,
                                           direction=DIRECTION_OPS['forward'])
    group_from_ref = OCIO.GroupTransform([matrix_from_ref, file_from_ref])
    cs.setTransform(group_from_ref, COLOR_SPACE_DIRECTION['from_reference'])

    return cs


def make_p3_st2084_color_space():
    cs = OCIO.ColorSpace(name=get_colorspace_name(P3_ST2084_CS))
    cs.setDescription("")
    cs.setBitDepth(OCIO.Constants.BIT_DEPTH_F32)
    cs.setAllocation(OCIO.Constants.ALLOCATION_UNIFORM)
    cs.setAllocationVars([0, 1])

    # to reference
    file_to_ref\
        = OCIO.FileTransform('st2084.spi1d',
                             direction=DIRECTION_OPS['forward'],
                             interpolation=INTERPOLATION_OPS['linear'])
    matrix_to_ref = OCIO.MatrixTransform(matrix=P3_TO_AP0_MTX,
                                         direction=DIRECTION_OPS['forward'])
    group_to_ref = OCIO.GroupTransform([file_to_ref, matrix_to_ref])
    cs.setTransform(group_to_ref, COLOR_SPACE_DIRECTION['to_reference'])

    # from reference
    file_from_ref\
        = OCIO.FileTransform('st2084.spi1d',
                             direction=DIRECTION_OPS['inverse'],
                             interpolation=INTERPOLATION_OPS['linear'])
    matrix_from_ref = OCIO.MatrixTransform(matrix=AP0_TO_P3_MTX,
                                           direction=DIRECTION_OPS['forward'])
    group_from_ref = OCIO.GroupTransform([matrix_from_ref, file_from_ref])
    cs.setTransform(group_from_ref, COLOR_SPACE_DIRECTION['from_reference'])

    return cs


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

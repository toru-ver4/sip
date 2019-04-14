#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
OCIO用のColorSpaceを作る。
"""

import os
import PyOpenColorIO as OCIO
from make_ocio_utility import get_colorspace_name
from make_ocio_utility import BT1886_CS, REFERENCE_ROLE, P3_ST2084_CS,\
    ALEXA_LOGC_CS, BT2020_ST2084_CS, BT2020_CS, BT2020_LOGC_CS, SRGB_CS,\
    BT2020_LOG3G10_CS


DIRECTION_OPS = {
    'forward': OCIO.Constants.TRANSFORM_DIR_FORWARD,
    'inverse': OCIO.Constants.TRANSFORM_DIR_INVERSE}
INTERPOLATION_OPS = {
    'linear': OCIO.Constants.INTERP_LINEAR,
    'tetrahedral': OCIO.Constants.INTERP_TETRAHEDRAL}
COLOR_SPACE_DIRECTION = {
    'to_reference': OCIO.Constants.COLORSPACE_DIR_TO_REFERENCE,
    'from_reference': OCIO.Constants.COLORSPACE_DIR_FROM_REFERENCE}


LUT_FILE_SRGB = "sRGB_to_Linear.spi1d"
LUT_FILE_GAMMA24 = "Gamma_2.4_to_Linear.spi1d"
LUT_FILE_ST2084 = "SMPTE_ST2084_to_Linear.spi1d"
LUT_FILE_LOG_C = "ARRI_LOG_C_to_Linear.spi1d"
LUT_FILE_VLOG = "Panasonic_VLog_to_Linear.spi1d"
LUT_FILE_SLOG3 = "SONY_S-Log3_(IRE_Base)_to_Linear.spi1d"
LUT_FILE_LOG3G10 = "RED_Log3G10_to_Linear.spi1d"
LUT_FILE_LOG3G12 = "RED_Log3G12_to_Linear.spi1d"
LUT_FILE_NLOG = "Nikon_N-Log_to_Linear.spi1d"
LUT_FILE_DLOG = "DJI_D-Log_to_Linear.spi1d"
LUT_FILE_FLOG = "FUJIFILM_F-Log_to_Linear.spi1d"

AP0_TO_BT709_MTX = [2.5219347298199275, -1.13702389648161, -0.38491083358651407, 0.0, -0.27547942789225904, 1.3698289786449884, -0.09434955068309422, 0.0, -0.015982869997415383, -0.14778923413163852, 1.1637721041802542, 0.0, 0.0, 0.0, 0.0, 1.0]
BT709_TO_AP0_MTX = [0.43957568421668025, 0.3839125893365086, 0.17651172648967858, 0.0, 0.08960038290392143, 0.8147141542066522, 0.09568546289518032, 0.0, 0.017415482729199242, 0.10873435223667391, 0.8738501650336234, 0.0, 0.0, 0.0, 0.0, 1.0]

AP0_TO_P3_MTX = [2.025287327569135, -0.6919621723088936, -0.3333251554520117, 0.0, -0.1826215060726654, 1.2866160058179315, -0.1039944996861809, 0.0, 0.008584552508200815, -0.05481629005504889, 1.0462317375942687, 0.0, 0.0, 0.0, 0.0, 1.0]
P3_TO_AP0_MTX = [0.5188414630660481, 0.2873003676579024, 0.1938581693189168, 0.0, 0.07361168404321847, 0.8212994981474907, 0.10508881781504453, 0.0, -0.00040039156270921895, 0.04067382482695071, 0.9597265667352549, 0.0, 0.0, 0.0, 0.0, 1.0]

ACES2065_1_TO_sRGB_MTX = [2.5216494298433054, -1.1368885542222593, -0.38491759319444535, 0.0, -0.2752135512440264, 1.3697051510263252, -0.09439245077651989, 0.0, -0.01592501009046427, -0.14780636811079967, 1.1638058159424305, 0.0, 0.0, 0.0, 0.0, 1.0]
sRGB_TO_ACES2065_1_MTX = [0.43958564415417334, 0.3839294030137976, 0.17653273636594352, 0.0, 0.08953957351755433, 0.8147498350914474, 0.09568360609267229, 0.0, 0.01738718324341058, 0.10873911432148052, 0.8738205876139565, 0.0, 0.0, 0.0, 0.0, 1.0]

ACES2065_1_TO_BT709_MTX = [2.5219347298199275, -1.13702389648161, -0.38491083358651407, 0.0, -0.27547942789225904, 1.3698289786449884, -0.09434955068309422, 0.0, -0.015982869997415383, -0.14778923413163852, 1.1637721041802542, 0.0, 0.0, 0.0, 0.0, 1.0]
BT709_TO_ACES2065_1_MTX = [0.43957568421668025, 0.3839125893365086, 0.17651172648967858, 0.0, 0.08960038290392143, 0.8147141542066522, 0.09568546289518032, 0.0, 0.017415482729199242, 0.10873435223667391, 0.8738501650336234, 0.0, 0.0, 0.0, 0.0, 1.0]

ACES2065_1_TO_BT2020_MTX = [1.49086870465701, -0.2687129790829561, -0.22215572570462597, 0.0, -0.07923721060283284, 1.1793685831111034, -0.10013137246080642, 0.0, 0.002778100767079354, -0.03043361463153356, 1.0276555139123698, 0.0, 0.0, 0.0, 0.0, 1.0]
BT2020_TO_ACES2065_1_MTX = [0.678891150633901, 0.1588684223720231, 0.16224042703694286, 0.0, 0.045570830898021907, 0.8607127720288462, 0.0937163970788858, 0.0, -0.00048571035183551524, 0.025060195736249565, 0.9754255146150821, 0.0, 0.0, 0.0, 0.0, 1.0]

ACES2065_1_TO_DCI_P3_MTX = [2.025287327569135, -0.6919621723088936, -0.3333251554520117, 0.0, -0.1826215060726654, 1.2866160058179315, -0.1039944996861809, 0.0, 0.008584552508200815, -0.05481629005504889, 1.0462317375942687, 0.0, 0.0, 0.0, 0.0, 1.0]
DCI_P3_TO_ACES2065_1_MTX = [0.5188414630660481, 0.2873003676579024, 0.1938581693189168, 0.0, 0.07361168404321847, 0.8212994981474907, 0.10508881781504453, 0.0, -0.00040039156270921895, 0.04067382482695071, 0.9597265667352549, 0.0, 0.0, 0.0, 0.0, 1.0]

ACES2065_1_TO_S_Gamut3_MTX = [1.3316572111294034, -0.18756110057228934, -0.14409611062855196, 0.0, -0.028013124402701994, 0.9887375645330418, 0.039275559936834584, 0.0, 0.012557452753235608, -0.005067905237689978, 0.992510452625251, 0.0, 0.0, 0.0, 0.0, 1.0]
S_Gamut3_TO_ACES2065_1_MTX = [0.7529825954091843, 0.14337021624570287, 0.10364718843865435, 0.0, 0.021707697396902932, 1.015318835457289, -0.03702653286868595, 0.0, -0.009416052762309595, 0.003370417865718212, 1.006045634932962, 0.0, 0.0, 0.0, 0.0, 1.0]

ACES2065_1_TO_S_Gamut3Cine_MTX = [1.5554591070218062, -0.39328079845219244, -0.16217830874193623, 0.0, 0.009021614455608342, 0.9185569566341988, 0.07242142895660851, 0.0, 0.04426406659061803, 0.011850260714646399, 0.9438856727316312, 0.0, 0.0, 0.0, 0.0, 1.0]
S_Gamut3Cine_TO_ACES2065_1_MTX = [0.6387886672293862, 0.2723514336942939, 0.0888598991690205, 0.0, -0.00391590608174729, 1.0880732308222416, -0.08415732489209929, 0.0, -0.029907202082976047, -0.02643257992898781, 1.0563397820484384, 0.0, 0.0, 0.0, 0.0, 1.0]

ACES2065_1_TO_ALEXA_Wide_Gamut_MTX = [1.5159863828803046, -0.36134185877532626, -0.1546444591922093, 0.0, -0.12832757994277402, 1.0193145872748801, 0.10901239485034993, 0.0, -0.0105107560646713, 0.0608329324823131, 0.9496764953636239, 0.0, 0.0, 0.0, 0.0, 1.0]
ALEXA_Wide_Gamut_TO_ACES2065_1_MTX = [0.6802059160731788, 0.23613674995240488, 0.08365740736382404, 0.0, 0.08541506948574616, 1.0174707719961325, -0.10288585499780477, 0.0, 0.002056264751198366, -0.06256228368093003, 1.0605062480735412, 0.0, 0.0, 0.0, 0.0, 1.0]

ACES2065_1_TO_ACEScg_MTX = [1.4514393160716577, -0.23651074688936, -0.21492856930836365, 0.0, -0.07655377331426277, 1.176229699811789, -0.09967592645036046, 0.0, 0.008316148424960776, -0.0060324497909093056, 0.9977163014129821, 0.0, 0.0, 0.0, 0.0, 1.0]
ACEScg_TO_ACES2065_1_MTX = [0.6954522413585676, 0.1406786964707304, 0.16386906221356923, 0.0, 0.04479456335249944, 0.8596711184429684, 0.09553431821028617, 0.0, -0.005525882558110763, 0.004025210305976633, 1.0015006722516306, 0.0, 0.0, 0.0, 0.0, 1.0]

ACES2065_1_TO_V_Gamut_MTX = [1.3854727150759145, -0.23703839319860245, -0.1484337843054005, 0.0, -0.029222206879136493, 1.0183489534695143, 0.01087309327003224, 0.0, 0.012227638155743062, 0.002999660015647367, 0.9847717590179481, 0.0, 0.0, 0.0, 0.0, 1.0]
V_Gamut_TO_ACES2065_1_MTX = [0.7243773909797958, 0.1682951219660712, 0.10732649807743383, 0.0, 0.020883920666206392, 0.98686469419941, -0.00774813444017126, 0.0, -0.009058168754218448, -0.005095454442369639, 1.0141538526750664, 0.0, 0.0, 0.0, 0.0, 1.0]

ACES2065_1_TO_REDWideGamutRGB_MTX = [1.2659750874973834, -0.13733317128270045, -0.12864056874710197, 0.0, -0.019875416136894024, 0.9415327918528358, 0.07834213436042345, 0.0, 0.06245948160522238, 0.20903250744273216, 0.7285078051637488, 0.0, 0.0, 0.0, 0.0, 1.0]
REDWideGamutRGB_TO_ACES2065_1_MTX = [0.7848686125011931, 0.085760033547838, 0.1293703649742697, 0.0, 0.02270961620650155, 1.0905568785915896, -0.11326601437264593, 0.0, -0.07380776746826465, -0.32026886252265924, 1.394076859469402, 0.0, 0.0, 0.0, 0.0, 1.0]


def make_ref_color_space():
    cs = OCIO.ColorSpace(name=get_colorspace_name(REFERENCE_ROLE))
    cs.setDescription("reference")
    cs.setBitDepth(OCIO.Constants.BIT_DEPTH_F32)
    cs.setAllocation(OCIO.Constants.ALLOCATION_LG2)
    cs.setAllocationVars([-8, 5, 0.00390625])

    return cs


def make_raw_color_space():
    cs = OCIO.ColorSpace(name='raw')
    cs.setDescription("raw")
    cs.setBitDepth(OCIO.Constants.BIT_DEPTH_F32)
    cs.setAllocation(OCIO.Constants.ALLOCATION_UNIFORM)
    cs.setAllocationVars([0, 1])

    return cs


def make_typical_color_space(name=get_colorspace_name(BT1886_CS),
                             description="bt1886",
                             allocation=OCIO.Constants.ALLOCATION_UNIFORM,
                             allocationVars=[0, 1],
                             eotf_lut_file=LUT_FILE_GAMMA24,
                             to_ref_mtx=BT709_TO_ACES2065_1_MTX,
                             from_ref_mtx=ACES2065_1_TO_BT709_MTX):
    """
    典型的な Color Space を作成する。
    """

    cs = OCIO.ColorSpace(name=name)
    cs.setDescription(description)
    cs.setBitDepth(OCIO.Constants.BIT_DEPTH_F32)
    cs.setAllocation(allocation)
    cs.setAllocationVars(allocationVars)

    # to reference
    file_to_ref = OCIO.FileTransform(eotf_lut_file,
                                     direction=DIRECTION_OPS['forward'],
                                     interpolation=INTERPOLATION_OPS['linear'])
    matrix_to_ref = OCIO.MatrixTransform(matrix=to_ref_mtx,
                                         direction=DIRECTION_OPS['forward'])
    group_to_ref = OCIO.GroupTransform([file_to_ref, matrix_to_ref])
    cs.setTransform(group_to_ref, COLOR_SPACE_DIRECTION['to_reference'])

    # from reference
    file_from_ref = OCIO.FileTransform(eotf_lut_file,
                                       direction=DIRECTION_OPS['inverse'],
                                       interpolation=INTERPOLATION_OPS['linear'])
    matrix_from_ref = OCIO.MatrixTransform(matrix=from_ref_mtx,
                                           direction=DIRECTION_OPS['forward'])
    group_from_ref = OCIO.GroupTransform([matrix_from_ref, file_from_ref])
    cs.setTransform(group_from_ref, COLOR_SPACE_DIRECTION['from_reference'])

    return cs


def make_srgb_color_space():
    cs = make_typical_color_space(name=get_colorspace_name(SRGB_CS),
                                  description="gamut: BT.709, gamma: sRGB",
                                  eotf_lut_file=LUT_FILE_SRGB,
                                  to_ref_mtx=BT709_TO_ACES2065_1_MTX,
                                  from_ref_mtx=ACES2065_1_TO_BT709_MTX)
    return cs


def make_bt1886_color_space():
    cs = make_typical_color_space(name=get_colorspace_name(BT1886_CS),
                                  description="gamut: BT.709, gamma: 2.4",
                                  eotf_lut_file=LUT_FILE_GAMMA24,
                                  to_ref_mtx=BT709_TO_ACES2065_1_MTX,
                                  from_ref_mtx=ACES2065_1_TO_BT709_MTX)
    return cs


def make_bt2020_color_space():
    cs = make_typical_color_space(name=get_colorspace_name(BT2020_CS),
                                  description="gamut: BT.2020, gamma: 2.4",
                                  eotf_lut_file=LUT_FILE_GAMMA24,
                                  to_ref_mtx=BT2020_TO_ACES2065_1_MTX,
                                  from_ref_mtx=ACES2065_1_TO_BT2020_MTX)
    return cs


def make_p3_st2084_color_space():
    cs = make_typical_color_space(name=get_colorspace_name(P3_ST2084_CS),
                                  description="gamut: DCI-P3, gamma: ST2084",
                                  eotf_lut_file=LUT_FILE_ST2084,
                                  to_ref_mtx=DCI_P3_TO_ACES2065_1_MTX,
                                  from_ref_mtx=ACES2065_1_TO_DCI_P3_MTX)
    return cs


def make_bt2020_st2084_color_space():
    cs = make_typical_color_space(name=get_colorspace_name(BT2020_ST2084_CS),
                                  description="gamut: BT.2020, gamma: ST2084",
                                  eotf_lut_file=LUT_FILE_ST2084,
                                  to_ref_mtx=BT2020_TO_ACES2065_1_MTX,
                                  from_ref_mtx=ACES2065_1_TO_BT2020_MTX)
    return cs


def make_arri_logc_color_space():
    cs = make_typical_color_space(
        name=get_colorspace_name(ALEXA_LOGC_CS),
        description="gamut: ALEXA Wide Gamut, gamma: LogC",
        eotf_lut_file=LUT_FILE_LOG_C,
        to_ref_mtx=ALEXA_Wide_Gamut_TO_ACES2065_1_MTX,
        from_ref_mtx=ACES2065_1_TO_ALEXA_Wide_Gamut_MTX)
    return cs


def make_bt2020_logc_color_space():
    cs = make_typical_color_space(
        name=get_colorspace_name(BT2020_LOGC_CS),
        description="gamut: BT.2020, gamma: LogC",
        eotf_lut_file=LUT_FILE_LOG_C,
        to_ref_mtx=BT2020_TO_ACES2065_1_MTX,
        from_ref_mtx=ACES2065_1_TO_BT2020_MTX)
    return cs


def make_bt2020_log3g10_color_space():
    cs = make_typical_color_space(
        name=get_colorspace_name(BT2020_LOG3G10_CS),
        description="gamut: BT.2020, gamma: Log3g10",
        eotf_lut_file=LUT_FILE_LOG3G10,
        to_ref_mtx=BT2020_TO_ACES2065_1_MTX,
        from_ref_mtx=ACES2065_1_TO_BT2020_MTX)
    return cs


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

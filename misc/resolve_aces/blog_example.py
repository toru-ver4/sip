import numpy as np
from colour import RGB_to_RGB
from colour.models import BT709_COLOURSPACE, BT2020_COLOURSPACE
chromatic_adapdation = "XYZ Scaling"
bt709_red = np.array([1023, 0, 0]) / 1023
gamma = 2.4
ap0_red = RGB_to_RGB(bt709_red, BT709_COLOURSPACE, BT2020_COLOURSPACE, chromatic_adapdation)
print(np.uint16(np.round((ap0_red ** (1/gamma)) * 1023)))

# import numpy as np
# from colour import RGB_COLOURSPACES
# from colour import RGB_to_RGB
# ACES_AP0 = 'ACES2065-1'
# ACES_AP1 = 'ACEScg'
# cs_name_list = ['ITU-R BT.709', 'P3-D65', 'ITU-R BT.2020', ACES_AP1, ACES_AP0]
# dst_cs = RGB_COLOURSPACES[ACES_AP0]
# src_primaries = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
# src_primaries = np.array(src_primaries)
# dst_primaries = {}
# for src_cs_name in cs_name_list:
#     src_cs = RGB_COLOURSPACES[src_cs_name]
#     chromatic_acaptation = "XYZ Scaling"
#     temp = RGB_to_RGB(src_primaries, src_cs, dst_cs, chromatic_acaptation)
#     temp = np.clip(temp, 0.0, 1.0)
#     dst_primaries[src_cs_name] = temp ** (1/2.4)

# print(dst_primaries)

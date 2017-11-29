# -*- coding: utf-8 -*-

"""
EOTF and Gamut controller
"""

import os


# window size
# -------------------
gamut_pane_width = 480
gamma_pane_width = 640
base_pane_width = gamut_pane_width + gamma_pane_width
base_pane_height = 960
gamma_scale_width = gamma_pane_width - 96

# font
# -----------------------
default_font_size = 14
plot_font_size = 13

# scale
# ----------------------
scale_default_value = 3
scale_max_value = 10
scale_min_value = 0

# eotf control
# ----------------
eotf_row_num = 4

# gamut control
# ----------------
gamut_row_num = 3

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

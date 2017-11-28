# -*- coding: utf-8 -*-

"""
EOTF and Gamut controller
"""

import os


# window size
plot_pane_width = 480
ctrl_pane_width = 512
base_pane_width = plot_pane_width + ctrl_pane_width
base_pane_height = 960
gamma_scale_width = ctrl_pane_width - 96

# scale
scale_default_value = 3
scale_max_value = 10
scale_min_value = 0

# eotf control
eotf_row_num = 4

# gamut control
gamut_row_num = 4

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

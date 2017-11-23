# -*- coding: utf-8 -*-

"""
EOTF and Gamut controller
"""

import os
import tkinter as tk
from tkinter import ttk


gamma_list = ["2.4", "HLG", "PQ", "LOG3G10"]


class TyStatus(ttk.LabelFrame):
    def __init__(self, master=None, text="", labelanchor=tk.NW):
        super().__init__(master, text=text, labelanchor=labelanchor)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text="display status")
        self.label.pack(fill=tk.BOTH, expand=1)


class EotfControl(ttk.LabelFrame):
    def __init__(self, master=None, text="", labelanchor=tk.NW):
        super().__init__(master, text=text, labelanchor=labelanchor)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # prepare the basic pane
        base_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        base_pane.pack(fill=tk.BOTH, expand=1)
        g_button_pane = ttk.PanedWindow(base_pane, orient=tk.VERTICAL)
        g_button_pane.pack(fill=tk.BOTH, expand=1)
        base_pane.add(g_button_pane)
        r_button_pane = ttk.PanedWindow(base_pane, orient=tk.VERTICAL)
        r_button_pane.pack(fill=tk.BOTH, expand=1)
        base_pane.add(r_button_pane)
        scale_pane = ttk.PanedWindow(base_pane, orient=tk.VERTICAL)
        scale_pane.pack(fill=tk.BOTH, expand=1)
        base_pane.add(scale_pane)

        # put widgets
        widgets_list = self.get_eotf_ctrl_widgets_array(g_button_pane,
                                                        r_button_pane,
                                                        scale_pane)
        self.set_widgets(widgets_list, g_button_pane,
                         r_button_pane, scale_pane)
        # select_button = ttk.Button(line_pane, text="gamma=2.2")
        # line_pane.add(select_button)
        # reset_button = ttk.Button(line_pane, text="Reset")
        # line_pane.add(reset_button)
        # scale = ttk.Scale(line_pane, orient='h')
        # line_pane.add(scale)

    def set_widgets(self, widgets_list, g_parent, r_parent, s_parent):
        """
        Set widgets to the panedwindow.

        Parameters
        ----------
        self : -
        widgets_list : A list of dictionaries below.
            [{"idx", "name", "gamma_button", "reset_button", "scale"}]
        g_parent : widget class
            parent for gamma button.
        r_parent : widget class
            parent for reset button.
        s_parent : widget class
            parent for scale.

        Returns
        -------
        None

        Notes
        -----
        None

        """
        for parts in widgets_list:
            g_parent.add(parts['gamma_button'])
            r_parent.add(parts['reset_button'])
            s_parent.add(parts['scale'])

    def get_eotf_ctrl_widgets_array(self, g_parent, r_parent, s_parent,
                                    gamma_list=gamma_list):
        """
        Get widgets array for control eotf.

        Parameters
        ----------
        self : -
        g_parent : widget class
            parent for gamma button.
        r_parent : widget class
            parent for reset button.
        s_parent : widget class
            parent for scale.
        gamma_list : array of character
            a list contain supported gamma curves.

        Returns
        -------
        A list of dictionaries below.
            [{"idx", "name", "gamma_button", "reset_button", "scale"}]

        Notes
        -----
        None

        """
        widget_list = [0] * len(gamma_list)
        for idx, gamma in enumerate(gamma_list):
            widget_list[idx] = {}
            widget_list[idx]['idx'] = idx
            widget_list[idx]['name'] = gamma
            widget_list[idx]['gamma_button'] \
                = ttk.Button(g_parent, text=gamma)
            widget_list[idx]['reset_button'] \
                = ttk.Button(r_parent, text="Reset")
            widget_list[idx]['scale'] \
                = ttk.Scale(s_parent, orient='h', from_=0, to=1, value=0.5)

        return widget_list


class GamutControl(ttk.LabelFrame):
    def __init__(self, master=None, text="", labelanchor=tk.NW):
        super().__init__(master, text=text, labelanchor=labelanchor)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text="gamut control")
        self.label.pack(fill=tk.BOTH, expand=1)


class EotfPlot(ttk.LabelFrame):
    def __init__(self, master=None, text="", labelanchor=tk.NW):
        super().__init__(master, text=text, labelanchor=labelanchor)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text="plot eotf")
        self.label.pack(fill=tk.BOTH, expand=1)


class GamutPlot(ttk.LabelFrame):
    def __init__(self, master=None, text="", labelanchor=tk.NW):
        super().__init__(master, text=text, labelanchor=labelanchor)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.label = ttk.Label(self, text="plot gamut")
        self.label.pack(fill=tk.BOTH, expand=1)




if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parts_array = get_parts_array()
    print(parts_array)

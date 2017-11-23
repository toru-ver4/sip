# -*- coding: utf-8 -*-

"""
EOTF and Gamut controller
"""
import matplotlib
matplotlib.use('TkAgg')

import os
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plot_utility as pu
import numpy as np
import gamma_curve as gm

gamma_list = ["2.4", "HLG", "PQ", "LOG3G10"]
scale_default_value = 3
scale_max_value = 5
scale_min_value = 0


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

    def set_callback_function(self, callback_func):
        self.callback_func = callback_func

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
        self.widgets_list\
            = self.get_eotf_ctrl_widgets_array(g_button_pane, r_button_pane,
                                               scale_pane)
        self.set_widgets(self.widgets_list, g_button_pane,
                         r_button_pane, scale_pane)
        self.set_callback_to_widgets(self.widgets_list,
                                     self.pre_callback_func, self.reset_func)

    def pre_callback_func(self, event, args):
        idx = args
        gamma = self.widgets_list[idx]['name']
        gain = self.widgets_list[idx]['scale'].get()
        self.callback_func(event=event, gamma=gamma, gain=gain)

    def reset_func(self, event, args):
        idx = args
        self.widgets_list[idx]['scale'].set(scale_default_value)
        self.pre_callback_func(event=None, args=args)

    def set_callback_to_widgets(self, widgets_list,
                                pre_callback_func, reset_func):
        """
        Set callback function to widgets.

        Parameters
        ----------
        self : -
        widgets_list : A list of dictionaries below.
            [{"idx", "name", "gamma_button", "reset_button", "scale"}]
        """
        for idx, widget in enumerate(widgets_list):
            widget['gamma_button'].bind("<ButtonRelease-1>",
                                        lambda event, args=idx:
                                        pre_callback_func(event, args))
            widget['reset_button'].bind("<ButtonRelease-1>",
                                        lambda event, args=idx:
                                        reset_func(event, args))
            widget['scale'].bind("<B1-Motion>", lambda event, args=idx:
                                 pre_callback_func(event, args))

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
                = ttk.Scale(s_parent, orient='h',
                            from_=scale_min_value, to=scale_max_value,
                            value=scale_default_value)

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
        self.first_draw(gamma="2.2")

    def first_draw(self, gamma="2.2"):
        self.x = np.linspace(0, 1, 1024)
        y = (self.x ** 2.2) * 100
        xtick = [x * 128 for x in range((1024//128)+1)]
        ytick = [x * 100 for x in range((1000//100)+1)]
        fig, ax1\
            = pu.plot_1_graph_ret_figure(fontsize=10,
                                         figsize=(5, 4),
                                         graph_title="Title",
                                         graph_title_size=None,
                                         xlabel="X Axis Label",
                                         ylabel="Y Axis Label",
                                         axis_label_size=None,
                                         legend_size=10,
                                         xlim=(0, 1024),
                                         ylim=(0, 1050),
                                         xtick=xtick,
                                         ytick=ytick,
                                         xtick_size=None, ytick_size=None,
                                         linewidth=3)
        self.line, = ax1.plot(self.x * 1024, y, label=gamma)
        plt.legend(loc='upper left')

        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def update_draw(self, gamma, gain):
        y = self.calc_gamma(gamma, gain)
        self.line.set_ydata(y)
        self.canvas.show()

    def calc_gamma(self, gamma, gain):
        if gamma == "2.4":
            y = (self.x ** 2.4) * gain/scale_default_value * 100
        elif gamma == "PQ":
            y = gm.get_bt2100_pq_curve(self.x) * gain/scale_default_value
        elif gamma == "HLG":
            y = (gm.get_bt2100_hlg_curve(self.x) ** 1.2) \
                * gain/scale_default_value * 1000
        else:
            y = (self.x ** 2.4) * gain/scale_default_value * 100

        y[y > 1000] = 1000
        return y

    def get_callback_func(self):
        return self.update_parameters

    def update_parameters(self, event, gamma='2.4',
                          gain="scale_default_value"):
        """
        Update EOTF parameters and re-plot the graph.

        Parameters
        ----------
        self : -
        gamma : character
            type of gamma curve.
        gain : double
            gain for gamma curve.

        Returns
        -------
        None

        Notes
        -----
        None

        """
        print("gamma={}, gain={}".format(gamma, gain))
        self.update_draw(gamma, gain)


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

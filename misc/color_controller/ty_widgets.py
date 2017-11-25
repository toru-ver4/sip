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
import color_convert as cc
import sys

gamma_list = ["2.4", "HLG", "PQ", "LOG3G10"]
gamut_list = ["REC709, REC2020, DCI", "RWG"]
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
        g_button_pane = ttk.Frame(base_pane)
        g_button_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        base_pane.add(g_button_pane)
        r_button_pane = ttk.Frame(base_pane)
        r_button_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        base_pane.add(r_button_pane)
        scale_pane = ttk.Frame(base_pane)
        scale_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
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
            parts['gamma_button'].pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            parts['reset_button'].pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            parts['scale'].pack(side=tk.TOP, fill=tk.BOTH, expand=1)

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
        self.rb_value = tk.StringVar()
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # self.label = ttk.Label(self, text="gamut control")
        # self.label.pack(fill=tk.BOTH, expand=1)
        base_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        base_pane.pack(fill=tk.BOTH, expand=1)
        g_button_frame = ttk.Frame(base_pane)
        g_button_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        base_pane.add(g_button_frame)
        r_button_frame = ttk.Frame(base_pane)
        r_button_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        base_pane.add(r_button_frame)
        widgets_list = self.get_widgets_array(g_button_frame, r_button_frame)
        print(widgets_list)

    def set_widgets(self, g_parent, r_parent):
        # for parts in widgets_list:
        #     parts['gamma_button'].pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        #     parts['reset_button'].pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        #     parts['scale'].pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        pass

    def get_widgets_array(self, g_parent, r_parent, gamut_list=gamut_list):
        """
        Get widgets array for control gamut.

        Parameters
        ----------
        self : -
        g_parent : widget class
            parent for gamut button.
        r_parent : widget class
            parent for radio button.
        gamut_list : array of character
            a list contain supported gamut.

        Returns
        -------
        A list of dictionaries below.
            {"gamut_button"[], "radio_button"[off/on], "primary"[r/g/b]}

        Notes
        -----
        None

        """
        widget_list = {"gamut_button": [], "radio_button": [], "primary": []}
        widget_list["gamut_button"] = [0] * len(gamut_list)
        widget_list["radio_button"] = [0] * 2
        widget_list["primary"] = [0] * 3
        for idx, gamut in enumerate(gamut_list):
            widget_list["gamut_button"][idx] = ttk.Button(g_parent, text=gamut)
        widget_list["radio_button"][0]\
            = ttk.Radiobutton(r_parent, text='clip off',
                              variable=self.rb_value, value="off")
        widget_list["radio_button"][1]\
            = ttk.Radiobutton(r_parent, text='clip on',
                              variable=self.rb_value, value="on")

        return widget_list


class EotfPlot(ttk.LabelFrame):
    def __init__(self, master=None, text="", labelanchor=tk.NW):
        super().__init__(master, text=text, labelanchor=labelanchor)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # self.label = ttk.Label(self, text="plot eotf")
        # self.label.pack(fill=tk.BOTH, expand=1)
        self.first_draw(gamma="2.4")

    def first_draw(self, gamma="2.4"):
        self.x = np.linspace(0, 1, 1024)
        y = (self.x ** 2.4) * 100
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
        sys.stdout.flush()
        self.update_draw(gamma, gain)


class GamutPlot(ttk.LabelFrame):
    def __init__(self, master=None, text="", labelanchor=tk.NW):
        super().__init__(master, text=text, labelanchor=labelanchor)
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # self.label = ttk.Label(self, text="plot gamut")
        # self.label.pack(fill=tk.BOTH, expand=1)
        self.first_draw()

    def first_draw(self, gamma="2.4", csv_name="./data/xyz_value.csv"):
        csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                csv_name)
        large_xyz = np.loadtxt(csv_file, delimiter=',', usecols=(1, 2, 3))
        large_xyz = np.reshape(large_xyz,
                               (1, large_xyz.shape[0], large_xyz.shape[1]))
        xy = cc.large_xyz_to_small_xy(large_xyz)
        min_idx = np.argmin(xy, axis=1)[0][1]
        max_idx = np.argmax(xy, axis=1)[0][0]
        xy = np.append(xy, np.array([[xy[0][max_idx]]]), axis=1)
        xy = np.append(xy, np.array([[xy[0][min_idx]]]), axis=1)

        xtick = [x * 0.1 for x in range(-1, 10)]
        ytick = [x * 0.1 for x in range(-1, 11)]
        fig, ax1\
            = pu.plot_1_graph_ret_figure(fontsize=10,
                                         figsize=(5, 5),
                                         graph_title="Title",
                                         graph_title_size=None,
                                         xlabel="X Axis Label",
                                         ylabel="Y Axis Label",
                                         axis_label_size=None,
                                         legend_size=10,
                                         xlim=(-0.1, 0.9),
                                         ylim=(-0.1, 1.0),
                                         xtick=xtick,
                                         ytick=ytick,
                                         xtick_size=None, ytick_size=None,
                                         linewidth=1)
        ax1.plot(xy[0, :, 0], xy[0, :, 1], 'k-', label="CIE2006")
        plt.legend(loc='upper right')

        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)        


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

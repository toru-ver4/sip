# -*- coding: utf-8 -*-

"""
EOTF and Gamut controller
"""

import os
import tkinter as tk
from tkinter import ttk
import ty_widgets as tyw
import parameter as prm
from tkinter.font import nametofont


class Application(ttk.Frame):
    def __init__(self, master=None, width=prm.base_pane_width):
        super().__init__(master)
        self.root = master
        default_font = nametofont("TkDefaultFont")
        default_font.configure(size=prm.default_font_size)
        self.root.option_add("*Font", default_font)
        self.root.geometry("{}x{}".format(prm.base_pane_width,
                                          prm.base_pane_height))
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # create base pane
        # ----------------
        status_frame = tyw.TyStatus(master=self.root, text="status")
        status_frame.pack(fill=tk.BOTH, expand=1)

        base_pane = ttk.PanedWindow(orient=tk.HORIZONTAL)
        base_pane.pack(fill=tk.BOTH, expand=1)
        gamma_pane = ttk.PanedWindow(base_pane, orient=tk.VERTICAL,
                                     width=prm.gamma_pane_width)
        gamma_pane.pack(fill=tk.BOTH, expand=1)
        base_pane.add(gamma_pane)
        gamut_pane = ttk.PanedWindow(base_pane, orient=tk.VERTICAL,
                                     width=prm.gamut_pane_width)
        gamut_pane.pack(fill=tk.BOTH, expand=1)
        base_pane.add(gamut_pane)

        # add additional pane
        # -------------------
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        eotf_ctrl_frame = tyw.EotfControl(master=gamma_pane,
                                          text="EOTF control")
        gamma_pane.add(eotf_ctrl_frame)

        gamut_ctrl_frame = tyw.GamutControl(master=gamut_pane,
                                            text="Gamut control")
        gamut_pane.add(gamut_ctrl_frame)

        eotf_plot_frame = tyw.EotfPlot(master=gamma_pane, text="plot eotf")
        gamma_pane.add(eotf_plot_frame)

        gamut_plot_frame = tyw.GamutPlot(master=gamut_pane, text="plot gamut")
        gamut_pane.add(gamut_plot_frame)

        # manage callback fuctions
        # ------------------------
        eotf_callback_func = eotf_plot_frame.get_callback_func()
        eotf_ctrl_frame.set_callback_func(eotf_callback_func)

        gamut_callback_func = gamut_plot_frame.get_callback_func()
        gamut_ctrl_frame.set_callback_func(gamut_callback_func)

    def _quit(self):
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent


def run():
    root = tk.Tk()
    root.geometry("1024x1024")
    app = Application(master=root)
    app.mainloop()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run()

# -*- coding: utf-8 -*-

"""
EOTF and Gamut controller
"""

import os
import tkinter as tk
from tkinter import ttk
import ty_widgets as tyw
import parameter as prm


class Application(ttk.Frame):
    def __init__(self, master=None, width=prm.base_pane_width):
        super().__init__(master)
        self.root = master
        self.root.geometry("{}x{}".format(prm.base_pane_width, 1024))
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # create base pane
        # ----------------
        base_pane = ttk.PanedWindow(orient=tk.HORIZONTAL)
        base_pane.pack(fill=tk.BOTH, expand=1)
        ctrl_pane = ttk.PanedWindow(base_pane, orient=tk.VERTICAL,
                                    width=prm.ctrl_pane_width)
        ctrl_pane.pack(fill=tk.BOTH, expand=1)
        base_pane.add(ctrl_pane)
        plot_pane = ttk.PanedWindow(base_pane, orient=tk.VERTICAL,
                                    width=prm.plot_pane_width)
        plot_pane.pack(fill=tk.BOTH, expand=1)
        base_pane.add(plot_pane)

        # add additional pane
        # -------------------
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        status_frame = tyw.TyStatus(master=ctrl_pane, text="status")
        ctrl_pane.add(status_frame)

        eotf_ctrl_frame = tyw.EotfControl(master=ctrl_pane,
                                          text="EOTF control")
        ctrl_pane.add(eotf_ctrl_frame)

        gamut_ctrl_frame = tyw.GamutControl(master=ctrl_pane,
                                            text="Gamut control")
        ctrl_pane.add(gamut_ctrl_frame)

        eotf_plot_frame = tyw.EotfPlot(master=plot_pane, text="plot eotf")
        plot_pane.add(eotf_plot_frame)

        gamut_plot_frame = tyw.GamutPlot(master=plot_pane, text="plot gamut")
        plot_pane.add(gamut_plot_frame)

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

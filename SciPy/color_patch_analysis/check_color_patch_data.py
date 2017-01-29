import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import pandas as pd


def show_color_patch_spectral_data():
    spectral_data = "./data/MacbethColorChecker_SpectralData.csv"
    data = np.loadtxt(spectral_data, delimiter=',', skiprows=3).T

    # plot
    # ----------------------------------
    v_num = 4
    h_num = 6
    plt.rcParams["font.size"] = 18
    f, axarr = plt.subplots(v_num, h_num, sharex='col', sharey='row',
                            figsize=(24, 16))
    for idx in range(24):
        h_idx = idx % h_num
        v_idx = idx // h_num
        axarr[v_idx, h_idx].grid()
        if v_idx == (v_num - 1):
            axarr[v_idx, h_idx].set_xlabel("wavelength [nm]")
        if h_idx == 0:
            axarr[v_idx, h_idx].set_ylabel("reflectance")
        axarr[v_idx, h_idx].set_xlim(380, 780)
        axarr[v_idx, h_idx].set_ylim(0, 1.0)
        axarr[v_idx, h_idx].set_xticks([400, 500, 600, 700])
        axarr[v_idx, h_idx].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axarr[v_idx, h_idx].plot(data[0], data[idx + 1])
    plt.show()


def make_color_patch_image():
    width = 200
    height = 200
    


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    show_color_patch_spectral_data()

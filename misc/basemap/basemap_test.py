# -*- coding: utf-8 -*-

"""
EOTF and Gamut controller
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def test():
    """
    以下URLのコードを完コピ
    https://qiita.com/msrks/items/ed18a2653bc177a24cca
    """
    fig = plt.figure(figsize=(12, 8))
    m = Basemap(projection='merc',
                resolution='i',  # l, i. h, f
                llcrnrlon=136,
                llcrnrlat=36,
                urcrnrlon=138,
                urcrnrlat=37.7)

    m.drawcoastlines(color='lightgray')
    m.drawcountries(color='lightgray')
    m.fillcontinents(color='white', lake_color='#90FEFF')
    m.drawmapboundary(fill_color='#eeeeee')

    kanazawa_lon = 136 + 39/60
    kanazawa_lat = 36 + 33/60

    x1, y1 = m(kanazawa_lon, kanazawa_lat)
    m.plot(x1, y1, 'm.', markersize=10)
    plt.text(x1 + 5000, y1 + 5000, u"kanazawa")

    plt.show()
    fig.savefig('map.png')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test()

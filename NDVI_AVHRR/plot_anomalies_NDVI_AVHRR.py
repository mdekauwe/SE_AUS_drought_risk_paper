#!/usr/bin/env python

"""
Plot NDVI anomalies during droughts

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (09.03.2020)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from sklearn.cluster import KMeans
import gdal

def main(src_ds):

    ndates = src_ds.RasterCount

    (aus, aus_lat, aus_lon) = get_data(src_ds, 1)
    nrows, ncols = aus.shape

    plt.imshow(np.flipud(aus))
    plt.colorbar()
    plt.show()
    #"""

    year = 1982
    month = 1
    st_count = 0
    for i in range(1, ndates + 1):

        if year == 1983:
            break
        print(i, year, month, st_count)

        month += 1
        st_count += 1

        if month == 13:
            month = 1
            year += 1

    print(st_count)
    sys.exit()


def get_data(src_ds, band_count):
    band = src_ds.GetRasterBand(band_count)

    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    transform = src_ds.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]
    ulx, xres, xskew, uly, yskew, yres  = src_ds.GetGeoTransform()
    lrx = ulx + (src_ds.RasterXSize * xres)
    lry = uly + (src_ds.RasterYSize * yres)

    lats = np.linspace(uly, lry, rows)
    lons = np.linspace(ulx, lrx, cols)

    lonx, laty = np.meshgrid(lats, lons)
    latx = np.ones((len(lats),len(lons))).shape

    # subset by SE Aus
    data = band.ReadAsArray(0, 0, cols, rows)
    idy = np.argwhere((lats>=-39.2) & (lats<-28.1))
    idx = np.argwhere((lons>=140.7) & (lons<154.))

    aus = data[idy.min():idy.max(),idx.min():idx.max()]
    aus_lat = lats[idy.min():idy.max()]
    aus_lon = lons[idx.min():idx.max()]

    return (aus, aus_lat, aus_lon)


if __name__ == "__main__":

    fn = "AVHRR_EVI2_SEAUS_1982_2019.tif"
    src_ds = gdal.Open(fn)
    main(src_ds)

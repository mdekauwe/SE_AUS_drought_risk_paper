#!/usr/bin/env python

"""


That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (23.04.2019)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy import stats

def calc_correlation(plc_fname, eo_fname, eo_lat_fname, eo_lon_fname,
                     nrows, ncols):

    ds = xr.open_dataset(plc_fname)
    try:
        plc_lat = ds.y.values
        plc_lon = ds.x.values
    except:
        plc_lat = ds.lat.values
        plc_lon = ds.lon.values


    plc = ds.plc[:,0,:,:].values
    plc = np.nanmax(plc, axis=0)

    eo = np.fromfile(eo_fname).reshape(nrows, ncols)
    eo_lat = np.fromfile(eo_lat_fname)
    eo_lon = np.fromfile(eo_lon_fname)

    #plt.imshow(plc[30:50,50:75])
    #plt.colorbar()
    #plt.show()

    x = np.zeros(0)
    y = np.zeros(0)
    for i in range(nrows):
        for j in range(ncols):
            #print(i,j)
            r = find_nearest(plc_lat, eo_lat[i])
            c = find_nearest(plc_lon, eo_lon[j])

            x = np.append(x, eo[i,j])
            y = np.append(y, plc[r,c])

            #print(plc[r,c])

            #print(plc_lat[r], plc_lon[c], eo_lat[i], eo_lon[j])
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    y = y[~np.isnan(x)]
    x = x[~np.isnan(x)]


    y = y[(x>-40.)&(x<40.)]
    x = x[(x>-40.)&(x<40.)]

    (slope, intercept, r_value,
     p_value, std_err) = stats.linregress(x,y)

    print("%f %f" % (r_value, p_value))
    print("r-squared:", r_value**2)

    plt.plot(x, y, "k.")
    plt.plot(x, intercept + slope*x, 'r', label='fitted line')
    plt.show()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

if __name__ == "__main__":

    """
    print(" ")
    print("NDVI")
    print("======")
    #
    ## Millennium drought
    #
    fdir = "/Users/mdekauwe/Desktop/drought_desktop/outputs"
    plc_fname = os.path.join(fdir, "all_yrs_plc.nc")

    fdir = "../NDVI_AVHRR/"
    ndvi_fname = os.path.join(fdir, "md_change.bin")
    ndvi_lat_fname = os.path.join(fdir, "lat_ndvi.bin")
    ndvi_lon_fname = os.path.join(fdir, "lon_ndvi.bin")
    nrows = 245
    ncols = 294
    calc_correlation(plc_fname, ndvi_fname, ndvi_lat_fname, ndvi_lon_fname,
                     nrows, ncols)


    #
    ## Current drought
    #
    fdir = "/Users/mdekauwe/Desktop/current/outputs"
    plc_fname = os.path.join(fdir, "all_yrs_plc.nc")

    fdir = "../NDVI_AVHRR/"
    ndvi_fname = os.path.join(fdir, "cd_change.bin")
    ndvi_lat_fname = os.path.join(fdir, "lat_ndvi.bin")
    ndvi_lon_fname = os.path.join(fdir, "lon_ndvi.bin")

    calc_correlation(plc_fname, ndvi_fname, ndvi_lat_fname, ndvi_lon_fname,
                     nrows, ncols)
    print(" ")
    print("======")

    """
    print("VOD")
    print("======")
    #
    ## Millennium drought
    #

    fdir = "/Users/mdekauwe/Desktop/drought_desktop/outputs"
    plc_fname = os.path.join(fdir, "all_yrs_plc.nc")

    nrows = 140
    ncols = 180
    fdir = "../VOD/"
    vod_fname = os.path.join(fdir, "md_change.bin")
    vod_lat_fname = os.path.join(fdir, "lat_vod.bin")
    vod_lon_fname = os.path.join(fdir, "lon_vod.bin")

    calc_correlation(plc_fname, vod_fname, vod_lat_fname, vod_lon_fname,
                     nrows, ncols)
    """

    #
    ## Current drought
    #
    # 841*0.2; 681 * 0.2 (0.05 to 0.25 deg)
    # cdo remapcon,r168x136 all_yrs_plc.nc plc_degraded.nc
    fdir = "/Users/mdekauwe/Desktop/current/outputs"
    plc_fname = os.path.join(fdir, "plc_degraded.nc")

    nrows = 46
    ncols = 52
    fdir = "../VOD_LPDR/"
    vod_fname = os.path.join(fdir, "cd_change.bin")
    vod_lat_fname = os.path.join(fdir, "lat_vod.bin")
    vod_lon_fname = os.path.join(fdir, "lon_vod.bin")

    calc_correlation(plc_fname, vod_fname, vod_lat_fname, vod_lon_fname,
                     nrows, ncols)
    print(" ")
    print("======")
    """

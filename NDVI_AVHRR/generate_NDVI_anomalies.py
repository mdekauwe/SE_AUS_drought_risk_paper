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
import xarray as xr

def main(fname):

    ds = xr.open_dataset(fname)
    bottom, top = np.min(ds.lat).values, np.max(ds.lat).values
    left, right = np.min(ds.lon).values, np.max(ds.lon).values

    print(top, bottom, left, right)

    ndates, nrows, ncols = ds.NDVI.shape




    # Get baseline period
    # 1998-1999
    nyears = (1999 - 1983) + 1
    ndvi_pre = np.zeros((nyears,nrows,ncols))
    yr_count = 0
    for i in range(ndates):
        date = ds.time.values[i]
        year = int(str(ds.time.values[i]).split("-")[0])
        month = str(ds.time.values[i]).split("-")[1]

        if year >= 1983 and year < 2000 and month == "01":
            ndvi_pre[yr_count,:,:] = ds.NDVI[i,:,:]
            yr_count += 1

    ndvi_pre = np.nanmean(ndvi_pre, axis=0)
    ndvi_pre = np.where(ndvi_pre < 0.0, np.nan, ndvi_pre)

    plt.imshow(ndvi_pre)
    plt.colorbar()
    plt.show()
    sys.exit()

    # 2000-2009
    nyears = (2009 - 2000) + 1
    ndvi_dur = np.zeros((nyears,nrows,ncols))

    vals = np.zeros((nrows,ncols))
    yr_count = 0
    for i in range(ndates):
        date = ds.time.values[i]
        year = int(str(ds.time.values[i]).split("-")[0])
        month = str(ds.time.values[i]).split("-")[1]

        if year >= 2000 and year < 2010 and month == "01":
            ndvi_dur[yr_count,:,:] = ds.NDVI[i,:,:]
            yr_count += 1

    print(yr_count, nyears)
    ndvi_dur = np.nanmean(ndvi_dur, axis=0)
    ndvi_dur = np.where(ndvi_dur < 0.0, np.nan, ndvi_dur)

    #plt.imshow(ndvi_pre)
    #plt.colorbar()
    #plt.show()
    #sys.exit()


    chg = ((ndvi_dur - ndvi_pre) / ndvi_pre) * 100.0

    print(chg.shape)
    chg.tofile("md_change.bin")

    # Get baseline period
    # 1983-2016
    nyears = (2016 - 1983) + 1
    ndvi_pre = np.zeros((nyears,nrows,ncols))
    yr_count = 0
    for i in range(ndates):
        date = ds.time.values[i]
        year = int(str(ds.time.values[i]).split("-")[0])
        month = str(ds.time.values[i]).split("-")[1]

        if year >= 1983 and year < 2017 and month == "01":
            ndvi_pre[yr_count,:,:] = ds.NDVI[i,:,:]
            yr_count += 1

    ndvi_pre = np.nanmean(ndvi_pre, axis=0)
    ndvi_pre = np.where(ndvi_pre < 0.0, np.nan, ndvi_pre)


    # 2017-2019
    nyears = (2019 - 2017) + 1
    ndvi_dur = np.zeros((nyears,nrows,ncols))

    vals = np.zeros((nrows,ncols))
    yr_count = 0
    for i in range(ndates):
        date = ds.time.values[i]
        year = int(str(ds.time.values[i]).split("-")[0])
        month = str(ds.time.values[i]).split("-")[1]

        if year >= 2017 and year < 2020 and month == "01":
            ndvi_dur[yr_count,:,:] = ds.NDVI[i,:,:]
            yr_count += 1

    print(yr_count, nyears)
    ndvi_dur = np.nanmean(ndvi_dur, axis=0)
    ndvi_dur = np.where(ndvi_dur < 0.0, np.nan, ndvi_dur)


    chg = ((ndvi_dur - ndvi_pre) / ndvi_pre) * 100.0

    print(chg.shape)
    chg.tofile("cd_change.bin")



def plot_map(ax, var, cmap, i, top, bottom, left, right):
    print(np.nanmin(var), np.nanmax(var))
    vmin, vmax = -30., 30.
    #top, bottom = 89.8, -89.8
    #left, right = 0, 359.8
    img = ax.imshow(var, origin='lower',
                    transform=ccrs.PlateCarree(),
                    interpolation='nearest', cmap=cmap,
                    extent=(left, right, bottom, top),
                    vmin=vmin, vmax=vmax)
    ax.coastlines(resolution='10m', linewidth=1.0, color='black')
    #ax.add_feature(cartopy.feature.OCEAN)

    ax.set_xlim(140.7, 154)
    ax.set_ylim(-39.2, -28.1)

    if i == 0 or i >= 5:

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='black', alpha=0.5,
                          linestyle='--')
    else:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=0.5, color='black', alpha=0.5,
                          linestyle='--')

    #if i < 5:
    #s    gl.xlabels_bottom = False
    if i > 5:
        gl.ylabels_left = False

    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlocator = mticker.FixedLocator([141, 145,  149, 153])
    gl.ylocator = mticker.FixedLocator([-29, -32, -35, -38])

    return img

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

    fn = "AVHRR_CDRv5_NDVI_yearSeason_mean_1982_2019.nc"
    main(fn)

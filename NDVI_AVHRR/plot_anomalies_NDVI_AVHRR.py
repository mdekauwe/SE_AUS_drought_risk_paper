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
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import gdal

def main(src_ds):

    ndates = src_ds.RasterCount

    (aus, aus_lat, aus_lon) = get_data(src_ds, 1)
    nrows, ncols = aus.shape

    #plt.imshow(np.flipud(aus))
    #plt.colorbar()
    #plt.show()
    #"""

    """
    year = 1982
    month = 1
    st_count = 1
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
    """

    st_count = 13 # 1983

    # Get baseline period
    # 1993-1999
    nyears = (1999 - 1983) + 1
    ndvi_pre = np.zeros((nyears,nrows,ncols))
    vals = np.zeros((3,nrows,ncols)) # summer
    yr_count = 0
    count = st_count
    for year in np.arange(1983, 2000):
        for month in np.arange(1, 13):


            if month == 12:

                print(year, month, count)

                (aus, aus_lat, aus_lon) = get_data(src_ds, count)
                ndvi_pre[yr_count,:,:] += aus

                (aus, aus_lat, aus_lon) = get_data(src_ds, count+1)
                ndvi_pre[yr_count,:,:] += aus


                (aus, aus_lat, aus_lon) = get_data(src_ds, count+2)
                ndvi_pre[yr_count,:,:] += aus

                ndvi_pre[yr_count,:,:] /= 3


                #plt.imshow(ndvi_pre[yr_count,:,:])
                #plt.colorbar()
                #plt.show()
                #sys.exit()

            count += 1
        yr_count += 1

    ndvi_pre = np.nanmean(ndvi_pre, axis=0)
    ndvi_pre = np.flipud(ndvi_pre)
    ndvi_pre = np.where(ndvi_pre < 0.0, np.nan, ndvi_pre)

    # 2000-2009
    nyears = 10
    ndvi_dur = np.zeros((nyears,nrows,ncols))
    yr_count = 0
    for year in np.arange(2000, 2010):
        for month in np.arange(1, 13):

            if month == 12:

                (aus, aus_lat, aus_lon) = get_data(src_ds, count)
                ndvi_dur[yr_count,:,:] += aus

                (aus, aus_lat, aus_lon) = get_data(src_ds, count+1)
                ndvi_dur[yr_count,:,:] += aus

                (aus, aus_lat, aus_lon) = get_data(src_ds, count+2)
                ndvi_dur[yr_count,:,:] += aus

                ndvi_dur[yr_count,:,:] /= 3


            count += 1
        yr_count += 1

    ndvi_dur = np.nanmean(ndvi_dur, axis=0)
    ndvi_dur = np.flipud(ndvi_dur)
    ndvi_dur = np.where(ndvi_dur < 0.0, np.nan, ndvi_dur)

    chg = ((ndvi_dur - ndvi_pre) / ndvi_pre) * 100.0

    fig = plt.figure(figsize=(9, 6))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = "14"
    plt.rcParams['font.sans-serif'] = "Helvetica"

    cmap = plt.cm.get_cmap('BrBG', 10) # discrete colour map

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))
    rows = 1
    cols = 1

    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.2,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.5,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode


    for i, ax in enumerate(axgr):
        # add a subplot into the array of plots
        #ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
        plims = plot_map(ax, chg, cmap, i, top, bottom, left, right)

        import cartopy.feature as cfeature
        states = cfeature.NaturalEarthFeature(category='cultural',
                                              name='admin_1_states_provinces_lines',
                                              scale='10m',facecolor='none')

        # plot state border
        SOURCE = 'Natural Earth'
        LICENSE = 'public domain'
        ax.add_feature(states, edgecolor='black', lw=0.5)

    cbar = axgr.cbar_axes[0].colorbar(plims)
    #cbar.ax.set_title("Percentage\ndifference(%)", fontsize=16)
    cbar.ax.set_title("% Difference", fontsize=16, pad=10)
    #cbar.ax.set_yticklabels([' ', '-30', '-15', '0', '15', '<=70'])

    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax.text(0.95, 0.05, "(b)", transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    ofname = os.path.join(plot_dir, "ndvi.png")
    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)


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

    fn = "AVHRR_EVI2_SEAUS_1982_2019.tif"
    src_ds = gdal.Open(fn)
    main(src_ds)

#!/usr/bin/env python

"""
Plot VOD anomalies during the current droughts

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (10.03.2020)"
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
import sys
import matplotlib.ticker as mticker
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid

def main(plot_dir, fname):

    df = pd.read_parquet(fname, engine='pyarrow')

    dfx = df[(df.hydro_year == 2003) & (df.season == "DJF")]
    dfx = dfx.sort_values(['lat', 'lon'], ascending=[True, True])
    ncols = len(np.unique(dfx.lon))
    nrows = len(np.unique(dfx.lat))

    #vod = dfx.vod_day.values
    #vod = vod.reshape(nrows, ncols)
    #plt.imshow(vod)
    #plt.colorbar()
    #plt.show()

    # Get baseline period
    # 2002-2016
    start_yr = 2002
    end_yr = 2016
    nyears = (end_yr - start_yr) + 1
    vod_pre = np.zeros((nrows,ncols)) # summer
    vod_count = np.zeros((nrows,ncols))

    for year in np.arange(start_yr, end_yr + 1):
        print(year)

        dfx = df[(df.hydro_year == year) & (df.season == "DJF")]
        dfx = dfx.sort_values(['lat', 'lon'], ascending=[True, True])
        data = dfx.vod_day.values
        data = data.reshape(nrows, ncols)

        if year == 2002:
            bottom, top = np.min(dfx.lat), np.max(dfx.lat)
            left, right = np.min(dfx.lon), np.max(dfx.lon)
            print(top, bottom, left, right)
            lats = dfx.lat.values
            lons = dfx.lon.values
            lats.tofile("lat_vod.bin")
            lons.tofile("lon_vod.bin")
            print(lats.shape)
            print(lons.shape)
            #sys.exit()

        data = np.where(np.isnan(data), 0.0, data)
        vod_count = np.where(data > 0.0, vod_count+1, vod_count)
        vod_pre = np.where(~np.isnan(data), vod_pre+data, vod_pre)

    vod_pre = np.where(~np.isnan(vod_pre), vod_pre / vod_count, vod_pre)


    # Get baseline period
    # 2002-2018
    start_yr = 2017
    end_yr = 2019
    nyears = (end_yr - start_yr) + 1
    vod_dro = np.zeros((nrows,ncols)) # summer
    vod_count = np.zeros((nrows,ncols))

    for year in np.arange(start_yr, end_yr + 1):
        print(year)

        dfx = df[(df.hydro_year == year) & (df.season == "DJF")]
        dfx = dfx.sort_values(['lat', 'lon'], ascending=[True, True])
        data = dfx.vod_day.values
        data = data.reshape(nrows, ncols)

        data = np.where(np.isnan(data), 0.0, data)
        vod_count = np.where(data > 0.0, vod_count+1, vod_count)
        vod_dro = np.where(~np.isnan(data), vod_dro+data, vod_dro)

    vod_dro = np.where(~np.isnan(vod_dro), vod_dro / vod_count, vod_dro)

    #plt.imshow(vod_dro)
    #plt.colorbar()
    #plt.show()

    chg = ((vod_dro - vod_pre) / vod_pre) * 100.0
    print(chg.shape)
    chg.tofile("cd_change.bin")

    nrows = 245
    ncols = 294
    top = -28.159023657504804
    bottom = -38.70067756757205
    left = 140.043380666312
    right = 153.31886842464448



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
    cbar.ax.set_title("% Difference", fontsize=16, pad=10)
    #cbar.ax.set_yticklabels([' ', '-30', '-15', '0', '15', '<=70'])

    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax.text(0.95, 0.05, "(a)", transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    ofname = os.path.join(plot_dir, "vod_current.png")
    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)
    #plt.show()

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
    plot_dir = "/Users/mdekauwe/Dropbox/Drought_risk_paper/figures/figs"
    fname = "LPDRv2_VOD_YrSeason_SEAUS_2002_2019.parquet"
    main(plot_dir, fname)

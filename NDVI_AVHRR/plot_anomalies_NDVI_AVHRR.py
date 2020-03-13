#!/usr/bin/env python

"""
Plot NDVI anomalies during two droughts

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
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import sys
import matplotlib.ticker as mticker
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid


def main(fname, plot_dir):

    ds = xr.open_dataset(fname)
    bottom, top = np.min(ds.lat).values, np.max(ds.lat).values
    left, right = np.min(ds.lon).values, np.max(ds.lon).values

    # Get baseline period
    # 1993-1999
    ndvi_pre = calc_average(ds, 1993, 1999) # Match VODs

    #plt.imshow(ndvi_pre)
    #plt.colorbar()
    #plt.show()
    #sys.exit()

    # Get drought period
    # 2000-2009
    ndvi_dur = calc_average(ds, 2000, 2009)

    chg = ((ndvi_dur - ndvi_pre) / ndvi_pre) * 100.0
    chg.tofile("md_change.bin")

    ofname = os.path.join(plot_dir, "ndvi_avhrr_md.png")
    plot_anomaly(ofname, chg, bottom, top, left, right, fig_label="(c)")

    # Get baseline period
    # 1993-2016
    ndvi_pre = calc_average(ds, 1993, 2016)

    # Get drought period
    # 2017-2019
    ndvi_dur = calc_average(ds, 2017, 2019)

    chg = ((ndvi_dur - ndvi_pre) / ndvi_pre) * 100.0
    chg.tofile("cd_change.bin")

    ofname = os.path.join(plot_dir, "ndvi_avhrr_cd.png")
    plot_anomaly(ofname, chg, bottom, top, left, right, fig_label="(d)")


def calc_average(ds, start_year, end_year):

    ndates, nrows, ncols = ds.NDVI.shape
    nyears = (end_year - start_year) + 1
    ndvi = np.zeros((nyears,nrows,ncols))

    yr_count = 0
    for i in range(ndates):
        date = ds.time.values[i]
        year = int(str(ds.time.values[i]).split("-")[0])
        month = str(ds.time.values[i]).split("-")[1]

        if year >= start_year and year <= end_year and month == "01":

            ndvi[yr_count,:,:] = np.where(~np.isnan(ds.NDVI[i,:,:]), \
                                          ds.NDVI[i,:,:], ndvi[yr_count,:,:])
            yr_count += 1

    ndvi = np.nanmean(ndvi, axis=0)
    ndvi = np.where(ndvi <= 0.05, np.nan, ndvi)

    return (ndvi)


def plot_anomaly(ofname, chg, bottom, top, left, right, fig_label):

    # visually fudge geo-transform issue until fixed.
    offset = -0.2
    top = -28.00497898242608 - offset
    bottom = -39.997488025421696 - offset

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
    ax.text(0.95, 0.05, fig_label, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

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

if __name__ == "__main__":

    plot_dir = "/Users/mdekauwe/Dropbox/Drought_risk_paper/figures/figs"
    fn = "AVHRR_CDRv5_NDVI_yearSeason_mean_1982_2019.nc"
    main(fn, plot_dir)

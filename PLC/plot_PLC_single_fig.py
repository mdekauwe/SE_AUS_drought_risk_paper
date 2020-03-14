#!/usr/bin/env python
"""
Plot PLC for both droughts in a single figure.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (15.03.2020)"
__email__ = "mdekauwe@gmail.com"

import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import sys
import matplotlib.ticker as mticker
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
from calendar import monthrange
import cartopy.feature as cfeature

def calc_plc(fname, plc_type="max"):

    ds = xr.open_dataset(fname)
    lat = ds.y.values
    lon = ds.x.values
    bottom, top = lat[0], lat[-1]
    left, right = lon[0], lon[-1]
    plc = ds.plc[:,0,:,:].values
    if plc_type == "mean":
        plc = np.nanmean(plc, axis=0)
    elif plc_type == "max":
        plc = np.nanmax(plc, axis=0)
    elif plc_type == "median":
        plc = np.nanmedian(plc, axis=0)

    return (plc, bottom, top, left, right)

def plot_plc(md, cd, ofname, bottom, top, left, right):

    fig = plt.figure(figsize=(12, 6))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = "14"
    plt.rcParams['font.sans-serif'] = "Helvetica"

    cmap = plt.cm.get_cmap('YlOrRd', 10) # discrete colour map
    #cmap = plt.cm.YlOrRd

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))
    rows = 1
    cols = 2

    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.3,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.3,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode

    data = [md, cd]
    fig_labels = ["(a)", "(b)"]
    for i, ax in enumerate(axgr):
        # add a subplot into the array of plots
        #ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
        plims = plot_map(ax, data[i], cmap, i, top, bottom, left, right,
                         fig_labels[i])

        #plims = plot_map(ax, ds.plc[0,0,:,:], cmap, i)


        states = cfeature.NaturalEarthFeature(category='cultural',
                                              name='admin_1_states_provinces_lines',
                                              scale='10m',facecolor='none')

        # plot state border
        SOURCE = 'Natural Earth'
        LICENSE = 'public domain'
        ax.add_feature(states, edgecolor='black', lw=0.5)

    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("PLC (%)", fontsize=16, pad=10)


    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)

def plot_map(ax, var, cmap, i, top, bottom, left, right, fig_label):
    print(np.nanmin(var), np.nanmax(var))
    vmin, vmax = 0, 90 #88
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

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='black', alpha=0.5,
                      linestyle='--')

    if i > 0:
        gl.ylabels_left = False

    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlocator = mticker.FixedLocator([141, 145,  149, 153])
    gl.ylocator = mticker.FixedLocator([-29, -32, -35, -38])

    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax.text(0.92, 0.08, fig_label, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    return img



if __name__ == "__main__":

    plot_dir = "/Users/mdekauwe/Dropbox/Drought_risk_paper/figures/figs"

    md_fname = "../CABLE_outputs/Millennium/all_yrs_plc.nc"
    cd_fname = "../CABLE_outputs/Current/all_yrs_plc.nc"
    (plc_md, bottom, top, left, right) = calc_plc(md_fname)
    (plc_cd, bottom, top, left, right) = calc_plc(cd_fname)

    ofname = os.path.join(plot_dir, "both_plcs.png")
    plot_plc(plc_md, plc_cd, ofname, bottom, top, left, right)

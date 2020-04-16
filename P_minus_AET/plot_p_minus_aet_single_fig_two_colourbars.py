#!/usr/bin/env python
"""
Plot P-AET for both prior to and during MD in a single figure.
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

def calc_p_minus_aet(fname, start_year, end_year):

    ds = xr.open_dataset(fname)
    lat = ds.y.values
    lon = ds.x.values
    bottom, top = lat[0], lat[-1]
    left, right = lon[0], lon[-1]

    nmonths, nrows, ncols = ds.Rainf.shape
    nyears = (end_year - start_year) + 1
    aet = np.zeros((nyears,nrows,ncols))
    ppt = np.zeros((nyears,nrows,ncols))
    sec_2_day = 86400.0
    count = 0
    yr_count = 0
    mth_count = 1


    if start_year == 2017:
        # shift the start point onwards...as data starts in 2016
        for year in np.arange(2016, 2017):
            for month in np.arange(1, 13):
                #print(year, month)
                count += 1

    for year in np.arange(start_year, end_year+1):
        print(year)
        for month in np.arange(1, 13):

            days_in_month = monthrange(year, month)[1]
            conv = sec_2_day * days_in_month

            yr_val = str(ds.time[count].values).split("-")[0]
            print(yr_val, nyears)

            aet[yr_count,:,:] += ds.Evap[count,:,:] * conv
            ppt[yr_count,:,:] += ds.Rainf[count,:,:] * conv
            mth_count += 1

            if mth_count == 13:
                mth_count = 1
                yr_count += 1

            count += 1

    ppt = np.nanmean(ppt, axis=0)
    aet = np.nanmean(aet, axis=0)
    cmi = np.where(~np.isnan(aet), ppt-aet, np.nan)

    return (cmi, bottom, top, left, right)

def plot_drought(prior, during, ofname, bottom, top, left, right):

    fig = plt.figure(figsize=(12, 6))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = "14"
    plt.rcParams['font.sans-serif'] = "Helvetica"

    cmap = plt.cm.get_cmap('BrBG', 10) # discrete colour map

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))
    rows = 1
    cols = 2

    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.6,
                    cbar_location='right',
                    cbar_mode='each',
                    cbar_pad=0.1,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode

    data = [prior, during]
    fig_labels = ["(a)", "(b)"]
    vmaxs = [50, 200]
    vmins = [-50, -200]
    for i, ax in enumerate(axgr):
        # add a subplot into the array of plots
        #ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
        plims = plot_map(ax, data[i], cmap, i, top, bottom, left, right,
                         fig_labels[i], vmins[i], vmaxs[i])

        #plims = plot_map(ax, ds.plc[0,0,:,:], cmap, i)


        states = cfeature.NaturalEarthFeature(category='cultural',
                                              name='admin_1_states_provinces_lines',
                                              scale='10m',facecolor='none')

        # plot state border
        SOURCE = 'Natural Earth'
        LICENSE = 'public domain'
        ax.add_feature(states, edgecolor='black', lw=0.5)

        cbar = axgr.cbar_axes[i].colorbar(plims)
        if i == 1:
            cbar.ax.set_title("P-AET\n(mm yr$^{-1}$)", fontsize=16, pad=12)

    #cbar.ax.set_yticklabels([' ', '$\minus$40', '$\minus$20', '0', '20', '40-1300'])

    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)

def plot_map(ax, var, cmap, i, top, bottom, left, right, fig_label, vmin, vmax):
    print(np.nanmin(var), np.nanmax(var))

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

    #md_fname = "../CABLE_outputs/Millennium/all_yrs_CMI.nc"
    #cd_fname = "../CABLE_outputs/Current/all_yrs_CMI.nc"
    md_fname = "/Users/mdekauwe/research/drought_desktop/Drought/outputs/all_yrs_CMI.nc"
    cd_fname = "/Users/mdekauwe/research/drought_desktop/current/outputs/all_yrs_CMI.nc"
    (cmi_md, bottom, top, left, right) = calc_p_minus_aet(md_fname, 2000, 2009)
    (cmi_cd, bottom, top, left, right) = calc_p_minus_aet(cd_fname, 2017, 2019)

    ofname = os.path.join(plot_dir, "p_minus_aet.png")
    plot_drought(cmi_md, cmi_cd, ofname, bottom, top, left, right)

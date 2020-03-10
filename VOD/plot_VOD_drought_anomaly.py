#!/usr/bin/env python
"""
Plot VOD anomaly during Millennium drought

VOD (1993_2012) data from:
/srv/ccrc/data04/z3509830/LAI_precip_variability/Data/Vegetation_indices/VOD

"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (29.09.2019)"
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

def main(fname, plot_dir):

    ds = xr.open_dataset(fname, decode_times=False)
    bottom, top = np.min(ds.latitude).values, np.max(ds.latitude).values
    left, right =np.min(ds.longitude).values, np.max(ds.longitude).values

    ntime, nrows, ncols = ds.VOD.shape

    # Get baseline period
    # 1993-1999
    nyears = 7
    vod_pre = np.zeros((nyears,nrows,ncols))
    vals = np.zeros((3,nrows,ncols))
    yr_count = 0
    count = 0
    for year in np.arange(1993, 2000):
        for month in np.arange(1, 13):

            if month == 12:

                vod_count = np.zeros((nrows,ncols))

                vals = ds.VOD[count,:,:].values
                vals = np.where(np.isnan(vals), 0.0, vals)
                vod_count = np.where(vals > 0.0, vod_count+1, vod_count)
                vod_pre[yr_count,:,:] += vals

                vals = ds.VOD[count+1,:,:].values
                vals = np.where(np.isnan(vals), 0.0, vals)
                vod_count = np.where(vals > 0.0, vod_count+1, vod_count)
                vod_pre[yr_count,:,:] += vals

                vals = ds.VOD[count+2,:,:].values
                vals = np.where(np.isnan(vals), 0.0, vals)
                vod_count = np.where(vals > 0.0, vod_count+1, vod_count)
                vod_pre[yr_count,:,:] += vals

                vod_pre[yr_count,:,:] /= vod_count

            count += 1
        yr_count += 1

    vod_pre = np.mean(vod_pre, axis=0)
    vod_pre = np.flipud(vod_pre)

    # We will have incremented the counter by one too many on the final
    # iteration, fix this as we need to start at the right point for 2000
    count = count - 1

    # 2000-2009
    nyears = 10
    vod_dur = np.zeros((nyears,nrows,ncols))
    yr_count = 0
    for year in np.arange(2000, 2010):
        for month in np.arange(1, 13):

            if month == 12:

                vod_count = np.zeros((nrows,ncols))

                vals = ds.VOD[count,:,:].values
                vals = np.where(np.isnan(vals), 0.0, vals)
                vod_count = np.where(vals > 0.0, vod_count+1, vod_count)
                vod_dur[yr_count,:,:] += vals

                vals = ds.VOD[count+1,:,:].values
                vals = np.where(np.isnan(vals), 0.0, vals)
                vod_count = np.where(vals > 0.0, vod_count+1, vod_count)
                vod_dur[yr_count,:,:] += vals

                vals = ds.VOD[count+2,:,:].values
                vals = np.where(np.isnan(vals), 0.0, vals)
                vod_count = np.where(vals > 0.0, vod_count+1, vod_count)
                vod_dur[yr_count,:,:] += vals

                vod_dur[yr_count,:,:] /= vod_count


            count += 1
        yr_count += 1

    vod_dur = np.mean(vod_dur, axis=0)
    vod_dur = np.flipud(vod_dur)

    chg = ((vod_dur - vod_pre) / vod_pre) * 100.0

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

    ofname = os.path.join(plot_dir, "vod.png")
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

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    #fname = "raw/Australia_VOD_monthly_1993_2012_masked_gapfilled.nc"
    fname = "raw/Australia_VOD_monthly_1993_2012_non-masked_gapfilled_no_missing.nc"
    main(fname, plot_dir)

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

    ds = xr.open_dataset(fname)
    bottom, top = np.min(ds.latitude).values, np.max(ds.latitude).values
    left, right =np.min(ds.longitude).values, np.max(ds.longitude).values

    ntime, nrows, ncols = ds.ndvi.shape


    """
    st_count = 0
    for year in np.arange(1981, 2000):
        for month in np.arange(1, 13):

            if year > 1989 and year < 2000:
                break
            print(year, month, ds.time[count].values)
            st_count += 1


    print(st_count)
    print(ds.time[count].values)

    sys.exit()
    """
    st_count = 108 # 1990

    # Get baseline period
    # 1993-1999
    nyears = 10
    ndvi_pre = np.zeros((nyears,nrows,ncols))
    vals = np.zeros((3,nrows,ncols)) # summer
    yr_count = 0
    count = st_count
    for year in np.arange(1990, 2000):
        for month in np.arange(1, 13):
            if month == 12:
                #vals = ds.VOD[count,:,:].values
                #vals = np.where(np.isnan(vals), 0, vals)
                ndvi_pre[yr_count,:,:] += ds.ndvi[count,:,:]
                ndvi_pre[yr_count,:,:] += ds.ndvi[count+1,:,:]
                ndvi_pre[yr_count,:,:] += ds.ndvi[count+2,:,:]
                ndvi_pre[yr_count,:,:] /= 3

            count += 1
        yr_count += 1

    ndvi_pre = np.mean(ndvi_pre, axis=0)
    ndvi_pre = np.flipud(ndvi_pre)
    ndvi_pre = np.where(ndvi_pre < 0.0, np.nan, ndvi_pre)


    # We will have incremented the counter by one too many on the final
    # iteration, fix this as we need to start at the right point for 2000
    count = count - 1

    # 2000-2009
    nyears = 10
    ndvi_dur = np.zeros((nyears,nrows,ncols))
    yr_count = 0
    for year in np.arange(2000, 2010):
        for month in np.arange(1, 13):

            if month == 12:
                ndvi_dur[yr_count,:,:] += ds.ndvi[count,:,:]
                ndvi_dur[yr_count,:,:] += ds.ndvi[count+1,:,:]
                ndvi_dur[yr_count,:,:] += ds.ndvi[count+2,:,:]
                ndvi_dur[yr_count,:,:] /= 3

            count += 1
        yr_count += 1

    ndvi_dur = np.mean(ndvi_dur, axis=0)
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


    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("Percentage\ndifference(%)", fontsize=16)
    #cbar.ax.set_yticklabels([' ', '-30', '-15', '0', '15', '<=70'])

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

    ax.set_xlim(140, 154)
    ax.set_ylim(-39.4, -28)

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
    fname = "ndvi3g_geo_v1_1_1981to2017_ndviMonMax_SE_AUS.nc"
    main(fname, plot_dir)

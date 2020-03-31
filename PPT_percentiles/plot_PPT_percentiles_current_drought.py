#!/usr/bin/env python
"""
Plot rainfall percentiles during the Millennium drought
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (26.11.2019)"
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


    lat = ds.latitude.values
    lon = ds.longitude.values
    bottom, top = lat[0], lat[-1]
    left, right = lon[0], lon[-1]

    fig = plt.figure(figsize=(20, 8))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = "14"
    plt.rcParams['font.sans-serif'] = "Helvetica"

    cmap = plt.cm.RdYlBu
    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    rows = 1
    cols = 3

    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.2,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.5,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode

    ppt_count = 0

    for year in np.arange(1900, 2016): # 1970-2019

        ppt_count += 1



    year = 2016
    for i, ax in enumerate(axgr):
        # add a subplot into the array of plots
        #ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
        plims = plot_map(ax, ds.pr[ppt_count+i,:,:], year, cmap, i, top, bottom,
                         left, right)

        import cartopy.feature as cfeature
        states = cfeature.NaturalEarthFeature(category='cultural',
                                              name='admin_1_states_provinces_lines',
                                              scale='10m',facecolor='none')

        # plot state border
        SOURCE = 'Natural Earth'
        LICENSE = 'public domain'
        ax.add_feature(states, edgecolor='black', lw=0.5)

        year += 1

    #bounds = np.linspace(0, 100, 9)
    #bounds = np.append(bounds, bounds[-1]+1)
    #norm = colors.BoundaryNorm(bounds, cmap.N)
    #cbar = axgr.cbar_axes[0].colorbar(plims, norm=norm, boundaries=bounds, ticks=bounds)
    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("Percentile", fontsize=16, pad=10)

    #for i, cax in enumerate(axgr.cbar_axes):
    #    cax.set_yticks([0.5, 5, 15, 25, 50, 75, 85, 94.5, 99.5])
    #    cax.set_yticklabels(["0-1", "1-10", "10-20", "20-30", "30-70", "70-80", \
    #                         "80-90", "90-99", "99-100"])

    ofname = os.path.join(plot_dir, "AWAP_PPT_percentiles_2016_2018.png")
    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)
    plt.show()

def plot_map(ax, var, year, cmap, i, top, bottom, left, right):
    vmin, vmax = 0.0, 100.0
    #top, bottom = 90, -90
    #left, right = -180, 180
    #top, bottom = -10, -44
    #left, right = 112, 154
    #print(np.nanmin(var), np.nanmax(var))

    #bounds = [0.5, 5, 15, 25, 75, 85, 94.5, 99.5]
    bounds = np.linspace(0, 100+10, 10)
    #bounds = np.append(bounds, bounds[-1]+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    img = ax.imshow(var * 100., origin='lower',
                    transform=ccrs.PlateCarree(),
                    interpolation='nearest', cmap=cmap, norm=norm,
                    extent=(left, right, bottom, top),
                    vmin=vmin, vmax=vmax)

    ax.coastlines(resolution='10m', linewidth=1.0, color='black')
    #ax.add_feature(cartopy.feature.OCEAN)
    ax.set_title("%d$-$%d" % (year, year+1), fontsize=16)
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
    gl.xlabels_bottom = True
    if i > 1:
        gl.ylabels_left = False

    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlocator = mticker.FixedLocator([141, 145,  149, 153])
    gl.ylocator = mticker.FixedLocator([-29, -32, -35, -38])

    #if i == 0 :
    #    ax.text(-0.2, 0.5, 'Latitude', va='bottom', ha='center',
    #            rotation='vertical', rotation_mode='anchor',
    #            transform=ax.transAxes, fontsize=16)
    #if i == 1:
    #    ax.text(0.5, -0.2, 'Longitude', va='bottom', ha='center',
    #            rotation='horizontal', rotation_mode='anchor',
    #            transform=ax.transAxes, fontsize=16)

    return img


if __name__ == "__main__":

    plot_dir = "/Users/mdekauwe/Dropbox/Drought_risk_paper/figures/figs"

    fname = "AWAP_annual_rainfall_percentiles_1900_2018_jul_jun_years.nc"

    main(fname, plot_dir)

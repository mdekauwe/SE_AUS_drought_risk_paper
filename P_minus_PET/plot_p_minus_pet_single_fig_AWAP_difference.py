#!/usr/bin/env python
"""
Plot P-PET for both prior to and during MD in a single figure.
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

def calc_p_minus_pet(pet_fname, ppt_fname, landmask_fname, start_year, end_year):

    ds_pet = xr.open_dataset(pet_fname, decode_times=False)
    ds_ppt = xr.open_dataset(ppt_fname, decode_times=False)
    ds_land = xr.open_dataset(landmask_fname, decode_times=False)

    land = ds_land.landsea.values

    bottom, top = np.min(ds_ppt.latitude).values, np.max(ds_ppt.latitude).values
    left, right = (np.min(ds_ppt.longitude).values,
                   np.max(ds_ppt.longitude).values)

    ntime, nrows, ncols = ds_ppt.rain.shape

    nyears = (end_year - start_year) + 1
    pet = np.zeros((nyears,nrows,ncols))
    ppt = np.zeros((nyears,nrows,ncols))
    mth_count = 1
    yr_count = 0
    pet_count = 0
    ppt_count = 0

    # PET and PPT have different starting years, account for offsets...
    if start_year == 1990:

        for year in np.arange(1970, 1990): # 1970-2019
            for month in np.arange(1, 13):
                #print(year, month)
                ppt_count += 1


    elif start_year == 2000:
        # shift the start point onwards...
        for year in np.arange(1990, 2000): # 1990-2019
            for month in np.arange(1, 13):
                pet_count += 1

        for year in np.arange(1970, 2000): # 1970-2019
            for month in np.arange(1, 13):
                #print(year, month)
                ppt_count += 1

    for year in np.arange(start_year, end_year+1):
        print(year)
        for month in np.arange(1, 13):

            pet[yr_count,:,:] += ds_pet.PET[pet_count,:,:]
            ppt[yr_count,:,:] += ds_ppt.rain[ppt_count,:,:]
            mth_count += 1

            if mth_count == 13:
                mth_count = 1
                yr_count += 1

            pet_count += 1
            ppt_count += 1

    ppt = np.mean(ppt, axis=0)
    pet = np.mean(pet, axis=0)
    cmi = ppt - pet
    cmi = np.flipud(cmi)

    # land
    cmi = np.where(land  == 0, cmi, np.nan)

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
                    axes_pad=0.8,
                    cbar_location='right',
                    cbar_mode='each',
                    cbar_pad=0.1,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode


    data = [prior, during-prior]
    fig_labels = ["(a)", "(b)"]

    for i, ax in enumerate(axgr):
        # add a subplot into the array of plots
        #ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())

        if i == 0:
            plims = plot_map(ax, data[i], cmap, i, top, bottom, left, right,
                             fig_labels[i], num=i)
        else:
            plimsx = plot_map(ax, data[i], cmap, i, top, bottom, left, right,
                             fig_labels[i], num=i)
        #plims = plot_map(ax, ds.plc[0,0,:,:], cmap, i)


        states = cfeature.NaturalEarthFeature(category='cultural',
                                              name='admin_1_states_provinces_lines',
                                              scale='10m',facecolor='none')

        # plot state border
        SOURCE = 'Natural Earth'
        LICENSE = 'public domain'
        ax.add_feature(states, edgecolor='black', lw=0.5)

    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("P-PET\n(mm yr$^{-1}$)", fontsize=16, pad=12)

    cbar = axgr.cbar_axes[1].colorbar(plimsx)
    cbar.ax.set_title("Difference\n(mm yr$^{-1}$)", fontsize=16, pad=12)

    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)

    plt.show()

def plot_map(ax, var, cmap, i, top, bottom, left, right, fig_label, num=0):
    print(np.nanmin(var), np.nanmax(var))
    if num == 0:
        vmin, vmax = -1600, 1600
    else:
        vmin, vmax = -400, 400
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

    pet_fname = "AWAP_monthly_PriestleyTaylor_PET_1990_2019.nc"
    ppt_fname = "AWAP_monthly_rain_1970_2019.nc"
    landmask_fname = "gridinfo_AWAP_CSIRO_AU_NAT_new_iveg_mask.nc"
    (cmi_prior_md, bottom,
     top, left, right) = calc_p_minus_pet(pet_fname, ppt_fname, landmask_fname,
                                          1990, 1999)
    (cmi_during_md, bottom,
     top, left, right) = calc_p_minus_pet(pet_fname, ppt_fname, landmask_fname,
                                          2000, 2009)

    ofname = os.path.join(plot_dir, "p_minus_pet_md.png")
    plot_drought(cmi_prior_md, cmi_during_md, ofname, bottom, top,
                 left, right)

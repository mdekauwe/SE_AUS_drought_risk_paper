#!/usr/bin/env python
"""
Plot DJF for each year of the Millennium drought
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (25.07.2019)"
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

def main(pet_fname, ppt_fname, plot_dir):

    ds_pet = xr.open_dataset(pet_fname, decode_times=False)
    ds_ppt = xr.open_dataset(ppt_fname, decode_times=False)

    bottom, top = np.min(ds_ppt.latitude).values, np.max(ds_ppt.latitude).values
    left, right =np.min(ds_ppt.longitude).values, np.max(ds_ppt.longitude).values

    ntime, nrows, ncols = ds_ppt.precip.shape

    nyears = 10
    pet = np.zeros((nyears,nrows,ncols))
    ppt = np.zeros((nyears,nrows,ncols))
    mth_count = 1
    yr_count = 0

    # shift the start point onwards...
    count = 0
    for year in np.arange(1990, 2000):
        for month in np.arange(1, 13):
            #print(year, month)
            count += 1

    for year in np.arange(2000, 2010):
        #print(year)
        for month in np.arange(1, 13):

            if year == 2000 and month >= 7:

                pet[yr_count,:,:] += ds_pet.PET[count,:,:]
                ppt[yr_count,:,:] += ds_ppt.precip[count,:,:]

                mth_count += 1

            elif year > 2000 and year <= 2009:

                pet[yr_count,:,:] += ds_pet.PET[count,:,:]
                ppt[yr_count,:,:] += ds_ppt.precip[count,:,:]
                mth_count += 1

            elif year == 2009 and month <= 6:

                pet[yr_count,:,:] += ds_pet.PET[count,:,:]
                ppt[yr_count,:,:] += ds_ppt.precip[count,:,:]
                mth_count += 1



            if mth_count == 13:
                mth_count = 1
                yr_count += 1


            count += 1

    ppt = np.mean(ds_ppt.precip[0:count,:,:], axis=0)
    pet = np.mean(ds_pet.PET[0:count,:,:], axis=0)
    cmi = ppt - pet

    # just keep deficit areas
    #cmi = np.where(cmi >= 300., np.nan, cmi)

    cmi = np.flipud(cmi)
    #plt.imshow(cmi)
    #plt.colorbar()
    #plt.show()
    #sys.exit()
    fig = plt.figure(figsize=(9, 6))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = "14"
    plt.rcParams['font.sans-serif'] = "Helvetica"

    cmap = plt.cm.get_cmap('BrBG', 8) # discrete colour map

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
        plims = plot_map(ax, cmi, cmap, i, top, bottom, left, right)
        #plims = plot_map(ax, ds.plc[0,0,:,:], cmap, i)

        import cartopy.feature as cfeature
        states = cfeature.NaturalEarthFeature(category='cultural',
                                              name='admin_1_states_provinces_lines',
                                              scale='10m',facecolor='none')

        # plot state border
        SOURCE = 'Natural Earth'
        LICENSE = 'public domain'
        ax.add_feature(states, edgecolor='black', lw=0.5)

    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("P-PET\n(mm yr$^{-1}$)", fontsize=16, pad=12)

    ofname = os.path.join(plot_dir, "p_minus_pet_during.png")
    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)

def plot_map(ax, var, cmap, i, top, bottom, left, right):
    print(np.nanmin(var), np.nanmax(var))
    vmin, vmax = -160, 160
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

    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax.text(0.95, 0.05, "(b)", transform=ax.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    return img


if __name__ == "__main__":

    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    pet_fname = "/Users/mdekauwe/research/SE_AUS_drought_risk_paper/P_minus_PET/ANUCLIM_PriestleyTaylor_PET_monthly_mean_1990_2010_NDVI.nc"
    ppt_fname = "/Users/mdekauwe/research/SE_AUS_drought_risk_paper/P_minus_PET/ANUCLIM_precip_monthly_1990_2010_NDVI_res.nc"
    main(pet_fname, ppt_fname, plot_dir)

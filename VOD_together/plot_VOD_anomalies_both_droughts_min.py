#!/usr/bin/env python

"""
Plot VOD anomalies during the current droughts - here we are plotting the worst
anomaly

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
import xarray as xr

import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import sys
import matplotlib.ticker as mticker
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid


def get_change_md(fname):

    ds = xr.open_dataset(fname, decode_times=False)
    bottom, top = np.min(ds.latitude).values, np.max(ds.latitude).values
    left, right =np.min(ds.longitude).values, np.max(ds.longitude).values

    ntime, nrows, ncols = ds.VOD.shape

    # Get baseline period
    # 1993-1999
    start_year = 1993
    end_year = 1999
    nyears = (end_year - start_year) + 1
    vod_pre = np.zeros((nyears,nrows,ncols))
    vals = np.zeros((3,nrows,ncols))
    yr_count = 0
    count = 0
    for year in np.arange(start_year, end_year + 1):
        for month in np.arange(1, 13):

            if month == 12:

                if year == 1993:

                    lats = ds.latitude.values
                    lons = ds.longitude.values
                    #lats.tofile("lat_vod.bin")
                    #lons.tofile("lon_vod.bin")

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
    start_year = 2000
    end_year = 2009
    nyears = (end_year - start_year) + 1
    vod_dur = np.ones((nrows,ncols)) * np.nan



    yr_count = 0

    vals = ds.VOD[count,:,:].values
    vod_dur[:,:] = vals

    for year in np.arange(start_year, end_year+1):
        for month in np.arange(1, 13):

            if month == 12:

                vod_count = np.zeros((nrows,ncols))

                vals = ds.VOD[count,:,:].values
                #vals = np.where(np.isnan(vals), 0.0, vals)
                vod_dur[:,:] = np.where(np.logical_and(~np.isnan(vals),
                                        vals < vod_dur), vals, vod_dur)

                vals = ds.VOD[count+1,:,:].values
                #vals = np.where(np.isnan(vals), 0.0, vals)
                vod_dur[:,:] = np.where(np.logical_and(~np.isnan(vals),
                                        vals < vod_dur), vals, vod_dur)

                vals = ds.VOD[count+2,:,:].values
                #vals = np.where(np.isnan(vals), 0.0, vals)
                vod_dur[:,:] = np.where(np.logical_and(~np.isnan(vals),
                                        vals < vod_dur), vals, vod_dur)


            count += 1
        yr_count += 1




    vod_dur = np.flipud(vod_dur)





    diff = np.where(np.absolute(vod_dur - vod_pre < 0.01),
                    vod_dur - vod_pre, 0.0)
    diff = np.where(diff == 0.0, np.nan, diff)
    #plt.imshow(diff )
    #plt.colorbar()
    #plt.show()
    #sys.exit()
    chg = (diff / vod_pre) * 100.0

    #chg = ((vod_dur - vod_pre) / vod_pre) * 100.0
    print(chg.shape)
    #chg.tofile("md_change.bin")

    return (chg, top, bottom, left, right)

def get_change_current(fname):

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
    # 2010-2016
    start_yr = 2010
    end_yr = 2016
    vod_pre = np.zeros((nrows,ncols)) # summer
    vod_count = np.zeros((nrows,ncols))

    for year in np.arange(start_yr, end_yr + 1):
        print(year)

        dfx = df[(df.hydro_year == year) & (df.season == "DJF")]
        dfx = dfx.sort_values(['lat', 'lon'], ascending=[True, True])
        data = dfx.vod_day.values
        data = data.reshape(nrows, ncols)

        if year == 2010:
            bottom, top = np.min(dfx.lat), np.max(dfx.lat)
            left, right = np.min(dfx.lon), np.max(dfx.lon)
            print(top, bottom, left, right)
            lats = dfx.lat.values
            lons = dfx.lon.values
            #lats.tofile("lat_vod.bin")
            #lons.tofile("lon_vod.bin")
            print(lats.shape)
            print(lons.shape)
            #sys.exit()

        data = np.where(np.isnan(data), 0.0, data)
        vod_count = np.where(data > 0.0, vod_count+1, vod_count)
        vod_pre = np.where(~np.isnan(data), vod_pre+data, vod_pre)

    vod_pre = np.where(~np.isnan(vod_pre), vod_pre / vod_count, vod_pre)


    # Get Drought
    # 2017-2018
    start_yr = 2017
    end_yr = 2018
    vod_dro = np.ones((nrows,ncols)) * np.nan

    dfx = df[(df.hydro_year == 2017) & (df.season == "DJF")]
    dfx = dfx.sort_values(['lat', 'lon'], ascending=[True, True])
    data = dfx.vod_day.values
    data = data.reshape(nrows, ncols)
    vod_dro = np.where(~np.isnan(data), data, vod_dro)

    for year in np.arange(start_yr, end_yr + 1):
        print(year)


        dfx = df[(df.hydro_year == year) & (df.season == "DJF")]
        dfx = dfx.sort_values(['lat', 'lon'], ascending=[True, True])
        data = dfx.vod_day.values
        data = data.reshape(nrows, ncols)

        #data = np.where(np.isnan(data), 0.0, data)

        vod_dro = np.where(np.logical_and(~np.isnan(data), data < vod_dro),
                            data, vod_dro)


    #plt.imshow(vod_dro)
    #plt.colorbar()
    #plt.show()

    diff = np.where(np.absolute(vod_dro - vod_pre < 0.01),
                    vod_dro - vod_pre, 0.0)
    diff = np.where(diff == 0.0, np.nan, diff)

    chg = (diff / vod_pre) * 100.0

    #print(chg.shape)
    #chg.tofile("cd_change.bin")

    #nrows = 245
    #ncols = 294
    #top = -28.159023657504804
    #bottom = -38.70067756757205
    #left = 140.043380666312
    #right = 153.31886842464448
    return (chg, top, bottom, left, right)



def plot_droughts(data, ofname, top, bottom, left, right):

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
                    axes_pad=0.3,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.1,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode


    fig_labels = ["(a)", "(b)"]
    for i, ax in enumerate(axgr):
        # add a subplot into the array of plots
        #ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
        plims = plot_map(ax, data[i], cmap, i, top[i], bottom[i], left[i],
                         right[i], fig_labels[i])

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


    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)
    #plt.show()

def plot_map(ax, var, cmap, i, top, bottom, left, right, fig_label):
    print(np.nanmin(var), np.nanmax(var))
    vmin, vmax = -50., 50.
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
    if i > 0:
        gl.ylabels_left = False
    #gl.xlabels_bottom = False
    gl.xlabels_bottom = True

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

    fname = "../VOD/Australia_VOD_monthly_1993_2012_non-masked_gapfilled_no_missing.nc"
    (chg_md, top_md, bottom_md, left_md, right_md) = get_change_md(fname)

    fname = "../VOD_LPDR/LPDRv2_VOD_YrSeason_SEAUS_2002_2019.parquet"
    (chg_cd, top_cd, bottom_cd, left_cd, right_cd) = get_change_current(fname)

    data = [chg_md, chg_cd]
    top = [top_md, top_cd]
    bottom = [bottom_md, bottom_cd]
    left = [left_md, left_cd]
    right = [right_md, right_cd]

    ofname = os.path.join(plot_dir, "vod_md_and_current_min.png")
    plot_droughts(data, ofname, top, bottom, left, right)

#!/usr/bin/env python
"""
Plot PDSI (2.5 deg) for SE AUS using data from
http://www.cgd.ucar.edu/cas/catalog/climind/pdsi.html

The Palmer Drought Severity Index (PDSI) uses readily available temperature and
precipitation data to estimate relative dryness. It is a standardized index that
generally spans -10 (dry) to +10 (wet). Maps of operational agencies like NOAA
typically show a range of -4 to +4, but more extreme values are possible. The
PDSI has been reasonably successful at quantifying long-term drought. As it uses
temperature data and a physical water balance model, it can capture the basic
effect of global warming on drought through changes in potential
evapotranspiration.

Dai, A. 2017. Dai Global Palmer Drought Severity Index (PDSI). Research Data
Archive at the National Center for Atmospheric Research, Computational and
Information Systems Laboratory. https://doi.org/10.5065/D6QF8R93.
Accessed† dd mmm yyyy.

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (07.04.2016)"
__email__ = "mdekauwe@gmail.com"

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime as dt
import calendar
import pandas as pd
import brewer2mpl
from brewer2mpl import diverging
import brewer2mpl
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import AxesGrid
import glob
import os
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
sns.set(style="white")

def colorbar_index(cax=None, ncolours=None, cmap=None, orientation=None):
    cmap = cmap_discretize(cmap, ncolours)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolours+0.5)
    colorbar = plt.colorbar(mappable, cax=cax, orientation=orientation)
    colorbar.set_ticks(np.linspace(0, ncolours, ncolours))

    return colorbar

def cmap_discretize(cmap, N):
    """
    Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)

    """
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                      for i in range(N + 1)]
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


"""
fname = "pdsi.monthly.maps.1870-2005.fawc=1.r2.5x2.5.nc"
f = nc.Dataset(fname, 'r')
lats = f.variables['lat'][:]
lons = f.variables['lon'][:]
time = f.variables['time'][:]
# 2000-2005
pdsi = f.variables['PDSI'][:,:,:]
"""

fname = "pdsisc.monthly.maps.1850-2014.fawc=1.r2.5x2.5.ipe=2.nc"
f = nc.Dataset(fname, 'r')
lats = f.variables['lat'][:]
lons = f.variables['lon'][:]
time = f.variables['time'][:]
# 1850-2012; # 2001-2010 1812-1931
pdsi = f.variables['sc_PDSI_pm'][:,:,:]
#for i in range(len(time)):
#    print(i, time[i])
#sys.exit()


plt.rcParams['axes.labelsize'] = 14
plt.rcParams['font.sans-serif'] = "Helvetica"

fig = plt.figure(figsize=(14, 10))
grid = AxesGrid(fig, [0.05,0.05,0.9,0.9], nrows_ncols=(10,12), axes_pad=0.1,
                cbar_mode='single', cbar_pad=0.2, cbar_size="3%",
                cbar_location='bottom', share_all=True)

m = Basemap(projection='cyl', llcrnrlon=lons[0], llcrnrlat=lats[0], \
            urcrnrlon=lons[-1], urcrnrlat=lats[-1], resolution='c')

# Range on colourbar
ncolours = 11
vmin = -5.0
vmax = 5.0


for cnt in range(10*12):


    # hex colours from here http://colorbrewer2.org/
    # we could just use PRGn, but it didn't seem to use white as the zero colour, it
    # was more of a grey, this fixes that
    #bmap = sns.blend_palette(["indigo", "white", "darkgreen"], ncolours, as_cmap=True)
    bmap = sns.blend_palette(["#762a83", "white", "#1b7837"], ncolours, as_cmap=True)

    ax = grid[cnt]
    m.ax = ax
    m.drawcoastlines(linewidth=0.5, color='k')
    m.drawcountries(linewidth=0.5, color='k')
    image = m.imshow(pdsi[1806+cnt,:,:], bmap,
                     colors.Normalize(vmin=vmin, vmax=vmax, clip=True),
                     interpolation='nearest')
    cbar = colorbar_index(cax=grid.cbar_axes[0], ncolours=ncolours, cmap=bmap,
                          orientation='horizontal')
    cbar.set_ticklabels(np.linspace(vmin, vmax, ncolours))
    cbar.set_label("Palmer Drought Severity Index (-)", fontsize=16)

    ax.set_xlim(140, 154)
    ax.set_ylim(-39, -28)

    if cnt == 0:
        yr = "2000-1"
    elif cnt == 12:
        yr = "2001-2"
    elif cnt == 24:
        yr = "2002-3"
    elif cnt == 36:
        yr = "2003-4"
    elif cnt == 48:
        yr = "2004-5"
    elif cnt == 60:
        yr = "2005-6"
    elif cnt == 72:
        yr = "2006-7"
    elif cnt == 84:
        yr = "2007-8"
    elif cnt == 96:
        yr = "2008-9"
    elif cnt == 108:
        yr = "2009-10"


    if cnt == 0 or cnt == 12 or cnt == 24 or cnt == 36 or cnt == 48 or\
       cnt == 60 or cnt == 72 or cnt == 84 or cnt == 96 or cnt == 108:
        textstr='%s' % (yr)
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(-0.75, 0.6, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)

    if cnt == 0:
        textstr='Jul'
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    elif cnt == 1:
        textstr='Aug'
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    elif cnt == 2:
        textstr='Sep'
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    elif cnt == 3:
        textstr='Oct'
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    elif cnt == 4:
        textstr='Nov'
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    elif cnt == 5:
        textstr='Dec'
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    elif cnt == 6:
        textstr='Jan'
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    elif cnt == 7:
        textstr='Feb'
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    elif cnt == 8:
        textstr='Mar'
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    elif cnt == 9:
        textstr='Apr'
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    elif cnt == 10:
        textstr='May'
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
    elif cnt == 11:
        textstr='Jun'
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, ec="white")
        ax.text(0.3, 1.3, textstr, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)


fig.savefig("PDSI_SE_AUS.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()

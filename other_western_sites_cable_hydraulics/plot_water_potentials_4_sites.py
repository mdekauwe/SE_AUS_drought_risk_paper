#!/usr/bin/env python

"""
Plot SWP

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (18.10.2017)"
__email__ = "mdekauwe@gmail.com"

import netCDF4 as nc
import matplotlib.pyplot as plt
import sys
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator
import datetime
import os
import glob
import string

def main(fname1, fname2, fname3, fname4, met_fname1, met_fname2, met_fname3,
         met_fname4, odir, labelsx):
    SEC_2_HLFHOUR = 1800.

    df1 = read_cable_file(fname1, type="CABLE")
    df2 = read_cable_file(fname2, type="CABLE")
    df3 = read_cable_file(fname3, type="CABLE")
    df4 = read_cable_file(fname4, type="CABLE")

    df1_met = read_cable_file(met_fname1, type="MET")
    df2_met = read_cable_file(met_fname2, type="MET")
    df3_met = read_cable_file(met_fname3, type="MET")
    df4_met = read_cable_file(met_fname4, type="MET")

    df1_met['Rainf'] *= SEC_2_HLFHOUR
    df2_met['Rainf'] *= SEC_2_HLFHOUR
    df3_met['Rainf'] *= SEC_2_HLFHOUR
    df4_met['Rainf'] *= SEC_2_HLFHOUR
    method = {'Rainf':'sum'}
    df1_met = df1_met.resample("D").agg(method)
    df2_met = df2_met.resample("D").agg(method)
    df3_met = df3_met.resample("D").agg(method)
    df4_met = df4_met.resample("D").agg(method)

    fig = plt.figure(figsize=(9,12))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    colours = plt.cm.Set2(np.linspace(0, 1, 7))

    labels_gen = label_generator('lower', start="(", end=")")
    props = dict(boxstyle='round', facecolor='white', alpha=1.0,
                 ec="white")

    ax1 = fig.add_subplot(4,1,1)
    ax2 = fig.add_subplot(4,1,2)
    ax3 = fig.add_subplot(4,1,3)
    ax4 = fig.add_subplot(4,1,4)

    axx1 = ax1.twinx()
    axx2 = ax2.twinx()
    axx3 = ax3.twinx()
    axx4 = ax4.twinx()

    df1x = df1[(df1.index.hour == 6)].copy()
    df1y = df1[(df1.index.hour == 12)].copy()

    df2x = df2[(df2.index.hour == 6)].copy()
    df2y = df2[(df2.index.hour == 12)].copy()

    df3x = df3[(df3.index.hour == 6)].copy()
    df3y = df3[(df3.index.hour == 12)].copy()

    df4x = df4[(df4.index.hour == 6)].copy()
    df4y = df4[(df4.index.hour == 12)].copy()

    axes = [ax1,ax2,ax3, ax4]
    #axes2 = [axx1, ax]


    ax1.plot(df1.index, df1["weighted_psi_soil"], c=colours[0],
             lw=1.5, ls="-", label="$\Psi$$_{sw}$", zorder=2)
    ax1.plot(df1x.index, df1x["psi_stem"], c=colours[1],
             lw=1.5, ls="-", label="$\Psi$$_{x}$", zorder=1)
    ax1.plot(df1y.index, df1y["psi_leaf"], c=colours[2],
             lw=1.5, ls="-", label="$\Psi$$_{l}$")


    ax2.plot(df2.index, df2["weighted_psi_soil"], c=colours[0],
             lw=1.5, ls="-", label="$\Psi$$_{sw}$", zorder=2)
    ax2.plot(df2x.index, df2x["psi_stem"], c=colours[1],
             lw=1.5, ls="-", label="$\Psi$$_{x}$", zorder=1)
    ax2.plot(df2y.index, df2y["psi_leaf"], c=colours[2],
             lw=1.5, ls="-", label="$\Psi$$_{l}$")


    ax3.plot(df3.index, df3["weighted_psi_soil"], c=colours[0],
             lw=1.5, ls="-", label="$\Psi$$_{sw}$", zorder=2)
    ax3.plot(df3x.index, df3x["psi_stem"], c=colours[1],
             lw=1.5, ls="-", label="$\Psi$$_{x}$", zorder=1)
    ax3.plot(df3y.index, df3y["psi_leaf"], c=colours[2],
             lw=1.5, ls="-", label="$\Psi$$_{l}$")

    ax4.plot(df4.index, df4["weighted_psi_soil"], c=colours[0],
             lw=1.5, ls="-", label="$\Psi$$_{sw}$", zorder=2)
    ax4.plot(df4x.index, df4x["psi_stem"], c=colours[1],
             lw=1.5, ls="-", label="$\Psi$$_{x}$", zorder=1)
    ax4.plot(df4y.index, df4y["psi_leaf"], c=colours[2],
             lw=1.5, ls="-", label="$\Psi$$_{l}$")

    ax1.set_ylim(-5, 0.1)
    ax2.set_ylim(-5, 0.1)
    ax3.set_ylim(-5, 0.1)
    ax4.set_ylim(-5, 0.1)

    axx1.plot(df1_met.index, df1_met["Rainf"].cumsum(), alpha=0.3,
              color="black", label="Rainfall")
    axx2.plot(df2_met.index, df2_met["Rainf"].cumsum(), alpha=0.3, color="black")
    axx3.plot(df3_met.index, df3_met["Rainf"].cumsum(), alpha=0.3, color="black")
    axx4.plot(df4_met.index, df4_met["Rainf"].cumsum(), alpha=0.3, color="black")

    print(df1_met["Rainf"].cumsum()[-1])
    print(df2_met["Rainf"].cumsum()[-1])
    print(df3_met["Rainf"].cumsum()[-1])
    print(df4_met["Rainf"].cumsum()[-1])
    axx1.set_ylim(0, 3000)
    axx2.set_ylim(-5, 3000)
    axx3.set_ylim(-5, 3000)
    axx4.set_ylim(-5, 3000)

    #axx1.bar(df1_met.index, df1_met["Rainf"], alpha=0.3, color="black")
    #axx1.set_yticks([0, 15, 30])

    #axx2.bar(df2_met.index, df2_met["Rainf"], alpha=0.3, color="black")
    #axx2.set_yticks([0, 15, 30])

    #axx3.bar(df3_met.index, df3_met["Rainf"], alpha=0.3, color="black")
    #axx3.set_yticks([0, 15, 30])

    ax2.set_ylabel("Water potential (MPa)", position=(0.5, 0.0))
    axx2.set_ylabel("Cumulative Rainfall (mm)", position=(0.5, 0.0))

    ax1.legend(numpoints=1, loc=(0.01, 0.14), frameon=False)
    axx1.legend(numpoints=1, loc=(0.01, 0.53), frameon=False)

    for a in axes:
        a.set_xlim([datetime.date(2016,1,1), datetime.date(2020, 1, 1)])

    from matplotlib.ticker import MaxNLocator
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax2.yaxis.set_major_locator(MaxNLocator(5))
    ax3.yaxis.set_major_locator(MaxNLocator(5))
    ax4.yaxis.set_major_locator(MaxNLocator(5))
    axx1.yaxis.set_major_locator(MaxNLocator(4))
    axx2.yaxis.set_major_locator(MaxNLocator(4))
    axx3.yaxis.set_major_locator(MaxNLocator(4))
    axx4.yaxis.set_major_locator(MaxNLocator(4))

    ax4.xaxis.set_major_locator(MaxNLocator(6))


    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    fig_label = "%s %s" % (next(labels_gen), labelsx[0])
    ax1.text(0.67, 0.1, fig_label,
            transform=ax1.transAxes, fontsize=12, verticalalignment='top',
            bbox=props)
    fig_label = "%s %s" % (next(labels_gen), labelsx[1])
    ax2.text(0.67, 0.1, fig_label,
            transform=ax2.transAxes, fontsize=12, verticalalignment='top',
            bbox=props)
    fig_label = "%s %s" % (next(labels_gen), labelsx[2])
    ax3.text(0.67, 0.1, fig_label,
            transform=ax3.transAxes, fontsize=12, verticalalignment='top',
            bbox=props)
    fig_label = "%s %s" % (next(labels_gen), labelsx[3])
    ax4.text(0.67, 0.1, fig_label,
            transform=ax4.transAxes, fontsize=12, verticalalignment='top',
            bbox=props)

    plot_fname = os.path.join(odir, "four_western_sites_water_potentials.pdf")
    fig.savefig(plot_fname, bbox_inches='tight', pad_inches=0.1)

def read_cable_file(fname, type=None):

    f = nc.Dataset(fname)
    time = nc.num2date(f.variables['time'][:],
                        f.variables['time'].units)
    #print (f.variables['time'][0], time[0], time[1])

    if type == "CABLE":
        df = pd.DataFrame(f.variables['weighted_psi_soil'][:,0,0],
                          columns=['weighted_psi_soil'])
        df['psi_leaf'] = f.variables['psi_leaf'][:,0,0]
        df['psi_stem'] = f.variables['psi_stem'][:,0,0]
        df['psi_soil1'] = f.variables['psi_soil'][:,0,0,0]
        df['psi_soil2'] = f.variables['psi_soil'][:,1,0,0]
        df['psi_soil3'] = f.variables['psi_soil'][:,2,0,0]
        df['psi_soil4'] = f.variables['psi_soil'][:,3,0,0]
        df['psi_soil5'] = f.variables['psi_soil'][:,4,0,0]
        df['psi_soil6'] = f.variables['psi_soil'][:,5,0,0]
        df['LAI'] = f.variables['LAI'][:,0,0]
        print(df.LAI.max())
    elif type == "MET":

        df = pd.DataFrame(f.variables['Rainf'][:,0,0], columns=['Rainf'])

    df['dates'] = time
    df = df.set_index('dates')

    return df

def label_generator(case='lower', start='', end=''):
    choose_type = {'lower': string.ascii_lowercase,
                   'upper': string.ascii_uppercase}
    generator = ('%s%s%s' %(start, letter, end) for letter in choose_type[case])

    return generator

if __name__ == "__main__":


    fname1 = "outputs/hydraulics_-30.40_151.60.nc"
    met_fname1 = "AWAP_single_pixel_-30.40_151.60.nc"
    fname2 = "outputs/hydraulics_-31.10_150.95.nc"
    met_fname2 = "AWAP_single_pixel_-31.10_150.95.nc"
    fname3 = "outputs/hydraulics_-31.10_142.50.nc"
    met_fname3 = "AWAP_single_pixel_-31.10_142.50.nc"
    fname4 = "outputs/hydraulics_-30.00_141.40.nc"
    met_fname4 = "AWAP_single_pixel_-30.00_141.40.nc"

    labelsx = ["GRW: 30.40$^{\circ}$S,151.60$^{\circ}$E", \
               "GRW: 31.10$^{\circ}$S,150.95$^{\circ}$E", \
               "SAW: 31.10$^{\circ}$S,142.50$^{\circ}$E",\
               "SAW: 30.00$^{\circ}$S,141.40$^{\circ}$E"]
    odir = "/Users/mdekauwe/Dropbox/Drought_risk_paper/figures/figs"
    #odir = "/Users/mdekauwe/Desktop"
    main(fname1, fname2, fname3, fname4, met_fname1, met_fname2, met_fname3,
         met_fname4, odir, labelsx)

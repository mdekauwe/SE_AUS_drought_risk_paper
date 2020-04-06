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

def main(fname, met_fname, odir):
    SEC_2_HLFHOUR = 1800.

    df = read_cable_file(fname, type="CABLE")
    df_met = read_cable_file(met_fname, type="MET")
    df_met['Rainf'] *= SEC_2_HLFHOUR
    method = {'Rainf':'sum'}
    df_met = df_met.resample("D").agg(method)

    fig = plt.figure(figsize=(9,6))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.2)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    colours = plt.cm.Set2(np.linspace(0, 1, 7))

    ax1 = fig.add_subplot(1,1,1)

    axx1 = ax1.twinx()


    df1 = df[(df.index.hour == 6)].copy()
    df2 = df[(df.index.hour == 12)].copy()

    axes = [ax1]
    axes2 = [axx1]


    ax1.plot(df.index, df["weighted_psi_soil"], c=colours[0],
             lw=1.5, ls="-", label="$\Psi$$_{s,weight}$")
    ax1.plot(df2.index, df2["psi_stem"], c=colours[1],
             lw=1.5, ls="-", label="$\Psi$$_{x}$")
    ax1.plot(df2.index, df2["psi_leaf"], c=colours[2],
             lw=1.5, ls="-", label="$\Psi$$_{l}$")
    ax1.set_ylim(-2.5, 0.1)

    axx1.bar(df_met.index, df_met["Rainf"], alpha=0.3, color="black")
    axx1.set_yticks([0, 15, 30])

    ax1.set_ylabel("Water potential (MPa)")
    axx1.set_ylabel("Rainfall (mm d$^{-1}$)")

    ax1.legend(numpoints=1, loc=(0.01, 0.14), frameon=False)


    for a in axes:
    #    #a.set_xlim([datetime.date(2002,8,1), datetime.date(2003, 8, 1)])
        a.set_xlim([datetime.date(2006,11,1), datetime.date(2007, 4, 1)])
    #    a.set_xlim([datetime.date(2002,10,1), datetime.date(2003, 4, 1)])

    from matplotlib.ticker import MaxNLocator
    ax1.yaxis.set_major_locator(MaxNLocator(5))

    plot_fname = os.path.join(odir, "tumba_water_potentials.pdf")
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

    elif type == "MET":
        df = pd.DataFrame(f.variables['Rainf'][:,0,0], columns=['Rainf'])

    df['dates'] = time
    df = df.set_index('dates')

    return df

if __name__ == "__main__":


    fname = "outputs/hydraulics_tumba.nc"
    met_dir = "/Users/mdekauwe/research/OzFlux"
    met_fname = os.path.join(met_dir, "TumbarumbaOzFlux2.0_met.nc")
    #odir = "/Users/mdekauwe/Dropbox/Drought_risk_paper/figures/figs"
    odir = "/Users/mdekauwe/Desktop"
    main(fname, met_fname, odir )

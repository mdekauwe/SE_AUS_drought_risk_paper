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

def main(fname, met_fname, plot_fname=None):
    SEC_2_HLFHOUR = 1800.

    df = read_cable_file(fname, type="CABLE")
    df_met = read_cable_file(met_fname, type="MET")
    df_met['Rainf'] *= SEC_2_HLFHOUR
    method = {'Rainf':'sum'}
    df_met = df_met.resample("D").agg(method)

    fig = plt.figure(figsize=(9,8))
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

    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)
    axx1 = ax1.twinx()
    axx2 = ax2.twinx()
    axx3 = ax3.twinx()

    df1 = df[(df.index.hour >= 5) & (df.index.hour < 6) &
             (df.index.minute >= 30)].copy()
    df2 = df[(df.index.hour >= 12) & (df.index.hour < 13) &
             (df.index.minute < 30)].copy()

    window = 48 * 3

    axes = [ax1, ax2, ax3]
    axes2 = [axx1, axx2, axx3]
    ax1.plot(df.index, df["theta1"].rolling(window=window).mean(), c=colours[1],
             lw=1.5, ls="-", label="1+2", zorder=1)
    ax1.plot(df.index, df["theta2"].rolling(window=window).mean(), c=colours[2],
             lw=1.5, ls="-", label="3+4")
    ax1.plot(df.index, df["theta3"].rolling(window=window).mean(), c=colours[0],
             lw=1.5, ls="-", label="5+6")
    #ax1.plot(df.index, df["theta4"], c=colours[4],
    #         lw=1.5, ls="-", label="4")
    #ax1.plot(df.index, df["theta5"], c=colours[5],
    #         lw=1.5, ls="-", label="5")
    #ax1.plot(df.index, df["theta6"], c=colours[6],
    #         lw=1.5, ls="-", label="6")
    #ax1.set_ylim(-0.3, 0.0)

    ax2.plot(df.index, df["weighted_psi_soil"], c=colours[2],
             lw=1.5, ls="-", label="Pre-dawn $\Psi$$_{s}$ weight")

    ax3.plot(df2.index, df2["psi_leaf"], c=colours[2],
             lw=1.5, ls="-", label="Midday $\Psi$$_{l}$")
    ax3.plot(df2.index, df2["psi_stem"], c=colours[0],
             lw=1.5, ls="-", label="Midday $\Psi$$_{stem}$")
    #ax3.set_ylim(-5, 0)
    axx1.bar(df_met.index, df_met["Rainf"], alpha=0.3, color="black")
    axx2.bar(df_met.index, df_met["Rainf"], alpha=0.3, color="black")
    axx3.bar(df_met.index, df_met["Rainf"], alpha=0.3, color="black")

    ax1.set_ylabel("$\Theta$ (m$^{3}$ m$^{-3}$)")
    ax2.set_ylabel("Water potential (MPa)")
    axx2.set_ylabel("Rainfall (mm d$^{-1}$)")

    ax1.legend(numpoints=1, loc="best")
    ax2.legend(numpoints=1, loc="best")
    ax3.legend(numpoints=1, loc="best")

    #for a in axes:
    #    #a.set_xlim([datetime.date(2002,8,1), datetime.date(2003, 8, 1)])
    #    #a.set_xlim([datetime.date(2004,1,1), datetime.date(2004, 8, 1)])
    #    a.set_xlim([datetime.date(2002,10,1), datetime.date(2003, 4, 1)])


    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    if plot_fname is None:
        plt.show()
    else:
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

        #df['theta1'] = f.variables['SoilMoist'][:,0,0,0]
        #df['theta2'] = f.variables['SoilMoist'][:,1,0,0]
        #df['theta3'] = f.variables['SoilMoist'][:,2,0,0]
        #df['theta4'] = f.variables['SoilMoist'][:,3,0,0]
        #df['theta5'] = f.variables['SoilMoist'][:,4,0,0]
        #df['theta6'] = f.variables['SoilMoist'][:,5,0,0]

        # layer thickness
        zse = [.022, .058, .154, .409, 1.085, 2.872]

        frac1 = zse[0] / (zse[0] + zse[1])
        frac2 = zse[1] / (zse[0] + zse[1])
        frac3 = zse[2] / (zse[2] + zse[3])
        frac4 = zse[3] / (zse[2] + zse[3])
        frac5 = zse[4] / (zse[4] + zse[4])
        frac6 = zse[5] / (zse[5] + zse[5])
        df['theta1'] = (f.variables['SoilMoist'][:,0,0,0] * frac1) + \
                       (f.variables['SoilMoist'][:,1,0,0] * frac2)
        df['theta2'] = (f.variables['SoilMoist'][:,2,0,0] * frac3) + \
                       (f.variables['SoilMoist'][:,3,0,0] * frac4)
        df['theta3'] = (f.variables['SoilMoist'][:,4,0,0] * frac4) + \
                       (f.variables['SoilMoist'][:,5,0,0] * frac5)

    elif type == "MET":
        df = pd.DataFrame(f.variables['Rainf'][:,0,0], columns=['Rainf'])

    df['dates'] = time
    df = df.set_index('dates')

    return df

if __name__ == "__main__":

    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-f", "--fname", dest="fname",
                      action="store", help="filename",
                      type="string")
    parser.add_option("-p", "--plot_fname", dest="plot_fname", action="store",
                      help="Benchmark plot filename", type="string")
    (options, args) = parser.parse_args()

    met_dir = "/Users/mdekauwe/research/OzFlux"
    met_fname = os.path.join(met_dir, "TumbarumbaOzFlux2.0_met.nc")
    #met_dir = "/Users/mdekauwe/research/OzFlux"
    #met_fname = os.path.join(met_dir, "WombatStateForestOzFlux2.0_met.nc")

    #main(options.fname, met_fname, options.plot_fname)

    fname = "outputs/hydraulics_desica.nc"
    main(fname, met_fname)

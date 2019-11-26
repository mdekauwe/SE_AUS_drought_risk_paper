#!/usr/bin/env python

"""
Plot visual benchmark (average seasonal cycle) of old vs new model runs.

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
from optparse import OptionParser

def main(site, met_fname, flux_fname, fname1, fname2, plot_fname=None):

    df1 = read_cable_file(fname1, type="CABLE")
    df1 = resample_timestep(df1, type="CABLE")

    df2 = read_cable_file(fname2, type="CABLE")
    df2 = resample_timestep(df2, type="CABLE")

    df_flx = read_cable_file(flux_fname, type="FLUX")
    df_flx = resample_timestep(df_flx, type="FLUX")

    df_met = read_cable_file(met_fname, type="MET")
    df_met = resample_timestep(df_met, type="MET")

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

    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    axx1 = ax1.twinx()
    axx2 = ax2.twinx()

    axes = [ax1, ax2,]
    axes2 = [axx1, axx2,]
    vars = ["GPP", "Qle"]
    for a, x, v in zip(axes, axes2, vars):
        if v == "Qle":
            a.plot(df_flx[v].index.to_pydatetime(), df_flx[v].rolling(window=7).mean(), c=colours[1], lw=2.0,
                ls="-", label="Observations")
        a.plot(df1[v].index.to_pydatetime(), df1[v].rolling(window=7).mean(), c=colours[0],
               lw=1.5, ls="-", label="Standard")
        a.plot(df2[v].index.to_pydatetime(), df2[v].rolling(window=7).mean(), c=colours[2],
               lw=1.5, ls="-", label="Hydraulics")

        x.bar(df_met.index, df_met["Rainf"], alpha=0.3, color="black")
    ax2.set_ylim(0, 200)
    labels = ["GPP (g C m$^{-2}$ d$^{-1}$)", "LE (W m$^{-2}$)"]
    for a, l in zip(axes, labels):
        a.set_ylabel(l, fontsize=12)

    for x, l in zip(axes2, labels):
        x.set_ylabel("Rainfall (mm d$^{-1}$)", fontsize=12)


    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.legend(numpoints=1, loc="best")


    for a in axes:
    #    #a.set_xlim([datetime.date(2002,10,1), datetime.date(2003, 4, 1)])
        a.set_xlim([datetime.date(2012,7,1), datetime.date(2013, 8, 1)])
    #    #a.set_xlim([datetime.date(2006,11,1), datetime.date(2007, 4, 1)])

    if plot_fname is None:
        plt.show()
    else:
        #fig.autofmt_xdate()
        fig.savefig(plot_fname, bbox_inches='tight', pad_inches=0.1)


def read_cable_file(fname, type=None):

    f = nc.Dataset(fname)
    time = nc.num2date(f.variables['time'][:],
                        f.variables['time'].units)


    if type == "CABLE":
        df = pd.DataFrame(f.variables['GPP'][:,0,0], columns=['GPP'])
        df['Qle'] = f.variables['Qle'][:,0,0]
        df['LAI'] = f.variables['LAI'][:,0,0]
        df['TVeg'] = f.variables['TVeg'][:,0,0]
        df['ESoil'] = f.variables['ESoil'][:,0,0]
        df['NEE'] = f.variables['NEE'][:,0,0]
    elif type == "FLUX":
        df = pd.DataFrame(f.variables['Qle'][:,0,0], columns=['Qle'])
        df['GPP'] = f.variables['GPP'][:,0,0]

    elif type == "MET":
        df = pd.DataFrame(f.variables['Rainf'][:,0,0], columns=['Rainf'])

    df['dates'] = time
    df = df.set_index('dates')

    return df



def resample_timestep(df, type=None):

    UMOL_TO_MOL = 1E-6
    MOL_C_TO_GRAMS_C = 12.0
    SEC_2_HLFHOUR = 1800.

    if type == "CABLE":
        # umol/m2/s -> g/C/30min
        df['GPP'] *= UMOL_TO_MOL * MOL_C_TO_GRAMS_C * SEC_2_HLFHOUR

        # kg/m2/s -> mm/30min
        df['TVeg'] *= SEC_2_HLFHOUR

        method = {'GPP':'sum', 'TVeg':'sum', "Qle":"mean"}
    elif type == "FLUX":
        # umol/m2/s -> g/C/30min
        df['GPP'] *= UMOL_TO_MOL * MOL_C_TO_GRAMS_C * SEC_2_HLFHOUR

        method = {'GPP':'sum', "Qle":"mean"}

    elif type == "MET":
        # kg/m2/s -> mm/30min
        df['Rainf'] *= SEC_2_HLFHOUR

        method = {'Rainf':'sum'}

    df = df.resample("D").agg(method)

    return df


    return dates

if __name__ == "__main__":

    site = "WombatStateForest"
    output_dir = "outputs"
    #flux_dir = "../../flux_files/"
    #met_dir = "../../met_data/plumber_met/"
    met_dir = "/Users/mdekauwe/research/OzFlux"
    flux_dir = "/Users/mdekauwe/research/OzFlux"

    """
    parser = OptionParser()
    parser.add_option("-a", "--fname1", dest="fname1",
                      action="store", help="filename",
                      type="string",
                      default=os.path.join(output_dir, "original_out.nc"))
    parser.add_option("-b", "--fname2", dest="fname2",
                      action="store", help="filename",
                      type="string",
                      default=os.path.join(output_dir, "%s_out.nc" % site))
    parser.add_option("-p", "--plot_fname", dest="plot_fname", action="store",
                      help="Benchmark plot filename", type="string")
    (options, args) = parser.parse_args()
    """

    flux_fname = os.path.join(flux_dir, "WombatStateForestOzFlux2.0_flux.nc")
    met_fname = os.path.join(met_dir, "WombatStateForestOzFlux2.0_met.nc")
    #flux_fname = os.path.join(flux_dir, "GreatWesternWoodlandsOzFlux2.0_flux.nc")
    #met_fname = os.path.join(met_dir, "GreatWesternWoodlandsOzFlux2.0_met.nc")


    fname1 = "outputs/standard.nc"
    fname2 = "outputs/hydraulics_desica.nc"
    main(site, met_fname, flux_fname,fname1, fname2)

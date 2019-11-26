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

def main(fname, plot_fname=None):

    df = read_cable_file(fname)
    df = resample_to_seasonal_cycle(df)

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

    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)


    axes = [ax1, ax2,]
    vars = ["GPP", "TVeg"]
    for a, v in zip(axes, vars):
        a.plot(df[v].rolling(window=7).mean(), c="royalblue", lw=1.0, ls="-")


    labels = ["GPP (g C m$^{-2}$ d$^{-1}$)", "TVeg (mm d$^{-1}$)"]
    for a, l in zip(axes, labels):
        a.set_ylabel(l, fontsize=12)


    plt.setp(ax1.get_xticklabels(), visible=False)

    if plot_fname is None:
        plt.show()
    else:
        fig.savefig(plot_fname, bbox_inches='tight', pad_inches=0.1)


def read_cable_file(fname):

    f = nc.Dataset(fname)
    time = nc.num2date(f.variables['time'][:],
                        f.variables['time'].units)
    #print (f.variables['time'][0], time[0], time[1])
    df = pd.DataFrame(f.variables['GPP'][:,0,0], columns=['GPP'])
    df['Qle'] = f.variables['Qle'][:,0,0]
    df['LAI'] = f.variables['LAI'][:,0,0]
    df['TVeg'] = f.variables['TVeg'][:,0,0]
    df['ESoil'] = f.variables['ESoil'][:,0,0]
    df['NEE'] = f.variables['NEE'][:,0,0]

    df['dates'] = time
    df = df.set_index('dates')

    return df

def resample_to_seasonal_cycle(df, OBS=False):

    UMOL_TO_MOL = 1E-6
    MOL_C_TO_GRAMS_C = 12.0
    SEC_2_HLFHOUR = 1800.

    # umol/m2/s -> g/C/30min
    df['GPP'] *= UMOL_TO_MOL * MOL_C_TO_GRAMS_C * SEC_2_HLFHOUR

    # kg/m2/s -> mm/30min
    df['TVeg'] *= SEC_2_HLFHOUR

    method = {'GPP':'sum', 'TVeg':'sum'}
    df = df.resample("D").agg(method)

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

    main(options.fname, options.plot_fname)

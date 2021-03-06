#!/usr/bin/env python

"""
Get LAI, soil param ranges for veg class

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (18.03.2020)"
__email__ = "mdekauwe@gmail.com"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import sys

def main():

    ds = xr.open_dataset(os.path.join("/Users/mdekauwe/Desktop/",
                                      "gridinfo_AWAP_OpenLandMap_new_iveg.nc"))
    #print(ds)

    lat_bnds, lon_bnds = [-40, -28], [140, 154]
    ds = ds.sel(latitude=slice(*lat_bnds), longitude=slice(*lon_bnds))

    iveg = ds["iveg"][:,:].values
    lai = ds["LAI"][:,:,:].values
    sucs = ds["sucs"][:,:].values
    bch = ds["bch"][:,:].values

    idx_rf = np.argwhere(iveg == 18.0)
    idx_wsf = np.argwhere(iveg == 19.0)
    idx_dsf = np.argwhere(iveg == 20.0)
    idx_grw = np.argwhere(iveg == 21.0)
    idx_saw = np.argwhere(iveg == 22.0)

    for veg_num in np.arange(18.0, 23.0):
        lai_veg = lai[:,iveg == veg_num]

        q3 = np.percentile(lai_veg, 75)
        q1 = np.percentile(lai_veg, 25)
        iqr = q3 - q1
        upp = q3 + (1.5 * iqr)
        low = q1 - (1.5 * iqr)

        print("size1", veg_num, lai_veg.shape)
        xx = lai_veg[11,:]
        xx = xx[xx> 1.5]


        print("size2", veg_num, len(xx))

        print(veg_num, np.nanmin(lai_veg), np.nanmax(lai_veg))
        print(veg_num, np.round(q1, 2), np.round(q3,2))
        #print(veg_num, np.round(low, 2),
        #      np.round(upp,2))



    print(" ")
    KPA_2_MPa = 0.001
    M_HEAD_TO_MPa = 9.8 * KPA_2_MPa

    for veg_num in np.arange(18.0, 23.0):

        sucs_veg = sucs[iveg == veg_num]
        q3 = np.percentile(sucs_veg, 75) * M_HEAD_TO_MPa
        q1 = np.percentile(sucs_veg, 25) * M_HEAD_TO_MPa

        print("%.5f,%.5f" % (q1, q3))



    print(" ")
    for veg_num in np.arange(18.0, 23.0):

        bch_veg = bch[iveg == veg_num]
        q3 = np.percentile(bch_veg, 75)
        q1 = np.percentile(bch_veg, 25)

        print("%.2f,%.2f" % (q1, q3))

        #print("%.2f,%.2f,%.2f" % (np.nanmin(bch_veg),
        #                          np.nanmax(bch_veg),
        #                          np.nanmean(bch_veg)))

if __name__ == "__main__":

    main()

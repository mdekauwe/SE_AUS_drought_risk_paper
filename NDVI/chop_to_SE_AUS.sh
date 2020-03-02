#!/bin/bash

# Raw data ...  #/srv/ccrc/data51/z3466821/NDVI/GIMMSv3.1.1
cdo sellonlatbox,112,154,-44,-10 ndvi3g_geo_v1_1_1981to2017_mergetime_ndviMonMax.nc tmp.nc
mv tmp.nc ndvi3g_geo_v1_1_1981to2017_ndviMonMax_SE_AUS.nc

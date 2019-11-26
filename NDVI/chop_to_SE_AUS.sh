#!/bin/bash

cdo sellonlatbox,112,154,-44,-10 ndvi3g_geo_v1_1_1981to2017_mergetime_ndviMonMax.nc tmp.nc
mv tmp.nc ndvi3g_geo_v1_1_1981to2017_ndviMonMax_SE_AUS.nc

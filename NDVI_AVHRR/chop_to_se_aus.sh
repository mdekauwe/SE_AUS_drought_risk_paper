#!/bin/bash

gdal_translate -projwin 140 -28 154 -40 AVHRR_EVI2_monmean_Australia_1982_2019.tif AVHRR_EVI2_SEAUS_1982_2019.tif

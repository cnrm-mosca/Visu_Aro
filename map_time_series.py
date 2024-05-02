#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
#------------ map_time_series_comp_obs.py --------------------
#   created 05/07/2023 by haradercoustaue
#
# !! Add back targ to graphs 
#----------------------------------------------------------

from lambertools import matchLambert, nearestLambert 
import xarray as xr
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import scipy.signal
import scipy.stats
import time
import ssl
import requests
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#----------- User Input ------------------------------------

# Enter year as a string
year = '2015'

# File locations
source_dir   = "./"

# File name

varname     = "SFX.TS"
filename    = varname+".ICMSH0123+cat.sfx.nc"

#-----------------------------------------------------------

# Allow Python to pass proxy to retrieve political and geographical borders
requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

if source_dir[-1]!="/":
        source_dir=source_dir+"/"
        print("Warning : you forgot the forward slash in the file path.")


data    = xr.open_dataset(source_dir+filename) 
dim_change = dict(X="x",Y="y")
name_change = dict(latitude="lat",longitude="lon")
data=data.rename_dims(dims_dict=dim_change)
data=data.rename(name_dict=name_change)



# Points
f, ax1 = plt.subplots(1,1, figsize=(8, 8))

ind_x,ind_y = nearestLambert(16.221,38.448,data)
sel_pred=data[varname].isel(x=ind_x, y=ind_y) 
sel_pred.plot(color='green', ls="dashed",ax=ax1)
plt.tight_layout()

# Save plots
plt.savefig(source_dir+varname+"_time_series.png")

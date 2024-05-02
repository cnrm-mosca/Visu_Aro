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
import seaborn as sns
import metpy 
import ssl
import requests
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#----------- User Input ------------------------------------

# Enter year as a string
year = '2003'

# File locations
source_dir   = "/scratch/mbec/haradercoustaue/Data_NN/"

# File name

filename    = "pred_CNRM_fra_test_1979_2018.nc"
simul_name  = "CNRM_fra_test_1979_2018_nsfc"
datalabel   = "ERAI down"

# Target name 

targname    = "tas_EUR-11_ECMWF-ERAINT_evaluation_r1i1p1_CNRM-ALADIN63_v1_day_19790101-20181231.nc" 
targlabel   = "Aladin forced ERAI"

# Remapping needs to be performed on sotrtm34-sidev before transferring the 
# obs here. Note that down_X_GCM_EUC12_ERAI_hist_1979_1979_fullvar_smth3_with_sfc.nc
# does not contain enough grid information for conservative regridding, cdo must 
# use remapnn or remapbil  

obsname     = "eobs_remapbil_dom_FRA_toflt_tg_ens_mean_0.1deg_reg_v27.0e.nc"
obslabel    = "eobs"

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


data    = xr.open_dataset(source_dir+"OUTPUT/"+filename) 
target  = xr.open_dataset(source_dir+"INPUT/"+targname) 
obs     = xr.open_dataset(source_dir+"OBS/"+obsname) 


# Convert to celsius
target['tas'] = target['tas']-273.15
data['pred'] = data['pred']


# Keep only data points corresponding to predictant period
# Apply a 9h tolerance to start/end because eobs dates are at midnight and ERAI daily means at 9h
time_start = target['time'].sel(time=data['time'].values[0], method ="nearest",tolerance=np.timedelta64(9,'h')).values
time_stop  = target['time'].sel(time=data['time'].values[-1], method ="nearest",tolerance=np.timedelta64(9,'h')).values

# Take a subset of data corresponding to 
# the domain of our predictand output 
target = matchLambert(data, target,3)
target = target.sel(time=slice(time_start, time_stop))

time_start = obs['time'].sel(time=data['time'].values[0], method ="nearest",tolerance=np.timedelta64(9,'h')).values
time_stop  = obs['time'].sel(time=data['time'].values[-1], method ="nearest",tolerance=np.timedelta64(9,'h')).values

obs = obs.sel(time=slice(time_start, time_stop))

# Set lat lon bounds based on data 
lat_min = min(data['lat'].values.flatten())
lat_max = max(data['lat'].values.flatten())

lon_min = min(data['lon'].values.flatten())
lon_max = max(data['lon'].values.flatten())

# Verification of leap years : normally the algo removes these, but the first version kept them
# so we add a check and remove if necessary


if  np.any(np.array([(data.time.dt.month==2) & (data.time.dt.day==29)])):
    print("Predictands contain leap years, keeping")
else:
    print("Removing leap years from data")
    target = target.sel(time=~((target.time.dt.month==2) & (target.time.dt.day==29)))
    obs = obs.sel(time=~((obs.time.dt.month==2) & (obs.time.dt.day==29)))

print("target after subsetting to match predictand")
print(target)


target = target.sel(time=year)
obs    = obs.sel(time=year) 
data   = data.sel(time=year) 

# -------- Plotting Antoine's Metrics ----------------- 

# Point metrics 

dict_pts = dict(Montpellier ={'lat':43.61, 'lon':3.87, 'x':np.nan, 'y':np.nan}, Mont_Aigoual ={'lat':44.12, 'lon':3.57, 'x':np.nan, 'y':np.nan}, Sisteron ={'lat':44.20, 'lon':5.94, 'x':np.nan, 'y':np.nan}, Paris={'lat':48.86, 'lon':2.34, 'x':np.nan, 'y':np.nan}, Rennes={'lat':48.11,'lon':-1.67, 'x':np.nan, 'y':np.nan}, Dijon={'lat':47.31, 'lon':5.04, 'x':np.nan, 'y':np.nan})

for city in dict_pts:
  dict_pts[city]['x'], dict_pts[city]['y'] = nearestLambert(dict_pts[city]['lat'],dict_pts[city]['lon'],data)


f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(12, 12))
ax_list = [ax1,ax2,ax3,ax4,ax5,ax6]

for ind,key in enumerate(dict_pts.keys()):
    sel_pred = data.isel(x=dict_pts[key]['x'], y=dict_pts[key]['y'])
    sel_pred.pred.plot(color='green', ls="dashed",ax=ax_list[ind],  label="Emulator")
    

    sel_obs = obs.isel(x=dict_pts[key]['x'], y=dict_pts[key]['y'])
    sel_obs.tg.plot(color='blue',ls='dashdot',ax=ax_list[ind],  label="Eobs")

 
    sel_targ = target.isel(x=dict_pts[key]['x'], y=dict_pts[key]['y'])
    sel_targ.tas.plot(color='red', ls=":",ax=ax_list[ind],  label="ALADIN")

    ax_list[ind].set_title(key)
    ax_list[ind].legend(loc='upper right')

plt.tight_layout()

# Save plots
plt.savefig(source_dir+"OUTPUT/"+simul_name+"_time_series_pts_w_obs_"+year+".png")

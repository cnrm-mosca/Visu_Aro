#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#----------------------------------------------------------
#   created 16/03/2023 by haradercoustaue
#
#----------------------------------------------------------

from lambertools import matchLambert, nearestLambert 
import xarray as xr
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import scipy.signal
import time
import ssl
import requests
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#----------- User Input ------------------------------------

# File locations
source_dir   = "/cnrm/mosca/USERS/haradercoustaue/Donnees/ARO/NO_SAVE/"

# File name 

filename    = "rsds_FRA-3_ECMWF-ERA5_evaluation_r1i1p1_CNRM-AROME46t1_fpsconv-x0n1-v1_mon_201501-201512.nc"
filelabel   = "FRA3-0.00"

# Comparison Simulation name

compname    = "rsds_FRA-3_ECMWF-ERA5_evaluation_r1i1p1_CNRM-AROME46t1_fpsconv-x0n1-v1_mon_201501_201512.nc"
complabel   = "FRA3-QSAT"

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

data = xr.open_dataset(source_dir+filelabel+"/"+filename) 
comp  = xr.open_dataset(source_dir+complabel+"/"+compname) 

# Get graph extent
# Builtin min/max functions need a flattened iterable list 
# otherwise the true/false evaluation at the end will fail 
lat_min = min(data['lat'].values.flatten())+0.7
lat_max = max(data['lat'].values.flatten())-0.7

lon_min = min(data['lon'].values.flatten())+1.3
lon_max = max(data['lon'].values.flatten())-0.3


# ----------- stdev plot --------------
# I replaced the xarray plot method with matplotlib's pcolormesh which manages the Lambert projection
# better (in my experience)

proj_orig = ccrs.LambertConformal(central_longitude=4.8, central_latitude=46.2, standard_parallels=[46.2])
proj = ccrs.PlateCarree()

fig,axes = plt.subplots(5,3, figsize=(12,20), layout="constrained",subplot_kw=dict(projection=proj_orig))

# ---------------- ANN -------------------------------

data['rsds'] = data['rsds'].assign_coords(season=data.time.dt.season)
# Initialise diff to assign coordinates, values to replace later 
data['diff'] = data['rsds'].assign_coords(season=comp.time.dt.season)
comp['rsds'] = comp['rsds'].assign_coords(season=comp.time.dt.season)

data['mean'] = data['rsds'].mean(axis=0)
comp['mean'] = comp['rsds'].mean(axis=0)
data['diff'] = comp['rsds'].values - data['rsds'] 
data['mean_diff'] = data['diff'].mean(axis=0) 

#Preparation seasons to follow
data_seasons = data['rsds'].groupby("season").mean()
comp_seasons = comp['rsds'].groupby("season").mean()
diff_seasons = data['diff'].groupby("season").mean()

ax = axes[0,0]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, data["mean"].values, cmap="Spectral_r", transform=proj,vmin=50, vmax=250)  
ax.set_title(filelabel+" Mean")

ax = axes[0,1]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, comp["mean"].values, cmap="Spectral_r", transform=proj, vmin=50, vmax=250)
ax.set_title(complabel+" Mean")

ax = axes[0,2]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, data["mean_diff"].values, cmap="coolwarm", transform=proj, vmin=-30,vmax=30)
ax.set_title(complabel+" - "+filelabel)



# ---------------- Winter -------------------------------

ax =  axes[1,0]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, data_seasons.sel(season="DJF").values, cmap="Spectral_r", transform=proj, vmin=50, vmax=250)  
ax.set_title(filelabel+" Mean DJF")

ax =  axes[1,1]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, comp_seasons.sel(season="DJF").values, cmap="Spectral_r", transform=proj, vmin=50, vmax=250)
ax.set_title(complabel+" Mean DJF")

ax =  axes[1,2]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, diff_seasons.sel(season="DJF").values, cmap="coolwarm", transform=proj, vmin=-30,vmax=30)
ax.set_title(complabel+" - "+filelabel+" DJF")

# ---------------- Spring -------------------------------

ax =  axes[2,0]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, data_seasons.sel(season="MAM").values, cmap="Spectral_r", transform=proj, vmin=50, vmax=250)  
ax.set_title(filelabel+" Mean MAM")

ax =  axes[2,1]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, comp_seasons.sel(season="MAM").values, cmap="Spectral_r", transform=proj, vmin=50, vmax=250)
ax.set_title(complabel+" Mean MAM")

ax =  axes[2,2]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, diff_seasons.sel(season="MAM").values, cmap="coolwarm", transform=proj, vmin=-30,vmax=30)
ax.set_title(complabel+" - "+filelabel+" MAM")


# ---------------- Summer -------------------------------

ax = axes[3,0]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, data_seasons.sel(season="JJA").values, cmap="Spectral_r", transform=proj, vmin=50, vmax=250)  
ax.set_title(filelabel+" Mean JJA")

ax = axes[3,1]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, comp_seasons.sel(season="JJA").values, cmap="Spectral_r", transform=proj, vmin=50, vmax=250)
ax.set_title(complabel+" Mean JJA")

ax = axes[3,2]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, diff_seasons.sel(season="JJA").values, cmap="coolwarm", transform=proj, vmin=-30,vmax=30)
ax.set_title(complabel+" - "+filelabel+" JJA")


# ---------------- Fall -------------------------------

ax = axes[4,0]
ax.coastlines()
pcm= ax.pcolormesh(data["lon"].values, data["lat"].values, data_seasons.sel(season="SON").values, cmap="Spectral_r", transform=proj, vmin=50, vmax=250)  
ax.set_title(filelabel+" Mean SON")

ax = axes[4,1]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, comp_seasons.sel(season="SON").values, cmap="Spectral_r", transform=proj, vmin=50, vmax=250)
ax.set_title(complabel+" Mean SON")
fig.colorbar(pcm,ax=axes[:, 0:2],orientation="horizontal",anchor=(1.2,-3),shrink=0.5) 

ax = axes[4,2]
ax.coastlines()
pcm = ax.pcolormesh(data["lon"].values, data["lat"].values, diff_seasons.sel(season="SON").values, cmap="coolwarm", transform=proj, vmin=-30,vmax=30)
ax.set_title(complabel+" - "+filelabel+" SON")
fig.colorbar(pcm,ax=axes[:, 2],orientation="horizontal",anchor=(-1,-3))



plt.savefig("rsds_clim_wseas_"+filelabel+"_"+complabel+".png")

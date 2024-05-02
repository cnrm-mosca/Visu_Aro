#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------lambertools----------------------------
#   created 29/06/2023 by haradercoustaue
#   Reunites several lambert grid matching functions
#   
#   fonction trend coded by dourya    
#----------------------------------------------------------


import xarray as xr
import numpy as np
from scipy import stats


def trend(array,nbday):
    '''
    Trend takes a regular lat lon grid (time,lat,lon) or a lambert (time,y,x).
    Excepting time which must be the 0th coord, the spatial coordinate order is not
    important and outputs will respect original order (time coordinate is squeezed by computation)
    Second coordinate is number of days per year and must be constant. Remember to remove leap days !  
    '''
    TRENDS=np.zeros(array.shape[1:3])
    yrs=int(array.shape[0]/nbday)
    yr_mean=array.reshape([yrs,nbday,array.shape[1],array.shape[2]]).mean(axis=1)
     
    for i in range(yr_mean.shape[1]):
        for j in range(yr_mean.shape[2]):
            TRENDS[i,j]=stats.linregress(range(yr_mean.shape[0]),yr_mean[:,i,j]).slope
           
    return(TRENDS)

def matchLambert(ds_target, ds_origin,precision):
    '''
     These are lambert grids
     ds_target is the grid destination
     ds_origin is the dataset which we will subset to match ds_target
     data sets must be of format ds[time,y,x]
     Word of Warning : it may be tempting to try and remplace some of following 
     code by coordinate name style selection, ie data.sel(lat=slice(43,44), lon=slice(-1,3))
     DONT DO IT, this fails miserably because lat, lon are not dimension/coordinate variables 
     the coordinates of lambert grids are x,y
    '''       

    latt  = np.round(ds_target['lat'].values, decimals=precision)
    lont  = np.round(ds_target['lon'].values, decimals=precision)

    lato = np.round(ds_origin['lat'].values, decimals=precision)
    lono = np.round(ds_origin['lon'].values, decimals=precision)

    iy_pts=[]
    ix_pts=[]

    # We take the corners of the smaller destination grid which is a subset of
    # the origin grid
    corners_lats = [latt[0,0],  latt[0,-1],  latt[-1,0], latt[-1,-1]]
    corners_lons = [lont[0,0], lont[0,-1], lont[-1,0], lont[-1,-1]]

    for ind in range(4):
        lat_pt = np.argwhere(lato==corners_lats[ind])
        lon_pt = np.argwhere(lono==corners_lons[ind])
        matches = np.vstack((lat_pt,lon_pt))
        unq,count = np.unique(matches, axis=0, return_counts=True)
        pt_matches = unq[count>1]
        if len(pt_matches==1):
            print(f"matchLambert : aligned grids at {precision} decimal point precision")
        else:
            print(f"lat pt is {lat_pt}")
            print(f"lon pt is {lon_pt}")
            raise ValueError("matchLambert : grids are incompatible or precision is too low, please regrid and try again.")
        iy_pts.append(pt_matches[0][0])
        ix_pts.append(pt_matches[0][1])

    iy_pts = np.sort(np.unique(iy_pts))
    ix_pts = np.sort(np.unique(ix_pts))

    #  Slice along indices, will not include endpoint so we add +1
    return ds_origin.isel(y=slice(iy_pts[0],iy_pts[-1]+1),x=slice(ix_pts[0],ix_pts[-1]+1))



def nearestLambert(lat_xy,lon_xy, ds):
    dist_list    = np.empty(len(ds['lat'].values.flatten()))
    dist_list[:] = np.nan

    dest_lat = np.deg2rad(lat_xy)
    dest_lon = np.deg2rad(lon_xy)

    ind = 0

    # Calculate great circle distance between lambert points (lats lons of ds) and the desired point(lat_xy, lon_xy)
    for alat,alon in zip(ds['lat'].values.flatten(),ds['lon'].values.flatten()):
        R = 6371.0 # Radius of the earth

        lat_pt = np.deg2rad(alat)
        dlat_pt = dest_lat - lat_pt
        lon_pt = np.deg2rad(alon)
        dlon_pt = dest_lon - lon_pt

        a =  np.sin(dlat_pt / 2) ** 2  + (1 -  np.sin(dlat_pt / 2)**2 -  np.sin((lat_pt+alat)/ 2)**2 )*(np.sin(dlon_pt/2)**2)
        c = 2*np.arcsin(np.sqrt(a))

        dist_list[ind] = R * c
        ind +=1

    # Get the point with the minimum distance then use unravel to get xy coordinates
    # This is the better way of getting indices of a lat/lon pt with lambert x,y indices => flatten, argmin and unravel
    # !! attention, x will be second index in inds !!
    locmin = np.argmin(dist_list)
    inds = np.unravel_index(locmin, ds['lat'].values.shape, order='C')

    # Ravel returns indicies as [y,x] (b/c data is pred(time,y,x)!!)
    # we thus return in format x,y with the second index first
    return inds[1],inds[0]



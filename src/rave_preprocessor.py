# -*- coding: utf-8 -*-
"""
Author: whung

This script is used for gridded FRP pre-processing.

DATA SOURCE: RAVE
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import os

import warnings
warnings.simplefilter(action='ignore')



def preprocessor(filename, time, lat_lim, lon_lim):
    namelist = pd.read_csv('./input/namelist', header=None, delimiter='=')
    namelist = namelist[1]
    path_frp = str(namelist[18])

    f_output   = './input/'+time+'/'+filename+'.'+time+'.nc'

    date = time[:8]
    hour = time[8:10]
    

    '''Reading Data'''
    f_ori = path_frp+'/Hourly_Emissions_3km_'+date+'0000_'+date+'2300.nc'
    hour_index = int(hour)
    
    if os.path.isfile(f_ori) == True:
        readin = Dataset(f_ori)
        yt = np.flip(readin['grid_latt'][:, 0])
        xt = readin['grid_lont'][0, :]
    
        data = np.squeeze(readin['FRP_MEAN'][hour_index, :, :])
        data = np.flipud(data)
        data = np.array(data)       # fill value = -1
        data[data==-1] = 0
    
        qa = readin['QA'][hour_index, :, :]
        qa = np.flipud(qa)
        data[qa==1] = 0             # use QA = 2 and 3 only
    
        index1  = np.squeeze(np.argwhere((yt>=lat_lim[0]) & (yt<=lat_lim[1])))
        index2  = np.squeeze(np.argwhere((xt>=lon_lim[0]) & (xt<=lon_lim[1])))
    
        if index1[0] == 0:
            yt = yt[index1[0]:index1[-1]+2]
            xt = xt[index2[0]-1:index2[-1]+2]
            data = data[index1[0]:index1[-1]+2, index2[0]-1:index2[-1]+2]
        elif index2[0] == 0:
            yt = yt[index1[0]-1:index1[-1]+2]
            xt = xt[index2[0]:index2[-1]+2]
            data = data[index1[0]-1:index1[-1]+2, index2[0]:index2[-1]+2]
        else:
            yt = yt[index1[0]-1:index1[-1]+2]
            xt = xt[index2[0]-1:index2[-1]+2]
            data = data[index1[0]-1:index1[-1]+2, index2[0]-1:index2[-1]+2]
    
        xt_grid, yt_grid = np.meshgrid(xt, yt)

        readin.close()
        del [readin, yt, xt, qa, index1, index2]
    
    else:
        return 1


    
    '''Write NetCDF File'''
    f = Dataset(f_output, 'w')
    f.createDimension('time', 1)
    f.createDimension('nlat', data.shape[0])
    f.createDimension('nlon', data.shape[1])
    
    var_input  = f.createVariable('frp', 'float',  ('time', 'nlat', 'nlon'))
    var_lat = f.createVariable('grid_lat', 'float', ('nlat', 'nlon'))
    var_lon = f.createVariable('grid_lon', 'float', ('nlat', 'nlon'))
    var_time  = f.createVariable('time', str, ('time',))
    
    var_input[:]  = data
    var_lat[:] = yt_grid
    var_lon[:] = xt_grid
    var_time[:] = np.array(time).astype(str)
    f.close()

    return 0

# -*- coding: utf-8 -*-
"""
Author: whung

This script is used to generate predicted fire map as model final product.
"""

import numpy as np
from netCDF4 import Dataset

import warnings
warnings.simplefilter(action='ignore')



def fire_mapper(f_ori, f_predic, opt_corr, scale_opt, scale_val):

    '''Reading Data'''
    ## initial
    readin = Dataset(f_ori)
    lat    = readin['grid_lat'][:]
    lon    = readin['grid_lon'][:]
    readin.close()
    
    ## prediction
    readin    = Dataset(f_predic)
    frame_lat = readin['frame_lat'][:]                     # frame x lat x lon
    frame_lon = readin['frame_lon'][:]                     # frame x lat x lon
    
    if opt_corr == 0:
        frame_fire = readin['frame_predic_post'][:, :, :, 0]    # frame x lat x lon 
    elif opt_corr == 1:
        frame_fire = readin['frame_predic_corr'][:, :, :, 0]    # frame x lat x lon
    readin.close()
    #print(frame_fire.shape, frame_lat.shape, frame_lon.shape)
    
    
    
    '''Fire Mapping'''
    X_fire = frame_fire[frame_fire!=0]
    X_lat  = frame_lat[frame_fire!=0]
    X_lon  = frame_lon[frame_fire!=0]
    
    fire_map = np.zeros(lat.shape)   # lat x lon
    
    for j in np.arange(len(X_fire)):
        index = np.squeeze(np.argwhere((lat==X_lat[j])&(lon==X_lon[j])))
        fire_map[index[0], index[1]] = np.nanmax([fire_map[index[0], index[1]], X_fire[j]])
        del index
    del [X_fire, X_lon, X_lat]
    
    if scale_opt == 1:
        fire_map = fire_map*scale_val
    
    print('Valid fires:', np.argwhere(fire_map!=0).shape)
    
    
    
    '''Write netCDF file'''
    ## spatial map
    f = Dataset(f_predic, 'a')
    f.createDimension('nlat', lat.shape[0])
    f.createDimension('nlon', lat.shape[1])
    var_lat  = f.createVariable('grid_lat', 'float', ('nlat', 'nlon'))
    var_lon  = f.createVariable('grid_lon', 'float', ('nlat', 'nlon'))
    var_fire = f.createVariable('grid_predic', 'float', ('time', 'nlat', 'nlon'))
    
    var_lat[:]  = lat
    var_lon[:]  = lon
    var_fire[:] = fire_map
    f.close()

# -*- coding: utf-8 -*-
"""
Author: Wei-Ting Hung

This script is used to generate model input files. Pre-process gridded fire map needed.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy import io, ndimage
from metpy.units import units
from metpy.calc import wind_direction
from netCDF4 import Dataset
from datetime import datetime
import os

import warnings
warnings.simplefilter(action='ignore')



def file_finder(item, date, hour):
    if item == 'elv':
        return '/groups/ESS/whung/Alldata/GMCollections/ELEV_4X_1Y_V1_Yamazaki.nc'
    elif item == 'ast':
        return '/groups/ESS/whung/Alldata/VIIRS_AST/VIIRS_AST_2020_grid3km.nc'
    elif item == 'fh':
        return '/groups/ESS/whung/Alldata/GLAD_FH/GLAD_FH_grid3km_2020.nc'
    elif item == 'vhi':
        week     = datetime(int(date[:4]), int(date[4:6]), int(date[6:])).isocalendar().week
        lastweek = datetime(int(date[:4]), 12, 31).isocalendar().week
        if week == lastweek:
            return '/groups/ESS/whung/Alldata/VIIRS_VHP/npp/VHP.G04.C07.npp.P2020'+('%03d'%(lastweek-1))+'.VH.nc', '/groups/ESS/whung/Alldata/VIIRS_VHP/j01/VHP.G04.C07.j01.P2020'+('%03d'%(lastweek-1))+'.VH.nc'
        else:
            return '/groups/ESS/whung/Alldata/VIIRS_VHP/npp/VHP.G04.C07.npp.P2020'+('%03d'%week)+'.VH.nc', '/groups/ESS/whung/Alldata/VIIRS_VHP/j01/VHP.G04.C07.j01.P2020'+('%03d'%week)+'.VH.nc'
    elif (item == 't2m') or (item == 't2m_f'):
        return '/groups/ESS/whung/Alldata/HRRR_2d/nc_file/'+date+'/hrrr_t2m_'+date+'_'+hour+'.nc'
    elif (item == 'sh2') or (item == 'sh2_f'):
        return '/groups/ESS/whung/Alldata/HRRR_2d/nc_file/'+date+'/hrrr_sh2_'+date+'_'+hour+'.nc'
    elif (item == 'tp') or (item == 'tp_f'):
        return '/groups/ESS/whung/Alldata/HRRR_2d/nc_file/'+date+'/hrrr_prate_'+date+'_'+hour+'.nc'
    elif (item == 'wd') or (item == 'wd_f') or (item == 'ws') or (item == 'ws_f'):
        return '/groups/ESS/whung/Alldata/HRRR_2d/nc_file/'+date+'/hrrr_u10_'+date+'_'+hour+'.nc', \
               '/groups/ESS/whung/Alldata/HRRR_2d/nc_file/'+date+'/hrrr_v10_'+date+'_'+hour+'.nc'
    elif (item == 'midws') or (item == 'midws_prefire'):
        return '/groups/ESS/whung/Alldata/HRRR_2d/nc_file/'+date+'/hrrr_u10_'+date+'_'+hour+'.nc', \
               '/groups/ESS/whung/Alldata/HRRR_2d/nc_file/'+date+'/hrrr_v10_'+date+'_'+hour+'.nc'

def mapping(xgrid, ygrid, data, xdata, ydata, map_method, fill_value):
    output = griddata((xdata, ydata), data, (xgrid, ygrid), method=map_method, fill_value=fill_value)
    return output

def wind_conversion(LAT, U, V):      # from HRRR FAQ (https://rapidrefresh.noaa.gov/faq/HRRR.faq.html)
    rotcon_p  = 0.622515
    lon_xx_p  = -97.5
    lat_tan_p = 38.5

    angle = rotcon_p*(LAT-lat_tan_p)*0.017453    # LAMBERT CONFORMAL PROJECTION
    #angle = rotcon_p*(LON-lon_xx_p)*0.017453     # CARTESIAN X-AXIS
    sinx2 = np.sin(angle)
    cosx2 = np.cos(angle)

    U_new = cosx2*U + sinx2*V
    V_new = (-1)*sinx2*U + cosx2*V
    return U_new, V_new

def normalization(var, NN):
    a = []
    b = []
    for i in np.arange(NN):
        a = np.append(a, np.nanmax(var[:, :, :, i])-np.nanmin(var[:, :, :, i]))
        b = np.append(b, np.nanmin(var[:, :, :, i]))
    return a, b


def main_driver(forecast_hour, f_input, f_output, lat_lim, lon_lim):

    '''Global Settings'''
    ## contants
    frp_thres  = 15      # fires with low FRP will be removed
    fsize      = 2       # extend fire grid
    
    ## variable list
    firelist   = ['frp']
    geolist    = ['elv', 'ast', 'doy', 'time']
    veglist    = ['fh', 'vhi']
    metlist    = ['t2m', 'sh2', 'tp', 'wd', 'ws']   #, 't2m_f', 'sh2_f', 'tp_f', 'wd_f', 'ws_f']
    #flamelist  = ['midws', 'midws_prefire']
    INPUTLIST  = firelist + geolist + veglist + metlist   # + flamelist



    '''Reading Fire Map & Input Settings'''
    if forecast_hour == 0:     # initial fire
        readin  = Dataset(f_input)
        time    = readin['time'][0]
        LAT     = readin['grid_lat'][:]
        LON     = readin['grid_lon'][:]
        FRP     = readin['frp'][0, :, :]
        readin.close()
        del readin
    else:
        readin  = Dataset(f_input)
        time    = readin['time'][0]
        LAT     = readin['grid_lat'][:]
        LON     = readin['grid_lon'][:]
        FRP     = readin['grid_predic'][0, :, :]
        readin.close()
        del readin
    
    NN = len(INPUTLIST)
    tt = time          # yyyymmddHHMM
    dd = time[:8]      # yyyymmdd
    hh = time[8:10]    # HH
    #print(tt, dd, hh)
    #print(lat_lim, lon_lim, FRP.shape)



    INPUT  = np.empty([LAT.shape[0], LAT.shape[1], NN])
    print('---- Selecting data...', dd, hh+'Z')



    '''Reading Input Variables'''
    ## frp, lat, lon
    INPUT[:, :, INPUTLIST.index('frp')] = np.copy(FRP)
    
    
    ## elv
    filename = file_finder('elv', dd, hh)
    
    if os.path.isfile(filename) == False:
        print('----Incorrect input file:', 'elv', 'Terminated!')
        exit()
    
    readin = Dataset(filename)
    yt     = readin['lat'][:]
    xt     = readin['lon'][:]
    index1 = np.squeeze(np.argwhere((yt>=lat_lim[0]) & (yt<=lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt>=lon_lim[0]) & (xt<=lon_lim[1])))
    
    yt   = yt[index1[0]-1:index1[-1]+2]
    xt   = xt[index2[0]-1:index2[-1]+2]
    data = readin['data'][index1[0]-1:index1[-1]+2, index2[0]-1:index2[-1]+2]
    data[data<0] = 0
    
    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(LAT, LON, data.flatten(), yt_grid.flatten(), xt_grid.flatten(), 'linear', np.nan)
    data_grid[data_grid<0] = np.nan
    
    INPUT[:, :, INPUTLIST.index('elv')] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]
    
    
    ## ast 
    filename = file_finder('ast', dd, hh)
    
    if os.path.isfile(filename) == False:
        print('----Incorrect input file:', 'ast', 'Terminated!')
        exit()
    
    readin = Dataset(filename)
    yt     = np.flip(readin['lat'][:, 0])
    xt     = readin['lon'][0, :]
    yt     = np.round(yt, 3)
    xt     = np.round(xt, 3)
    index1 = np.squeeze(np.argwhere((yt>=lat_lim[0]) & (yt<=lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt>=lon_lim[0]) & (xt<=lon_lim[1])))
    
    data = np.squeeze(readin['ast'][0, :, :])
    data = np.flipud(data)
    
    # ast, hour x lat x lon
    yt   = yt[index1[0]-1:index1[-1]+2]
    xt   = xt[index2[0]-1:index2[-1]+2]
    data = data[index1[0]-1:index1[-1]+2, index2[0]-1:index2[-1]+2]
    
    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(LAT, LON, data.flatten(), yt_grid.flatten(), xt_grid.flatten(), 'nearest', np.nan)
    
    INPUT[:, :, INPUTLIST.index('ast')] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]
    
    
    ## doy 
    data    = np.empty(LAT.shape)
    data[:] = int(pd.to_datetime(dd).strftime('%-j'))
    
    INPUT[:, :, INPUTLIST.index('doy')] = np.copy(data).astype(int)
    del data
    
    
    ## time
    readin = Dataset('/groups/ESS/whung/Alldata/TIMEZONES/timezones_voronoi_1x1.nc')
    yt     = np.flip(readin['lat'][:])
    xt     = readin['lon'][:]
    index1 = np.squeeze(np.argwhere((yt>=lat_lim[0]) & (yt<=lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt>=lon_lim[0]) & (xt<=lon_lim[1])))
    
    offset = np.squeeze(readin['UTC_OFFSET'][0, :, :])
    offset = np.flipud(offset)
    
    yt     = yt[index1[0]-1:index1[-1]+2]
    xt     = xt[index2[0]-1:index2[-1]+2]
    offset = offset[index1[0]-1:index1[-1]+2, index2[0]-1:index2[-1]+2]
    
    xt_grid, yt_grid = np.meshgrid(xt, yt)
    offset_grid = mapping(LAT, LON, offset.flatten(), yt_grid.flatten(), xt_grid.flatten(), 'nearest', np.nan)
            
    data = int(hh) + offset_grid
    data[data<0]  = data[data<0]+24
    data[data>24] = data[data>24]-24
    
    INPUT[:, :, INPUTLIST.index('time')] = np.copy(data).astype(int)

    readin.close()
    del [readin, xt, xt_grid, yt, yt_grid, offset, offset_grid, data]
    
    
    ## fh 
    filename = file_finder('fh', dd, hh)
    
    if os.path.isfile(filename) == False:
        print('----Incorrect input file:', 'fh', 'Terminated!')
        exit()
    
    readin = Dataset(filename)
    yt     = np.flip(readin['lat'][:, 0])
    xt     = readin['lon'][0, :]
    yt     = np.round(yt, 3)
    xt     = np.round(xt, 3)
    index1 = np.squeeze(np.argwhere((yt>=lat_lim[0]) & (yt<=lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt>=lon_lim[0]) & (xt<=lon_lim[1])))
    
    data = np.squeeze(readin['forest_canopy_height'][:])
    data = np.flipud(data)
    
    # fh, lat x lon
    yt   = yt[index1[0]-1:index1[-1]+2]
    xt   = xt[index2[0]-1:index2[-1]+2]
    data = data[index1[0]-1:index1[-1]+2, index2[0]-1:index2[-1]+2]
    
    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(LAT, LON, data.flatten(), yt_grid.flatten(), xt_grid.flatten(), 'linear', np.nan)
    data_grid[data_grid<0] = np.nan
    data_grid[np.isnan(data_grid)] = 0
    
    INPUT[:, :, INPUTLIST.index('fh')] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]
    
    
    ## vhi 
    filename_npp, filename_j01 = file_finder('vhi', dd, hh)
    
    if (os.path.isfile(filename_npp) == False) or (os.path.isfile(filename_j01) == False):
        print('----Incorrect input file:', 'vhi', 'Terminated!')
        exit()
    
    # npp
    readin = Dataset(filename_npp)
    yt     = np.flip(readin['latitude'][:])
    xt     = readin['longitude'][:]
    yt     = np.round(yt, 3)
    xt     = np.round(xt, 3)
    index1 = np.squeeze(np.argwhere((yt>=lat_lim[0]) & (yt<=lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt>=lon_lim[0]) & (xt<=lon_lim[1])))
    
    yt      = yt[index1[0]-1:index1[-1]+2]
    xt      = xt[index2[0]-1:index2[-1]+2]
    vci_npp = np.flipud(np.asarray(readin['VCI'][:]))
    tci_npp = np.flipud(np.asarray(readin['TCI'][:]))
    vci_npp = vci_npp[index1[0]-1:index1[-1]+2, index2[0]-1:index2[-1]+2]
    tci_npp = tci_npp[index1[0]-1:index1[-1]+2, index2[0]-1:index2[-1]+2]
    vci_npp[vci_npp==-999] = np.nan
    tci_npp[tci_npp==-999] = np.nan
    readin.close()
    del readin
    
    # j01
    readin  = Dataset(filename_j01)
    vci_j01 = np.flipud(np.asarray(readin['VCI'][:]))
    tci_j01 = np.flipud(np.asarray(readin['TCI'][:]))
    vci_j01 = vci_j01[index1[0]-1:index1[-1]+2, index2[0]-1:index2[-1]+2]
    tci_j01 = tci_j01[index1[0]-1:index1[-1]+2, index2[0]-1:index2[-1]+2]
    vci_j01[vci_j01==-999] = np.nan
    tci_j01[tci_j01==-999] = np.nan
    readin.close()
    del readin
    
    # combine two satellites
    vci = np.nanmean([vci_npp, vci_j01], axis=0)
    tci = np.nanmean([tci_npp, tci_j01], axis=0)
    vci = vci/100
    tci = tci/100
    del [vci_npp, tci_npp, vci_j01, tci_j01]
    
    # calculate vhi
    data = 0.3*vci + 0.7*tci
    data[np.isnan(data)] = -999
    del [vci, tci]
    
    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(LAT, LON, data.flatten(), yt_grid.flatten(), xt_grid.flatten(), 'nearest', np.nan)
    data_grid[data_grid==-999] = np.nan
    
    INPUT[:, :, INPUTLIST.index('vhi')] = np.copy(data_grid)
    del [filename_npp, filename_j01, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]
    
    
    ## t2m
    filename = file_finder('t2m', dd, hh)
    
    if os.path.isfile(filename) == False:
        print('----Incorrect input file:', 't2m', 'Terminated!')
        exit()
    
    readin = Dataset(filename)
    yt     = readin['latitude'][:]
    xt     = readin['longitude'][:]-360
    data   = readin['t2m'][:]
    
    data_grid = mapping(LAT, LON, data.flatten(), yt.flatten(), xt.flatten(), 'linear', np.nan)
    data_grid[data_grid<0] = np.nan
    
    INPUT[:, :, INPUTLIST.index('t2m')] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, data, data_grid]
    
    
    ## sh2
    filename = file_finder('sh2', dd, hh)
    
    if os.path.isfile(filename) == False:
        print('----Incorrect input file:', 'sh2', 'Terminated!')
        exit()
    
    readin = Dataset(filename)
    yt     = readin['latitude'][:]
    xt     = readin['longitude'][:]-360
    data   = readin['sh2'][:]
    
    data_grid = mapping(LAT, LON, data.flatten(), yt.flatten(), xt.flatten(), 'linear', np.nan)
    data_grid[data_grid<0] = np.nan
    
    INPUT[:, :, INPUTLIST.index('sh2')] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, data, data_grid]
    
    
    ## tp 
    filename = file_finder('tp', dd, hh)
    
    if os.path.isfile(filename) == False:
        print('----Incorrect input file:', 'tp', 'Terminated!')
        exit()
    
    readin = Dataset(filename)
    yt     = readin['latitude'][:]
    xt     = readin['longitude'][:]-360
    data   = readin['prate'][:]
    data   = data*3600      # kg m-2 s-1 -> hourly accumulation
    
    data_grid = mapping(LAT, LON, data.flatten(), yt.flatten(), xt.flatten(), 'linear', np.nan)
    data_grid[data_grid<0] = np.nan
    
    INPUT[:, :, INPUTLIST.index('tp')] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, data, data_grid]
    
    
    ## ws, wd 
    filename_u, filename_v = file_finder('ws', dd, hh)
    
    if (os.path.isfile(filename_u) == False) or (os.path.isfile(filename_v) == False):
        print('----Incorrect input file:', 'ws, wd', 'Terminated!')
        exit()
    
    readin_u = Dataset(filename_u)
    readin_v = Dataset(filename_v)
    yt       = readin_u['latitude'][:]
    xt       = readin_u['longitude'][:]-360
    u        = readin_u['u10'][:]
    v        = readin_v['v10'][:]
    
    u_new, v_new = wind_conversion(yt, u, v)
    
    u_grid = mapping(LAT, LON, u_new.flatten(), yt.flatten(), xt.flatten(), 'linear', np.nan)
    v_grid = mapping(LAT, LON, v_new.flatten(), yt.flatten(), xt.flatten(), 'linear', np.nan)
    
    # for canopy wind calculation
    UGRID = np.copy(u_grid)
    VGRID = np.copy(v_grid)
    
    # ws, lat x lon
    ws_grid = np.sqrt(u_grid**2+v_grid**2)
    
    # wd, lat x lon
    u_grid  = units.Quantity(u_grid, 'm/s')
    v_grid  = units.Quantity(v_grid, 'm/s')
    wd_grid = wind_direction(u_grid, v_grid, convention='from')
    
    INPUT[:, :, INPUTLIST.index('ws')] = np.copy(ws_grid)
    INPUT[:, :, INPUTLIST.index('wd')] = np.copy(wd_grid)
    del [filename_u, filename_v, readin_u, readin_v, yt, xt, u, v, u_new, v_new, u_grid, v_grid, ws_grid, wd_grid]
    
    
    ### t2m_f 
    #dd_f     = pd.to_datetime(tt)+pd.Timedelta(hours=1)
    #filename = file_finder('t2m', dd_f.strftime('%Y%m%d'), dd_f.strftime('%H'))
    #
    #if os.path.isfile(filename) == False:
    #    print('----Incorrect input file:', 't2m_f', 'Terminated!')
    #    exit()
    #
    #readin = Dataset(filename)
    #yt     = readin['latitude'][:]
    #xt     = readin['longitude'][:]-360
    #data   = readin[varname][:]
    #
    #data_grid = mapping(LAT, LON, data.flatten(), yt.flatten(), xt.flatten(), 'linear', np.nan)
    #data_grid[data_grid<0] = np.nan
    #
    #INPUT[:, :, INPUTLIST.index('t2m_f')] = np.copy(data_grid)
    #
    #readin.close()
    #del [filename, readin, dd_f, yt, xt, data, data_grid]
    #
    #
    ### sh2_f
    #dd_f     = pd.to_datetime(tt)+pd.Timedelta(hours=1)
    #filename = file_finder('sh2', dd_f.strftime('%Y%m%d'), dd_f.strftime('%H'))
    #
    #if os.path.isfile(filename) == False:
    #    print('----Incorrect input file:', 'sh2_f', 'Terminated!')
    #    exit()
    #
    #readin = Dataset(filename)
    #yt     = readin['latitude'][:]
    #xt     = readin['longitude'][:]-360
    #data   = readin[varname][:]
    #
    #data_grid = mapping(LAT, LON, data.flatten(), yt.flatten(), xt.flatten(), 'linear', np.nan)
    #data_grid[data_grid<0] = np.nan
    #
    #INPUT[:, :, INPUTLIST.index('sh2_f')] = np.copy(data_grid)
    #
    #readin.close()
    #del [filename, readin, dd_f, yt, xt, data, data_grid]
    #
    #
    ### tp 
    #dd_f     = pd.to_datetime(tt)+pd.Timedelta(hours=1)
    #filename = file_finder('tp', dd_f.strftime('%Y%m%d'), dd_f.strftime('%H'))
    #
    #if os.path.isfile(filename) == False:
    #    print('----Incorrect input file:', 'tp', 'Terminated!')
    #    exit()
    #
    #readin = Dataset(filename)
    #yt     = readin['latitude'][:]
    #xt     = readin['longitude'][:]-360
    #data   = readin['prate'][:]
    #data   = data*3600      # kg m-2 s-1 -> hourly accumulation
    #
    #data_grid = mapping(LAT, LON, data.flatten(), yt.flatten(), xt.flatten(), 'linear', np.nan)
    #data_grid[data_grid<0] = np.nan
    #
    #INPUT[:, :, INPUTLIST.index('tp_f')] = np.copy(data_grid)
    #
    #readin.close()
    #del [filename, readin, yt, xt, data, data_grid]
    #
    #
    ### ws_f, wd_f
    #dd_f                   = pd.to_datetime(tt)+pd.Timedelta(hours=1)
    #filename_u, filename_v = file_finder('ws', dd_f.strftime('%Y%m%d'), dd_f.strftime('%H'))
    #
    #if (os.path.isfile(filename_u) == False) or (os.path.isfile(filename_v) == False):
    #    print('----Incorrect input file:', 'ws, wd', 'Terminated!')
    #    exit()
    #
    #readin_u = Dataset(filename_u)
    #readin_v = Dataset(filename_v)
    #yt       = readin_u['latitude'][:]
    #xt       = readin_u['longitude'][:]-360
    #u        = readin_u['u10'][:]
    #v        = readin_v['v10'][:]
    #
    #u_new, v_new = wind_conversion(yt, u, v)
    #
    #u_grid = mapping(LAT, LON, u_new.flatten(), yt.flatten(), xt.flatten(), 'linear', np.nan)
    #v_grid = mapping(LAT, LON, v_new.flatten(), yt.flatten(), xt.flatten(), 'linear', np.nan)
    #
    ## ws, lat x lon
    #ws_grid = np.sqrt(u_grid**2+v_grid**2)
    #
    ## wd, lat x lon
    #u_grid  = units.Quantity(u_grid, 'm/s')
    #v_grid  = units.Quantity(v_grid, 'm/s')
    #wd_grid = wind_direction(u_grid, v_grid, convention='from')
    #
    #INPUT[:, :, INPUTLIST.index('ws_f')] = np.copy(ws_grid)
    #INPUT[:, :, INPUTLIST.index('wd_f')] = np.copy(wd_grid)
    #
    #readin.close()
    #del [dd_f, filename_u, filename_v, readin_u, readin_v, yt, xt, u, v, u_new, v_new, u_grid, v_grid, ws_grid, wd_grid]
    
    
    print('Start framing... Original data:', INPUT.shape)
    
    
    '''Fire Framing'''
    XX = LAT.shape[0]
    YY = LAT.shape[1]
    
    total = 0
    skip  = 0
    
    ## fire mask
    MASK = np.zeros(LAT.shape)
    MASK[FRP != 0] = 1
    
    
    ## fire location
    lw, num = ndimage.label(MASK)
    lw      = lw.astype(float)
    lw[lw==0] = np.nan
    
    
    ## fire frame
    for j in np.arange(1, num+1, 1):
        total = total + 1
        index = np.argwhere(lw==j)
        #print(j, index.shape)
    
        if index.shape[0] == 1:
            loc = np.squeeze(index)
        else:
            xx  = np.mean(LAT[index[:, 0], index[:, 1]])
            yy  = np.mean(LON[index[:, 0], index[:, 1]])
            dis = (LAT-xx)**2+(LON-yy)**2
            loc = np.squeeze(np.argwhere(dis==np.min(dis)))
            #print(xx, yy, np.min(dis))
            #print(dis[dis==np.min(dis)])
            del [xx, yy, dis]
    
        if loc.size > 2:
            loc = loc[0, :]
    
        if (loc[0]-fsize < 0) or (loc[0]+fsize > XX) or (loc[1]-fsize < 0) or (loc[1]+fsize > YY):
            # skip fires close to the boundary
            skip = skip + 1
            continue
    
        X_fire = INPUT[loc[0]-fsize:loc[0]+fsize+1, loc[1]-fsize:loc[1]+fsize+1, :]
        X_lat  = LAT[loc[0]-fsize:loc[0]+fsize+1, loc[1]-fsize:loc[1]+fsize+1]
        X_lon  = LON[loc[0]-fsize:loc[0]+fsize+1, loc[1]-fsize:loc[1]+fsize+1]
    
        if 'INPUTFRAME' in locals():
            INPUTFRAME  = np.append(INPUTFRAME, np.expand_dims(X_fire, axis=0), axis=0)
            LATFRAME    = np.append(LATFRAME,   np.expand_dims(X_lat, axis=0), axis=0)
            LONFRAME    = np.append(LONFRAME,   np.expand_dims(X_lon, axis=0), axis=0)
        else:
            INPUTFRAME  = np.expand_dims(X_fire, axis=0)
            LATFRAME    = np.expand_dims(X_lat, axis=0)
            LONFRAME    = np.expand_dims(X_lon, axis=0)
    
        del [index, loc, X_fire, X_lat, X_lon]
    del [lw, num, INPUT, FRP, LAT, LON, MASK]
    
    
    
    if 'INPUTFRAME' in locals():
    #    print(INPUTFRAME.shape)
    #    for i in np.arange(len(INPUTLIST)):
    #        print(INPUTLIST[i], np.argwhere(np.isnan(INPUTFRAME[:, :, :, i])).shape)
        pass
    else:
        print('---- '+tt+' no available frames.')
        exit()
    
    
    
    ## remove frames with NaN
    index = []
    for i in np.arange(NN):
        X = np.sum(INPUTFRAME[:, :, :, i], axis=(1, 2))
        #print(np.min(X), np.argwhere(np.isnan(X)).shape)
        index = np.append(index, np.squeeze(np.argwhere(np.isnan(X))))
        del X
    
    index = index.astype(int)
    
    if index.size != 0:
        INPUTFRAME = np.delete(INPUTFRAME,  index, axis=0)
        LATFRAME   = np.delete(LATFRAME,  index, axis=0)
        LONFRAME   = np.delete(LONFRAME,  index, axis=0)
    nan = index.size
    del index
    
    
    ## remove isolated small fires
    fire_map   = np.copy(INPUTFRAME[:, :, :, 0])
    fire_map[fire_map!=0] = 1
    fire_count = np.sum(fire_map, axis=(1, 2))
    fire_max   = np.max(INPUTFRAME[:, :, :, 0], axis=(1, 2))
    index      = np.squeeze(np.argwhere((fire_count==1) & (fire_max<frp_thres)))
    
    if index.size != 0:
        INPUTFRAME  = np.delete(INPUTFRAME,  index, axis=0)
        LATFRAME   = np.delete(LATFRAME,  index, axis=0)
        LONFRAME   = np.delete(LONFRAME,  index, axis=0)
    small = index.size
    del [fire_map, fire_count, fire_max, index]
    
    
    ## remove frames with water bodies (AST=17)
    index = np.argwhere(INPUTFRAME[:, :, :, INPUTLIST.index('ast')]==17)[:, 0]
    index = np.unique(index)
    
    if index.size != 0:
        INPUTFRAME  = np.delete(INPUTFRAME,  index, axis=0)
        LATFRAME   = np.delete(LATFRAME,  index, axis=0)
        LONFRAME   = np.delete(LONFRAME,  index, axis=0)
    water = index.size
    del index
    
    
    
    print('---- Fire framing complete!')
     
    if INPUTFRAME.shape[1] == 0:
        print('----No available frames!')
        exit()
    else:
        print('All:', total, 'Skip:', skip, 'NaN:', nan, 'Water', water)
        #print('All:', total, 'Skip:', skip, 'NaN:', nan, 'Small', small, 'Water', water)
        print('Final:', INPUTFRAME.shape)
    
    
    INPUTFRAME_ori = np.copy(INPUTFRAME)
    
    
    '''Scaling'''
    for X in ['frp', 'tp']:   #, 'tp_f']:
        i  = INPUTLIST.index(X)
        X1 = np.copy(INPUTFRAME[:, :, :, i])
        X1[X1!=0] = np.log(X1[X1!=0])
        INPUTFRAME[:, :, :, i] = np.copy(X1)
        del [i, X1]
    
    for X in ['sh2', 'ws']:   #, 'sh2_f', 'ws_f']:  #, 'flamews']:
        i  = INPUTLIST.index(X)
        X1 = np.copy(INPUTFRAME[:, :, :, i])
        X1 = np.sqrt(X1)
        INPUTFRAME[:, :, :, i] = np.copy(X1)
        del [i, X1]
    
    
    for X in ['fh', 'elv']:
        i  = INPUTLIST.index(X)
        X1 = np.copy(INPUTFRAME[:, :, :, i])
        X1 = (X1)**(1/3)
        INPUTFRAME[:, :, :, i] = np.copy(X1)
        del [i, X1]
    
    
    
    '''Normalization'''
    ## normalization coef
    coef = pd.read_csv('./model/model_normalization_coef.txt')
    coef = np.array(coef)
    a    = coef[0, np.append(0, np.arange(3, 14))]
    b    = coef[1, np.append(0, np.arange(3, 14))]
    
    for i in np.arange(NN):
        X = np.copy(INPUTFRAME[:, :, :, i])
        X = (X-b[i])/a[i]
        INPUTFRAME[:, :, :, i] = np.copy(X)
        del X
    
    
    
    '''Writing Model Input'''
    INPUTFRAME[np.isnan(INPUTFRAME)]   = -999
    
    f = Dataset(f_output, 'w')
    f.createDimension('time', 1)
    f.createDimension('flen', INPUTFRAME.shape[0])
    f.createDimension('xlen', INPUTFRAME.shape[1])
    f.createDimension('ylen', INPUTFRAME.shape[2])
    f.createDimension('num_input', NN)
    var_time   = f.createVariable('time', str, ('time',))
    var_input0 = f.createVariable('input_noscale', 'float',  ('flen', 'xlen', 'ylen', 'num_input'))
    var_input  = f.createVariable('input', 'float',  ('flen', 'xlen', 'ylen', 'num_input'))
    var_lat    = f.createVariable('frame_lat', 'float',  ('flen', 'xlen', 'ylen'))
    var_lon    = f.createVariable('frame_lon', 'float',  ('flen', 'xlen', 'ylen'))
    var_list   = f.createVariable('INPUTLIST', str, ('num_input',))
    
    var_time[:]   = np.array(time).astype(str)
    var_input0[:] = INPUTFRAME_ori
    var_input[:]  = INPUTFRAME
    var_lat[:]    = LATFRAME
    var_lon[:]    = LONFRAME
    var_list[:]   = np.array(INPUTLIST).astype(str)
    f.close()

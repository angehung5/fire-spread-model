# -*- coding: utf-8 -*-
"""
Author: whung

This script is used for gridded FRP pre-processing.

DATA SOURCE: RAVE
"""

import numpy as np
from netCDF4 import Dataset
import os

import warnings
warnings.simplefilter(action='ignore')



def preprocessor(filename, frp_option, time, lat_lim, lon_lim, plot_option):
    f_output   = './input/'+time+'/'+filename+'.'+time+'.nc'

    date = time[:8]
    hour = time[8:10]
    
    if not os.path.exists('./input/'+time):
        os.makedirs('./input/'+time)


    '''Reading Data'''
    if frp_option == 0:    # RAVE
        f_ori = '/groups/ESS/whung/Alldata/RAVE_3km/Hourly_Emissions_3km_'+date+'0000_'+date+'2300.nc'
        hour_index = int(hour)
    
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
        print('No available or unknown FRP source. Terminated!')
        exit()


    
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
    


    if plot_option == 1:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.basemap import Basemap

        '''Spatial map'''
        cmap = cm.get_cmap('jet').copy()
        cmap.set_over('#9400D3')
        
        fig, ax = plt.subplots(figsize=(18, 12))    # unit=100pixel
        h = ax.get_position()
        ax.set_position([h.x0-0.04, h.y0, h.width+0.06, h.height+0.06])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(3)
        
        plt.title(date+' '+hour+'Z Fire Radiative Power - Observation', fontsize=28)
        
        m = Basemap(llcrnrlon=lon_lim[0],urcrnrlon=lon_lim[-1],llcrnrlat=lat_lim[0],urcrnrlat=lat_lim[-1], projection='mill')
        m.drawcoastlines(color='k', linewidth=1)
        m.drawcountries(color='k', linewidth=1)
        m.drawstates(color='k', linewidth=1)
        m.drawmeridians(np.arange(lon_lim[0], lon_lim[-1]+1, 10), color='none', labels=[0,0,0,1], fontsize=28)
        m.drawparallels(np.arange(lat_lim[0], lat_lim[-1]+1, 5), color='none', labels=[1,0,0,0], fontsize=28)
        
        x, y = m(xt_grid[data!=0], yt_grid[data!=0])
        cs   = m.scatter(x, y, marker='o', c=data[data!=0], s=120, edgecolor='k', cmap=cmap, vmin=0, vmax=200)
        
        # colorbar
        cbax = fig.add_axes([h.x0-0.04, h.y0+0.02, h.width+0.04, 0.02])
        cb = plt.colorbar(cs, extend='max', orientation='horizontal', cax=cbax)
        cb.set_ticks(np.arange(0, 200+1, 20))
        cb.set_label('FRP (MW)', fontsize=28, fontweight='bold')
        cb.ax.tick_params(labelsize=28)
        
        plt.savefig(f_output[:-3]+'.obs.jpg')
        plt.close()
        del [fig, ax, h, m, x, y, cmap, cs, cb, cbax]
# -*- coding: utf-8 -*-
"""
Author: whung

This script is used for simple model evaluation. Two sections are included:
 1) Statistical metrix
 2) Plotting: FRP scatter plot, spatital map
"""

import numpy as np
import pandas as pd
import seaborn as sns
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os, fnmatch, sys
from mpl_toolkits.basemap import Basemap

import warnings
warnings.simplefilter(action='ignore')



def fire_flitering(lst, lat, lon, frp, dis_thres):
    mask = np.zeros_like(frp, dtype=bool)
    for x, y in lst:
        dis = np.sqrt((lat-x)**2+(lon-y)**2)
        mask |= dis <= dis_thres

    return np.array(frp*mask)


def main_driver(f_predic, frp_option, cor_option, lat_lim, lon_lim):

    init_time = f_predic[-19:-7]
    f_output  = f_predic[f_predic.rindex('/')+1:-3]
    frp_thres = 15        # small fires with FRP<15MW are not included
    dis_thres = 15/110    # max spread distance (degree), fires beyond this distance are assumed as NEW fires


    '''Reading Data'''
    ## normalization coef
    readin = pd.read_csv('./model/model_normalization_coef.txt')
    readin = np.array(readin)
    a      = readin[0, -1]
    b      = readin[1, -1]
    del readin
    

    ## initial fire mask
    for files in os.listdir('./input/'+init_time):
        if fnmatch.fnmatch(files, f"*{'f00'}*") and os.path.isfile(os.path.join('./input/'+init_time, files)):
            f_init = './input/'+init_time+'/'+files
    
    readin   = Dataset(f_init)
    fsize    = int((readin['input'][:].shape[1]-1)/2)
    lat_init = np.round(readin['frame_lat'][:, fsize, fsize], 3)
    lon_init = np.round(readin['frame_lon'][:, fsize, fsize], 3)
    frp_init = np.array(np.append(np.expand_dims(lat_init, axis=1), np.expand_dims(lon_init, axis=1), axis=1))
    readin.close()
    del [readin, fsize, lat_init, lon_init]
    
        
    ## forecast
    readin     = Dataset(f_predic)
    time       = readin['time'][0]
    lat        = np.round(readin['grid_lat'][:], 3)
    lon        = np.round(readin['grid_lon'][:], 3)
    frp_pre    = readin['grid_predic'][0, :, :]             # lat x alon
    frame_lat  = np.round(readin['frame_lat'][:], 3)
    frame_lon  = np.round(readin['frame_lon'][:], 3)
    frame_pre  = readin['frame_predic_ori'][:, :, :, 0]     # frame x lat x lon
    
    if cor_option == 0:
        frame_post = readin['frame_predic_post'][:, :, :, 0]    # frame x lat x lon
    elif cor_option == 1:
        frame_post = readin['frame_predic_corr'][:, :, :, 0]    # frame x lat x lon
    readin.close()
    del readin
    
    date = time[:8]
    hour = time[8:10]
        
    
    ## observation - map
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
    
        #data[data<frp_thres] = 0    # remove small fires
        
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
    
        grid_xt, grid_yt = np.meshgrid(xt, yt)
        frp_ori = fire_flitering(frp_init, lat, lon, data, dis_thres)
        frp_ori[frp_ori == 0] = 0.0
        #print(np.argwhere(data!=0).shape)
        #print(np.argwhere(frp_ori!=0).shape)

        readin.close()
        del [readin, xt, yt, data, qa, index1, index2]
    
    else:
        print('No available or unknown FRP source. Terminated!')
        exit()
     
    
    ## observation - frame
    frame_ori    = np.empty(frame_pre.shape)
    frame_ori[:] = np.nan
    
    nframe = frame_pre.shape[0]
    fsize  = int((frame_pre.shape[1]-1)/2)
    
    for i in np.arange(nframe):
        fire_loc = np.squeeze(np.argwhere((grid_yt==frame_lat[i, fsize, fsize]) & (grid_xt==frame_lon[i, fsize, fsize])))
    
        frame_ori[i, :, :] = frp_ori[fire_loc[0]-fsize:fire_loc[0]+fsize+1, fire_loc[1]-fsize:fire_loc[1]+fsize+1]
        del fire_loc
    
    
    
    
    '''Statistical metrix'''
    ## bias  - mean bias
    ## R2    - coefficient of determination
    ## rmse  - root mean square error
    ## mre   - mean relative error
    ## ratio - prediction/observation
    
    xx = frame_ori[(frame_ori!=0)&(frame_post!=0)]
    yy = frame_post[(frame_ori!=0)&(frame_post!=0)]
    
    if xx.size != 0:
        index = np.squeeze(np.argwhere((yy/xx>=np.percentile(np.array(yy/xx), 2.5)) & (yy/xx<=np.percentile(np.array(yy/xx), 97.5))))
        xx    = xx[index]
        yy    = yy[index]
    
    bias  = yy-xx
    R2    = (np.corrcoef(xx, yy)[0, 1])**2
    rmse  = np.sqrt(np.mean((yy-xx)**2))
    mre   = (yy-xx)/xx
    ratio = yy/xx
    del [xx, yy]
    
    
    ## similarity - image similarity
    similarity = np.zeros(nframe)
    for i in np.arange(nframe):
        X      = np.copy(frame_ori[i, :, :])
        X      = (X-np.min(X))/(np.max(X)-np.min(X))
        X_bool = np.zeros(X.shape)
        X_bool[X>np.mean(X)] = 1
    
        Y      = np.copy(frame_pre[i, :, :])
        Y_bool = np.zeros(Y.shape)
        Y_bool[Y>np.mean(Y)] = 1
    
        similarity[i] = np.sum(np.abs(X_bool-Y_bool))
        del [X, X_bool, Y, Y_bool]
    
    
    ## alarm - false alarm rate
    total = frp_ori.size - np.squeeze(np.argwhere((frp_ori==0)&(frp_pre==0))).shape[0]
    true  = np.squeeze(np.argwhere((frp_ori!=0)&(frp_pre!=0))).shape[0]
    false = np.squeeze(np.argwhere((frp_ori==0)&(frp_pre!=0))).shape[0]
    miss  = np.squeeze(np.argwhere((frp_ori!=0)&(frp_pre==0))).shape[0]
    
    
    
    #print('Bias', np.min(bias), '-', np.max(bias), 'avg=', np.mean(bias), '±', np.std(bias))
    #print('R-squared', R2)
    #print('RMSE', rmse)
    #print('RE', np.min(mre), '-', np.max(mre), 'avg=', np.mean(mre), '±', np.std(mre))
    #print('Ratio', np.min(ratio), '-', np.max(ratio), 'avg=', np.mean(ratio), '±', np.std(ratio))
    #print('Similarity', np.min(similarity), '-', np.max(similarity), 'avg=', np.mean(similarity), '±', np.std(similarity))
    #print('True alarm', true/total, 'False alarm', false/total, 'Miss alarm', miss/total)
    
    
    
    '''Saving metrix'''
    HEADER = ['BIAS_MEAN', 'BIAS_STD', 'R2', 'RMSE', 'RE_MEAN', 'RE_STD', 'RATIO_MEAN', 'RATIO_STD', 'SIM_MEAN', 'SIM_STD', 'TRUE_ALARM', 'FALSE_ALARM', 'MISS_ALARM']
    HEADER = ','.join(HEADER)
    FORMAT = '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f', '%.4f'
    OUTPUT = np.empty([1, 13])
    OUTPUT[:, 0]  = np.mean(bias)
    OUTPUT[:, 1]  = np.std(bias)
    OUTPUT[:, 2]  = R2
    OUTPUT[:, 3]  = rmse
    OUTPUT[:, 4]  = np.mean(mre)
    OUTPUT[:, 5]  = np.std(mre)
    OUTPUT[:, 6]  = np.mean(ratio)
    OUTPUT[:, 7]  = np.std(ratio)
    OUTPUT[:, 8]  = np.mean(similarity)
    OUTPUT[:, 9]  = np.std(similarity)
    OUTPUT[:, 10] = true/total
    OUTPUT[:, 11] = false/total
    OUTPUT[:, 12] = miss/total
    np.savetxt('./output/'+init_time+'/fire.evaluation.'+f_output[-16:]+'.txt', OUTPUT, fmt=FORMAT, delimiter=',', header=HEADER, comments='')
    del [HEADER, FORMAT, OUTPUT]
    
    
    
    xx = frame_ori[(frame_ori!=0)&(frame_post!=0)]
    if xx.size == 0:
        print('No matched fires. Skipping plottings!')
        exit()
    
    
    
    '''Plotting'''
    ## scatter plot
    xx = frame_ori[(frame_ori!=0)&(frame_post!=0)]
    yy = frame_post[(frame_ori!=0)&(frame_post!=0)]
    
    index = np.squeeze(np.argwhere((yy/xx>=np.percentile(np.array(yy/xx), 2.5)) & (yy/xx<=np.percentile(np.array(yy/xx), 97.5))))
    xx    = xx[index]
    yy    = yy[index]
    
    fig, ax = plt.subplots(figsize=(12, 12))    # unit=100pixel
    h = ax.get_position()
    ax.set_position([h.x0+0.02, h.y0, h.width+0.02, h.height+0.06])
    ax.tick_params(labelsize=28)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    
    plt.title('Model Evaluation', fontsize=28)
    
    plt.scatter(xx, yy, s=50, color='royalblue')
    plt.xlim([-50, 2000])
    plt.xticks(np.arange(0, 2000+1, 500))
    plt.xlabel('Observation (MW)', fontsize=28)
    plt.ylim([-50, 2000])
    plt.yticks(np.arange(0, 2000+1, 500))
    plt.ylabel('Prediction (MW)', fontsize=28)
    plt.savefig('./output/'+init_time+'/fire.evaluation.'+f_output[-16:]+'.jpg')
    plt.close()
    del [fig, ax, h, axis, xx, yy]
    
    
    
    ## density plot
    xx = frame_ori[(frame_ori!=0)&(frame_post!=0)]
    yy = frame_post[(frame_ori!=0)&(frame_post!=0)]
    
    index = np.squeeze(np.argwhere((yy/xx>=np.percentile(np.array(yy/xx), 2.5)) & (yy/xx<=np.percentile(np.array(yy/xx), 97.5))))
    xx    = xx[index]
    yy    = yy[index]
    
    xx_norm = (xx-np.min(xx))/(np.max(xx)-np.min(xx))
    yy_norm = (yy-np.min(yy))/(np.max(yy)-np.min(yy))
    
    plt.rcParams.update({'font.size': 30})
    fig, ax = plt.subplots(figsize=(12, 12))    # unit=100pixel
    h = ax.get_position()
    ax.set_position([h.x0+0.03, h.y0, h.width+0.02, h.height+0.06])
    ax.tick_params(labelsize=28)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    
    sns.kdeplot(xx, bw_adjust=2, color='lightgrey', alpha=0.8, fill=True, label='Observation')
    sns.kdeplot(yy, bw_adjust=2, color='orange',    alpha=0.7, fill=True, label='Prediction')
    
    ax.set_xlim([-10, 3000])
    ax.set_xticks(np.arange(0, 3000+0.01, 500))
    ax.set_xlabel('FRP (MW)', fontsize=28, fontweight='bold')
    ax.set_ylim([0, 0.005])
    ax.set_yticks(np.arange(0, 0.005+0.00001, 0.001))
    ax.set_yticklabels(np.arange(0, 5+0.1, 1))
    ax.set_ylabel('Density ($10^{-3}$)', fontsize=28, fontweight='bold')
    ax.legend(loc='upper right', prop={'size':26})
    
    plt.savefig('./output/'+init_time+'/fire.kdeplot.'+f_output[-16:]+'.jpg')
    plt.close()
    del [fig, ax, xx, yy, xx_norm, yy_norm]
    
    
    
    ## spatial map - prediction
    cmap = cm.get_cmap('jet').copy()
    cmap.set_over('#9400D3')
    
    fig, ax = plt.subplots(figsize=(18, 12))    # unit=100pixel
    h = ax.get_position()
    ax.set_position([h.x0-0.04, h.y0, h.width+0.06, h.height])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    
    plt.title(date+' '+hour+'Z Fire Radiative Power - Prediction', fontsize=28, y=1.05)
    
    m = Basemap(llcrnrlon=lon_lim[0],urcrnrlon=lon_lim[-1],llcrnrlat=lat_lim[0],urcrnrlat=lat_lim[-1], projection='mill')
    m.drawcoastlines(color='k', linewidth=1)
    m.drawcountries(color='k', linewidth=1)
    m.drawstates(color='k', linewidth=1)
    m.drawmeridians(np.arange(lon_lim[0], lon_lim[-1]+1, 10), color='none', labels=[0,0,0,1], fontsize=28)
    m.drawparallels(np.arange(lat_lim[0], lat_lim[-1]+1, 5), color='none', labels=[1,0,0,0], fontsize=28)
    
    x, y = m(lon[frp_pre!=0], lat[frp_pre!=0])
    cs   = m.scatter(x, y, marker='o', c=frp_pre[frp_pre!=0], s=120, edgecolor='k', cmap=cmap, vmin=0, vmax=200)
    
    # colorbar
    #cbax = fig.add_axes([h.x0-0.04, h.y0+0.02, h.width+0.04, 0.02])
    cb = plt.colorbar(cs, extend='max', orientation='vertical')   #, cax=cbax)
    cb.set_ticks(np.arange(0, 200+1, 20))
    cb.set_label('FRP (MW)', fontsize=28, fontweight='bold')
    cb.ax.tick_params(labelsize=28)
    
    plt.savefig('./output/'+init_time+'/'+f_output+'.jpg')
    plt.close()
    del [fig, ax, h, m, x, y, cmap, cs, cb]   #, cbax]
    
    
    
    ## spatial map - observation
    cmap = cm.get_cmap('jet').copy()
    cmap.set_over('#9400D3')
    
    fig, ax = plt.subplots(figsize=(18, 12))    # unit=100pixel
    h = ax.get_position()
    ax.set_position([h.x0-0.04, h.y0, h.width+0.06, h.height])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
    
    plt.title(date+' '+hour+'Z Fire Radiative Power - Observation', fontsize=28, y=1.05)
    
    m = Basemap(llcrnrlon=lon_lim[0],urcrnrlon=lon_lim[-1],llcrnrlat=lat_lim[0],urcrnrlat=lat_lim[-1], projection='mill')
    m.drawcoastlines(color='k', linewidth=1)
    m.drawcountries(color='k', linewidth=1)
    m.drawstates(color='k', linewidth=1)
    m.drawmeridians(np.arange(lon_lim[0], lon_lim[-1]+1, 10), color='none', labels=[0,0,0,1], fontsize=28)
    m.drawparallels(np.arange(lat_lim[0], lat_lim[-1]+1, 5), color='none', labels=[1,0,0,0], fontsize=28)
    
    x, y = m(lon[frp_ori!=0], lat[frp_ori!=0])
    cs   = m.scatter(x, y, marker='o', c=frp_ori[frp_ori!=0], s=120, edgecolor='k', cmap=cmap, vmin=0, vmax=200)
    
    # colorbar
    #cbax = fig.add_axes([h.x0-0.04, h.y0+0.02, h.width+0.04, 0.02])
    cb = plt.colorbar(cs, extend='max', orientation='vertical')   #, cax=cbax)
    cb.set_ticks(np.arange(0, 200+1, 20))
    cb.set_label('FRP (MW)', fontsize=28, fontweight='bold')
    cb.ax.tick_params(labelsize=28)
    
    plt.savefig('./output/'+init_time+'/'+f_output+'.obs.jpg')
    plt.close()
    del [fig, ax, h, m, x, y, cmap, cs, cb]   #, cbax]

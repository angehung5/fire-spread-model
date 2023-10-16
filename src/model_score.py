# -*- coding: utf-8 -*-
"""
Author: whung

This script is used to generate evaluation plot with forecast hour as x-axis and statistical matrix as y-axis.

IMPORTANT: THIS SCRIPT IS NOT A STANDARD STEP IN FIRE SPREAD MODEL AND IS NOT INCLUDED IN FIRE_MODEL.PY.
"""


import numpy as np
import pandas as pd
import os, fnmatch
import matplotlib.pyplot as plt



def scoreanalysis(time_start, time, forecast_time):
    f_output = './output/'+time_start+'/score.'+time_start+'.'+str(forecast_time)+'hr.jpg'
    hour     = np.arange(1, forecast_time+1, 1)
    header   = ['BIAS_MEAN','BIAS_STD','R2','RMSE','RE_MEAN','RE_STD','RATIO_MEAN','RATIO_STD','SIM_MEAN','SIM_STD','TRUE_ALARM','FALSE_ALARM','MISS_ALARM']


    '''Reading Data'''
    for i in np.arange(time):
        f_input = './output/'+time_start+'/fire.evaluation.'+time_start+'.f'+('%02d'%(i+1))+'.txt'

        if os.path.isfile(f_input) == False:
            fill    = np.empty(len(header))
            fill[:] = np.nan
            if not 'data' in locals():
                data = np.expand_dims(fill, axis=0)
            else:
                data = np.append(data, np.expand_dims(fill, axis=0), axis=0)
            del fill

        else:
            readin = np.loadtxt(f_input, delimiter=',', skiprows=1)
        
            if not 'data' in locals():
                data = np.expand_dims(readin, axis=0)
            else:
                data = np.append(data, np.expand_dims(readin, axis=0), axis=0)
            del readin
    

    if data.shape[0] < len(hour):
        fill    = np.empty([len(hour)-data.shape[0], len(header)])
        fill[:] = np.nan
        data    = np.append(data, fill, axis=0)

    
    bias_mean   = data[:, 0]
    bias_std    = data[:, 1]
    r2          = data[:, 2]
    rmse        = data[:, 3]
    re_mean     = data[:, 4]
    re_std      = data[:, 5]
    ratio_mean  = data[:, 6]
    ratio_std   = data[:, 7]
    sim_mean    = data[:, 8]
    sim_std     = data[:, 9]
    true_alarm  = data[:, 10]*100
    false_alarm = data[:, 11]*100
    miss_alarm  = data[:, 12]*100
    
    sim_mean[np.isnan(bias_mean)] = np.nan      # no matched pairs
    sim_std[np.isnan(bias_mean)]  = np.nan      # no matched pairs
    
    
    
    '''Plotting'''
    plt.rcParams.update({'font.size': 30})
    fig, ax = plt.subplots(3, 1, figsize=(24, 30), sharex=True)
    
    for axis in ['top','bottom','left','right']:
        ax[0].spines[axis].set_linewidth(2.5)
        ax[1].spines[axis].set_linewidth(2.5)
        ax[2].spines[axis].set_linewidth(2.5)
    
    ## BIAS/RMSE
    ax[0].hlines(0, hour[0]-1, hour[-1]+1, linestyle=':', color='k', linewidth=2)
    ax[0].errorbar(hour, bias_mean, yerr=bias_std, color='royalblue', linewidth=4, label='Bias')
    ax[0].plot(hour, rmse, color='orange', linewidth=4, label='RMSE')
    ax[0].set_ylim([-1500, 1500])
    ax[0].set_yticks(np.arange(-1500, 1500+1, 500))
    ax[0].set_ylabel('FRP (MW)')
    ax[0].legend(loc='upper left', ncol=2)
    
    ## R2/SIMILARITY
    #ax_sim = ax[1].twinx()
    h1, = ax[1].plot(hour, r2, color='magenta', linewidth=4, label='R2')
    #h2 = ax_sim.errorbar(hour, sim_mean, yerr=sim_std, color='gold', linewidth=4, label='Similarity')
    h2 = ax[1].errorbar(hour, sim_mean, yerr=sim_std, color='gold', linewidth=4, label='Similarity')

    ax[1].set_ylim([0, 1])
    ax[1].set_yticks(np.arange(0, 1+0.1, 0.2))
    ax[1].set_ylabel('R2/Similarity')
    #ax_sim.set_ylim([0, 20])
    #ax_sim.set_yticks(np.arange(0, 20+1, 5))
    #ax_sim.set_ylabel('Similarity')
    handle = [h1, h2]
    labels = [h.get_label() for h in handle]
    ax[1].legend(handle, labels, loc='upper left', ncol=2)
    
    ## ALARMS
    ax[2].plot(hour, true_alarm, color='skyblue', linewidth=4, label='True alarm')
    ax[2].plot(hour, false_alarm, color='r', linewidth=4, label='False alarm')
    ax[2].plot(hour, miss_alarm, color='grey', linewidth=4, label='Miss alarm')
    ax[2].set_ylim([0, 110])
    ax[2].set_yticks(np.arange(0, 100+1, 20))
    ax[2].set_ylabel('Alarm rate (%)')
    ax[2].legend(loc='upper left', ncol=3)
    
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.1, right=0.9, hspace=0.1)
    ax[0].set_title('Initial time: '+pd.to_datetime(time_start, format='%Y%m%d%H%M').strftime('%Y-%m-%d %H%MUTC'))
    ax[2].set_xlim([hour[0]-0.5, hour[-1]+0.5])
    ax[2].set_xticks(hour)
    ax[2].set_xlabel('Forecast hour')
    
    plt.savefig(f_output)
    plt.close()

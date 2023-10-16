# -*- coding: utf-8 -*-
"""
Author: whung

This is the main script to run the fire spread model.
"""

import numpy as np
import scipy.io as io
import pandas as pd
from netCDF4 import Dataset
import os, datetime, subprocess
import matplotlib.pyplot as plt

import rave_preprocessor, fire_inputgen, fire_forecast
from fire_corr import mlr_correction
from fire_mapgen import fire_mapper
from model_score import scoreanalysis



'''Settings'''
namelist = pd.read_csv('./input/namelist', header=None, delimiter='=')
namelist = namelist[1]

## input/ouput files
frp_input    = namelist[0].replace(' ', '')
model_input  = namelist[1].replace(' ', '')
model_output = namelist[2].replace(' ', '')

## frp source
frp_source   = int(namelist[3])    # 0: rave

## forecast period
time_start = str(namelist[4].replace(' ', ''))     # yyyymmddHHMM
time_end   = str(namelist[5].replace(' ', ''))     # yyyymmddHHMM
time_freq  = int(namelist[6].replace(' ', ''))     # unit: hr
time       = pd.date_range(start=time_start, end=time_end, freq=str(time_freq)+'H')

## domain
lat_lim = [float(namelist[7]), float(namelist[8])]
lon_lim = [float(namelist[9]), float(namelist[10])]

## function options
opt_frpgen     = int(namelist[11])    # input generator option (0: off, 1: on)
opt_inputgen   = int(namelist[12])    # input generator option (0: off, 1: on)
opt_forecast   = int(namelist[13])    # forecast model  option (0: off, 1: on)
opt_mapgen     = int(namelist[14])    # FRP map generator option (0: off, 1: on)
opt_corr       = int(namelist[15])    # model correction option (0: off, 1: on)

## output scaling option
scale_opt = int(namelist[16])        # output scale option (0: off, 1: on)
scale_val = float(namelist[17])      # output scale factor



TT = len(time)



print('--------------------')
print('---- Fire spread model initializing...')
print('---- Model cycle:', time_start, '-', time_end, ', freq=', time_freq, 'H, cycle=', TT)
print('---- Model domain:', lat_lim, lon_lim)


if not os.path.exists('./input/'+time_start):
    os.makedirs('./input/'+time_start)
if not os.path.exists('./output/'+time_start):
    os.makedirs('./output/'+time_start)

f_frp = './input/'+time_start+'/'+frp_input+'.'+time_start+'.nc'



'''Creating initial FRP file'''
if opt_frpgen == 1:
    if frp_source == 0:
        rave_preprocessor.preprocessor(frp_input, frp_source, time_start, lat_lim, lon_lim)



'''Running the Model'''
for i in np.arange(TT):
    print('--------------------')
    print('---- Cycle t+'+str(i+1), time[i].strftime('%Y%m%d%H%M'), ' running...')

    f_input  = './input/'+time_start+'/'+model_input+'.'+time_start+'.f'+('%02i'%i)+'.nc'
    f_output = './output/'+time_start+'/'+model_output+'.'+time_start+'.f'+('%02i'%(i+1))+'.nc'


    ## generate model input based on gridded frp
    if opt_inputgen == 1:
        code = fire_inputgen.main_driver(i, f_frp, f_input, lat_lim, lon_lim)
        if code == 0:
            print('---- Input generated!')
        elif code == 1:
            print('---- Model terminated due to no available input.')
            exit()
        elif code == 2:
            print('---- Model terminated due to no available fire frame.')
            exit()


    ## run model forecast + post-process
    if opt_forecast == 1:
        fire_forecast.spread_forecast(f_input, f_output)
        print('---- Forecast generated!')


    ## model correction
    if opt_corr == 1:
        mlr_correction(f_input, f_output)
        print('---- Model correction generated!')


    ## generate predicted fire maps
    if opt_mapgen == 1:
        fire_mapper(f_frp, f_output, opt_corr, scale_opt, scale_val)
        print('---- Fire map generated!')


    ## prepare for next cycle
    if i != TT-1:
        f_frp = f_output


    print('---- Cycle t+'+str(i+1), time[i].strftime('%Y%m%d%H%M'), ' complete!!!')
    del [f_input, f_output]



'''Create model complete file'''
with open('./output/'+time_start+'/complete_flag.txt', 'w') as file:
    file.write('MODEL COMPLETE!')

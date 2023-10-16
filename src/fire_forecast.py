# -*- coding: utf-8 -*-
"""
Author: whung

This script is used to generate fire spread forecast with well-trained model.
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import os, datetime
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.optimizers import Adam, Adamax     # Kingma and Ba 2015, https://arxiv.org/abs/1412.6980
from keras import backend as K

import warnings
warnings.simplefilter(action='ignore')



def normalization_coef():
    readin = pd.read_csv('./model/model_normalization_coef.txt')
    readin = np.array(readin)
    a1     = readin[0, 0]
    b1     = readin[1, 0]
    a2     = readin[0, -1]
    b2     = readin[1, -1]
    return a1, b1, a2, b2

def lr_tracer(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def frame_upscaling_test(DATA):
    resize = 8
    FF = DATA.shape[0]
    XX = DATA.shape[1]
    YY = DATA.shape[2]
    NN = DATA.shape[3]
    return tf.image.resize(DATA,  [XX*resize, YY*resize], method='nearest')

def loss_function(w0, normalized_zero):
    def loss(y_true, y_pred):
        y_error = K.square(y_true - y_pred)
        y_error = K.switch(K.equal(K.round(y_true*(10**7)), normalized_zero*(10**7)), w0*y_error , y_error)
        return K.mean(y_error)
    return loss

def scheduler(epoch, lr):
    if (epoch != 0) & (epoch % 100 == 0):
      return lr*0.7
    else:
      return lr



def spread_forecast(f_input, f_output):

    '''Settings'''
    ## constants
    learning_rate = 1e-4
    frp_thres     = 15      # fires with low FRP will be removed
    resize        = 8       # sacle factor
    nweight       = 0.6     # loss weight for zeros
    
    ## variable list
    firelist   = ['frp']
    geolist    = ['lat', 'lon', 'elv', 'ast', 'doy', 'time']
    veglist    = ['fh', 'vhi']
    metlist    = ['t2m', 'sh2', 'tp', 'wd', 'ws']    #, 't2m_f', 'sh2_f', 'tp_f', 'wd_f', 'ws_f']
    #flamelist  = ['flamews']
    INPUTLIST  = firelist + geolist + veglist + metlist # + flamelist
    OUTPUTLIST = ['frp_f']



    '''Reading Inputs'''
    readin    = Dataset(f_input)
    time      = readin['time'][0]
    time_fc   = (pd.to_datetime(readin['time'][0])+pd.Timedelta('1H')).strftime('%Y%m%d%H%M')
    X_lat     = readin['frame_lat'][:]
    X_lon     = readin['frame_lon'][:]
    X_initial = readin['input'][:]
    INPUTLIST = readin['INPUTLIST'][:]
    NN        = len(INPUTLIST)
    readin.close()
    
    print('---- Variable list:')
    print(INPUTLIST)
    print('---- Input data:')
    print(X_initial.shape, np.min(X_initial), np.max(X_initial))
    
    
    ## normalization coef
    a1, b1, a2, b2 = normalization_coef()    # 1: input/2: output
    
    normalized_zero = np.round((0-b2)/a2, 7)
    normalized_one  = np.round((1-b2)/a2, 7)
    
    
    
    start_time = datetime.datetime.now()
    
    
    
    '''Reading Model'''
    print('---- Model configuration:')
    opt   = Adamax(learning_rate=learning_rate, beta_1=0.90, beta_2=0.999, epsilon=1e-08, decay=1e-06)
    lr    = lr_tracer(opt)
    model = load_model('./model/fire_model.h5', custom_objects={'loss':loss_function(nweight, normalized_zero), 'lr':lr})
    model.summary()
    
    
    
    
    '''Forecast'''
    nlat = X_initial.shape[1]
    nlon = X_initial.shape[2]
    
    X = tf.convert_to_tensor(X_initial)
    X = frame_upscaling_test(X)
    
    Y = model.predict(X)
    Y[Y < normalized_one] = normalized_zero
    Y = np.array(Y)
    
    Y = tf.convert_to_tensor(Y)
    Y = tf.image.resize(Y, [nlat, nlon], method='nearest')
    Y = Y.numpy()
    
    
    ## de-normalization & re-scaling
    predic = Y*a2+b2
    predic[np.round(predic, 5)==0] = 0
    predic[predic!=0] = np.exp(predic[predic!=0])
    
    print('Original output:', Y.shape, np.min(Y), np.max(Y))
    print('Post-processed prediction:', predic.shape, np.min(predic), np.max(predic))
    
    
    ## save prediction
    f = Dataset(f_output, 'w')
    f.createDimension('flen', predic.shape[0])
    f.createDimension('xlen', nlat)
    f.createDimension('ylen', nlon)
    f.createDimension('num_input', NN)
    f.createDimension('num_output', 1)
    f.createDimension('time', 1)
    var_time = f.createVariable('time', str, ('time',))
    var_ori  = f.createVariable('frame_predic_ori', 'float', ('flen', 'xlen', 'ylen', 'num_output'))
    var_post = f.createVariable('frame_predic_post', 'float', ('flen', 'xlen', 'ylen', 'num_output'))
    var_lat  = f.createVariable('frame_lat', 'float', ('flen', 'xlen', 'ylen'))
    var_lon  = f.createVariable('frame_lon', 'float', ('flen', 'xlen', 'ylen'))
    var_list = f.createVariable('INPUTLIST', str, ('num_input',))
    var_time[:] = np.array(time_fc).astype(str)
    var_ori[:]  = Y
    var_post[:] = predic
    var_lat[:]  = X_lat
    var_lon[:]  = X_lon
    var_list[:] = np.array(INPUTLIST).astype(str)
    f.close()

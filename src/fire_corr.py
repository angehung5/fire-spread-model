# -*- coding: utf-8 -*-
"""
Author: whung

This script is used for model result correction, based on Multiple Linear Regression (MLR).
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import pickle

import warnings
warnings.simplefilter(action='ignore')



def mlr_correction(f_input, f_output):

    '''Reading Data'''
    ## normalization coef
    readin = pd.read_csv('./model/model_normalization_coef.txt')
    readin = np.array(readin)
    a      = readin[0, np.append(0, np.arange(3, 14))]
    b      = readin[1, np.append(0, np.arange(3, 14))]
    
    ## non-scaled input
    readin = Dataset(f_input)
    inputs = readin['input_noscale'][:]
    readin.close()
    del readin
    
    ## model first guess
    readin = Dataset(f_output)
    guess  = readin['frame_predic_post'][:, :, :, 0]
    readin.close()
    del readin
    
    inputs[:, :, :, 0] = guess
    
    
    '''Model correction'''
    model = pickle.load(open('./model/corr_model.sav', 'rb'))
    
    corr = np.copy(guess)
    mask = corr != 0
    corr[mask] = model.predict(inputs[mask, :].reshape(-1, inputs.shape[-1]))
    corr[corr < 0] = 0
    
    
    
    print('---- Model correction:')
    print('First guess:', guess.shape, np.min(guess), np.max(guess))
    print('Corrected prediction:', corr.shape, np.min(corr), np.max(corr))
    
    
    
    ## save prediction
    f = Dataset(f_output, 'a')
    var_corr = f.createVariable('frame_predic_corr', 'float', ('flen', 'xlen', 'ylen', 'num_output'))
    var_corr[:] = np.expand_dims(corr, axis=-1)
    f.close()

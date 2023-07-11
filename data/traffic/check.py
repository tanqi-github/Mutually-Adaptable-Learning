from get_traffic_data import get_traffic_data
import torch
import numpy as np
import sys
import argparse
sys.path.append("..")
from method.multiscale_atten import multiscale_atten
from ExpControl.Record_Result import record_result
# or one task one week.
import os
from progressbar import *
import time

def att_main():

    setting= {
        'npre-train':0.7,
        'npre-validate':0.5,
    }    

    Y_array, X_array =  get_traffic_data() # [N*nsample] [N*nsample*xlen*1]

    d = X_array.shape[3] 
    nlocation = Y_array.shape[0]
    nsample = Y_array.shape[1]

    npretrain = int(np.ceil(setting['npre-train']*nsample))
    nvalidation = int(np.ceil(setting['npre-validate']*nsample))


    y_train, y_val, y_test = Y_array[:,:nvalidation], Y_array[:,nvalidation:npretrain], Y_array[:,npretrain:] 
    x_train, x_val, x_test = X_array[:,:nvalidation], X_array[:,nvalidation:npretrain], X_array[:,npretrain:] 

    batch_size = 68
    idx_y = 0
    print(y_train[idx_y])
    for n in range(1):       

           

        xs_train = []
        xs_val =[]
        xs_test = []

        ## train.

        xs_train.append(np.concatenate(x_train, axis=2))
        xs_val.append(  np.concatenate(x_val,   axis=2))
        xs_test.append( np.concatenate(x_test,  axis=2))

    return xs_test,y_test

def lstm_main():
    setting= {
        'npre-train':0.7,
        'npre-validate':0.5,
    }    

    Y_array, X_array =  get_traffic_data() # [N*nsample] [N*nsample*xlen*1]

    X_array = np.concatenate(X_array,axis=2) #[nsample*xlen*N]

    d = X_array.shape[2] 
    nlocation = Y_array.shape[0]
    nsample = Y_array.shape[1]


    npretrain = int(np.ceil(setting['npre-train']*nsample))
    nvalidation = int(np.ceil(setting['npre-validate']*nsample))

    RMSE = np.zeros(nlocation)
    MAE = np.zeros(nlocation)
    RMSE_Baseline = np.zeros(nlocation)
    MAE_Baseline = np.zeros(nlocation)

    y_train, y_val, y_test = Y_array[:,:nvalidation], Y_array[:,nvalidation:npretrain], Y_array[:,npretrain:] 
    x_train, x_val, x_test = X_array[:nvalidation], X_array[nvalidation:npretrain], X_array[npretrain:] 

    return x_test,y_test

if __name__ == '__main__':
    xs_test1,y_test1  = att_main()
    xs_test2,y_test2  = lstm_main()

    print(np.all(y_test1==y_test2))
    print(np.all(xs_test1[0]==xs_test2))
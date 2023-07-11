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
import time

SEED = 777
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def main(h):

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

    RMSE = np.zeros(nlocation)
    MAE = np.zeros(nlocation)
    RMSE_Baseline = np.zeros(nlocation)
    MAE_Baseline = np.zeros(nlocation)

    y_train, y_val, y_test = Y_array[:,:nvalidation], Y_array[:,nvalidation:npretrain], Y_array[:,npretrain:] 
    x_train, x_val, x_test = X_array[:,:nvalidation], X_array[:,nvalidation:npretrain], X_array[:,npretrain:] 

    new_info_threshold = 0.1
    more_info_threshold = 0.2

    batch_size = 10

    for n in range(1):       
        
        model = multiscale_atten(N=1, D=batch_size*d, hidden_size=h, NLayer = 1)

        idx_y = 0

        xs_train = []
        xs_val =[]
        xs_test = []

        ## train.
        
        for j in range(0,x_train.shape[0],batch_size):
            jend = min(j + batch_size, x_train.shape[0])
            xs_train.append(np.concatenate(x_train[j:jend], axis=2)) #[N*nsample*xlen*1]
            xs_val.append(  np.concatenate(x_val[j:jend],   axis=2))
            xs_test.append( np.concatenate(x_test[j:jend],  axis=2))
        
        '''
        xs_train.append(np.concatenate(x_train, axis=2)) #[N*nsample*xlen*1]
        xs_val.append(  np.concatenate(x_val,   axis=2))
        xs_test.append( np.concatenate(x_test,  axis=2))
        '''

        start = model.expand_source(xs_tra=xs_train, y_tra=y_train[idx_y], xs_val=xs_val, y_val=y_val[idx_y], maxiter=10*100, nnew_source=len(xs_train) )

        [rmse,mae] = model.evaluate(xs=xs_test, y=y_test[idx_y])

        print('rmse: %.6f' % rmse)

        '''
        lstm_idx = model.select_lstm()
        
        lstm_idx = 1
        print('lstm %d'% lstm_idx)
        model.expand_atten(lstm_idx=lstm_idx, xs_tra=xs_train, y_tra=y_train[idx_y], xs_val=xs_val, y_val=y_val[idx_y], maxiter=100)

        [rmse,mae] = model.evaluate(xs=xs_test, y=y_test[idx_y])

        print('rmse: %.6f' % rmse)
        '''
        end = time.time()
        print('Time:%0.3f' % (end - start))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=int)
    args = parser.parse_args()

    for h in [5, 9, 18, 36, 74 ]: 
        main(h)





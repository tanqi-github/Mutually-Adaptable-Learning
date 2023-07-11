from get_traffic_data import get_traffic_data
import torch
import numpy as np
import sys
import argparse
sys.path.append("..")
from method.gated_RNN_Torch_RNN_LSTM_3_batch import gated_RNN_Torch
from ExpControl.Record_Result import record_result
# or one task one week.
import os
from progressbar import *

import time

SEED = 777
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def main():

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

    idx_y = 0

    for n in range(1):       
        
        model = gated_RNN_Torch(N=1, D=d, hidden_size=512, NLayer = 1)

        start = time.time()
        for epoch in range(100):
            model.train(x=x_train, y=y_train[idx_y], lr=0.001)
        end = time.time()
        print('Time:%0.3f' % (end - start))
        rmse = model.evaluate(x=x_test, y=y_test[idx_y])
        print(rmse)

        
        
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=int)
    args = parser.parse_args()

    for num_meta_updates in [20]: 
        for beta in [0.1]: 
            for alpha in [0.0001]: 
                for i in range(1):
                    main()





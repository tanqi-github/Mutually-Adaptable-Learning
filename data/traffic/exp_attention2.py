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


def main():

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
        
        model = multiscale_atten(N=1, D=batch_size*d, hidden_size=5, NLayer = 1)

        idx_y = 0

        xs_train = []
        xs_val =[]
        xs_test = []

        xs_train_data = []
        xs_val_data =[]
        xs_test_data = []

        for j in range(0,x_train.shape[0],batch_size):
            jend = min(j + batch_size, x_train.shape[0])
            xs_train_data.append(np.concatenate(x_train[j:jend], axis=2)) #[N*nsample*xlen*1]
            xs_val_data.append(  np.concatenate(x_val[j:jend],   axis=2))
            xs_test_data.append( np.concatenate(x_test[j:jend],  axis=2))
        
        ## train.
        
        for iround in range(3):
            for idata in range(len(xs_train_data)):
                is_expand = True
                if model.n_lstm == 0:
                    is_expand = True
                    new_info_rate = 1

                else:
                    
                    rmse_tra,rmse_val = model.check_source(xs_tra=xs_train, xn_tra=xs_train_data[idata][:,-1], xs_val=xs_val, xn_val=xs_val_data[idata][:,-1])
                    new_info_rate = rmse_tra / (np.max(x_val[j])-np.min(x_val[j]))
                    if new_info_rate > new_info_threshold:
                        is_expand = True

                if is_expand:
                    print("new info rate is %0.3f; expand %d th source" % (new_info_rate,idata))
                    xs_train.append(xs_train_data[idata])
                    xs_val.append(  xs_val_data[idata])
                    xs_test.append( xs_test_data[idata])
                    #xs_train[0].shape (2145, 30, 58)

                    model.expand_source(xs_tra=xs_train, y_tra=y_train[idx_y], xs_val=xs_val, y_val=y_val[idx_y], maxiter=10*100, nnew_source=1)

                else:
                    print("new info rate is %0.3f; ignore %d th source" %  (new_info_rate,j))
                    
            
                [rmse,mae] = model.evaluate(xs=xs_test, y=y_test[idx_y])
                print('rmse: %.6f' % rmse)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=int)
    args = parser.parse_args()

    for num_meta_updates in [20]: 
        for beta in [0.1]: 
            for alpha in [0.0001]: 
                for i in range(1):
                    main()





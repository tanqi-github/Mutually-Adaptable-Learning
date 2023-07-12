import numpy as np
import torch
import sys
import argparse
from construct_data import *
from construct_featureset import construct_taskdata_feature
import time
from os import listdir
from utils import maybe_create_dir
sys.path.append("..")
#from method.modules_hete_w_po import Seq2seq as Seq2seq_hete_w_po
#from method.modules_hete_f import Seq2seq as Seq2seq_hete_f
#from method.modules_s import Seq2seq as Seq2seq_s

from method.modules_hete_linear_output import Seq2seq as Seq2seq_hete_w_po
import copy
import random

use_cuda = False #torch.cuda.is_available()

random.seed(777)
np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed(777)
torch.backends.cudnn.deterministic = True

import time
import datetime

import argparse

def get_configs(dataset_name):
    parser = argparse.ArgumentParser(description='configs')

    if dataset_name == 'google_flu':
        parser.add_argument('--max_epochs', type=int, default=2000)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--pm_hidden_size', type=int, default=10)
        parser.add_argument('--xlen', type=int, default=60)
        parser.add_argument('--yscale', type=float, default=1)
        parser.add_argument('--init_version', type=str, default='041')
        parser.add_argument('--poiss', dest='poiss', action='store_true')
        parser.add_argument('--no-poiss', dest='poiss', action='store_false')
        parser.set_defaults(poiss=True)

        parser.add_argument('--ct', type=float, default=1e-4) # cutting threshold
        parser.add_argument('--interconn', dest='interconn', action='store_true')
        parser.add_argument('--max_nmodule', type=int, default=10)
        parser.set_defaults(interconn=False)

        parser.add_argument('--pm_lr', type=float, default=0.01)
        parser.add_argument('--wdecay', type=float, default=0.01)

        parser.add_argument('--optim_rdscale', type=float, default=0.2)
        parser.add_argument('--optim_patience', type=int, default=10) # don't drop in fever case
        parser.add_argument('--optim_validation_stop', type=float, default=0.8)
        parser.add_argument('--optim_measure_step', type=int, default=10)
        parser.add_argument('--sch', dest='use_scheduler', action='store_true')
        parser.add_argument('--no-sch', dest='use_scheduler', action='store_false')
        parser.set_defaults(use_scheduler=True)
        parser.add_argument('--sgd_output', type=int, default=1) # 0 for false, 1 for true
        parser.add_argument('--module_lr_decay', type=float,default=1.0)

        parser.add_argument('--pretrain_epoch', type=int, default=30)
        parser.add_argument('--posttrain_epoch', type=int, default=80) # don't drop in fever case

        parser.add_argument('--save_model', dest='save_model', action='store_true')
        parser.set_defaults(save_model=False)

    args = parser.parse_args()
    s = vars(args)
    s['method'] = 'Module_W'
    s['dataset_name'] = dataset_name
    s['yhorizon'] = 1
    if s['poiss']:
        s['likfun'] = 'poiss'
    else:
        s['likfun'] = 'gauss'

    return s

def main():

    dataset_name = 'google_flu'
    is_raw_dataset = True

    s = get_configs(dataset_name)
    s['is_raw_dataset'] = is_raw_dataset
    print(s) 
    
    start = time.time()
    if is_raw_dataset:
        data_list, task_data = construct_taskdata(s['dataset_name'],xlen=s['xlen'],yhorizon = s['yhorizon'])

    else:
        data_description, data_list, task_description, task_data, loc = construct_taskdata(s['dataset_name'])
        data_list = np.concatenate(data_list,2)

    done = time.time()
    elapsed = done - start
    print('finished data loading with %.6f seconds' % elapsed)


    for d in range(data_list.shape[2]): # ndatasize*nsample*xlen*1
        mind = np.min(data_list[:,:,d])
        maxd = np.max(data_list[:,:,d])
        data_list[:,:,d] = (data_list[:,:,d]- mind)/(maxd - mind +0.001)

    nbatch = data_list.shape[0]
    train_proportion = int(nbatch//(1/0.7))
    traintrain_proportion = int(train_proportion//(1/0.7))

    random_idx = np.random.permutation(train_proportion)
    if use_cuda:
        traintrain_data_idx = torch.from_numpy(random_idx[:traintrain_proportion]).long().cuda()
    else:
        traintrain_data_idx = torch.from_numpy(random_idx[:traintrain_proportion]).long()
    trainval_data_idx = random_idx[traintrain_proportion:]

    test_idxs = [i for i in range(0, len(task_data), 5)]
    print("test_idxs", test_idxs)
    rmses_tasks = []
    maes_tasks = []
    best_configs = []
    if use_cuda:
        X = torch.from_numpy(data_list).float().cuda()
        Y = torch.from_numpy(task_data).float().cuda()
        XF = torch.zeros(data_list.shape[0], task_data[0].shape[1],1).cuda()
    else:
        X = torch.from_numpy(data_list).float()
        Y = torch.from_numpy(task_data).float()
        XF = torch.zeros(data_list.shape[0], task_data[0].shape[1], 1)

    print(Y.size())
    print(X.size())
    print(XF.size())

    lock = 0
    nmodule_dict = dict(zip(range(s['max_nmodule']+2), np.zeros(s['max_nmodule']+2)))

    time_stamp = str(datetime.datetime.now())
    model_dir= os.path.join('saved_model', dataset_name)
    maybe_create_dir(model_dir)

    for test_idx in test_idxs:
        rmses = []
        maes = []
        rmses_val = []
        maes_val = []
        configs = [] 
        

        pm = Seq2seq_hete_w_po( Dx=X.size()[2], Dxf=XF.size()[2], x_len=X.size()[1], output_horizon=s['yhorizon'], hidden_size=s['pm_hidden_size'], nlayer = 1,
            likfun=s['likfun'], interconnect = s['interconn'])
        mtpyes=[None]*s['max_nmodule'] # None, 'rnn','cnn','mlp' ATTEN: no hete

        if lock == 0:
            print(pm.show_description())
            lock = 1

        print('begin train')
        pm.gf = test_idx
        pre_rmse_val = 1000
        for j in range(len(mtpyes)):
            pm.expand(mtpyes[j])

            if s['sgd_output'] == 0:
                pm.update_output_layer( x = X[traintrain_data_idx], y=Y[test_idx,traintrain_data_idx], beta=1)

            pm.train(   xs_tra = X[traintrain_data_idx], y_tra=Y[test_idx,traintrain_data_idx], xf_tra = XF[traintrain_data_idx],
                        xs_val=X[trainval_data_idx], y_val=Y[test_idx,trainval_data_idx], xf_val = XF[trainval_data_idx], 
                        maxiter=s['max_epochs'], weight_decay = s['wdecay'],
                        use_scheduler=s['use_scheduler'],
                        pretrain_epoch=s['pretrain_epoch'],
                        posttrain_epoch=s['posttrain_epoch'],
                        lr=s['module_lr_decay']**j*s['pm_lr'],update_linear=s['sgd_output'],
                        measure_step=s['optim_measure_step'])
            if s['interconn']:
                pm.cutting_conns(s['ct'])

            if s['sgd_output'] == 0:
                pm.update_output_layer( x = X[traintrain_data_idx], y=Y[test_idx,traintrain_data_idx], beta=1)

            [rmse_val, mae_val] = pm.eval(xs= X[trainval_data_idx], 
                y=Y[test_idx,trainval_data_idx], xf = XF[trainval_data_idx])

            [rmse, mae] = pm.eval(xs= X[train_proportion:],
                y=Y[test_idx,train_proportion:], xf = XF[train_proportion:])


            print('test_idx: %d; it: %d; val rmse: %.3f; val mae: %.3f...rmse: %.3f; mae: %.3f.' % (test_idx, j, rmse_val, mae_val, rmse, mae))
            rmses_val.append(rmse_val)
            maes_val.append(mae_val)
            rmses.append(rmse)
            maes.append(mae)

            if (pre_rmse_val - rmse_val < 0.01) and pm.exam_incremental_b(x=X[:train_proportion], xf = XF[:train_proportion], y=Y[test_idx,:train_proportion], inner_boost=5,threshold = 0.01) == 1:
                break

            pre_rmse_val = rmse_val
 
            ## update the previous affected module
            if j > 0:
                hmis = pm.calculate_hmi(x=X[:train_proportion], xf = XF[:train_proportion], y=Y[test_idx,:train_proportion],inner_boost=20)
                print('hmis is '+str(hmis))
                for tempi in np.argwhere(hmis>0.01):
                    pm.train(   xs_tra = X[traintrain_data_idx], y_tra=Y[test_idx,traintrain_data_idx], xf_tra = XF[traintrain_data_idx],
                                xs_val=X[trainval_data_idx], y_val=Y[test_idx,trainval_data_idx], xf_val = XF[trainval_data_idx], 
                                maxiter=s['max_epochs'], weight_decay = s['wdecay'],
                                use_scheduler=s['use_scheduler'],
                                pretrain_epoch=s['pretrain_epoch'],
                                posttrain_epoch=s['posttrain_epoch'],update_linear=s['sgd_output'],
                                lr=s['module_lr_decay']**j*s['pm_lr'], measure_step=s['optim_measure_step'], 
                                target_module= tempi[0])
                    if s['interconn']:
                        pm.cutting_conns(s['ct'])

                    [rmse_val, mae_val] = pm.eval(xs= X[trainval_data_idx], 
                        y=Y[test_idx,trainval_data_idx], xf = XF[trainval_data_idx])

                    [rmse, mae] = pm.eval(xs= X[train_proportion:],
                        y=Y[test_idx,train_proportion:], xf = XF[train_proportion:])
            
                    print('test_idx: %d; after updating module: %d with mi %.3f; val rmse: %.3f; val mae: %.3f...rmse: %.3f; mae: %.3f.' % (test_idx, tempi[0], hmis[tempi[0]], rmse_val, mae_val, rmse, mae))

                    rmses_val.append(rmse_val)
                    maes_val.append(mae_val)
                    rmses.append(rmse)
                    maes.append(mae)

            print('test_idx: %d;  val rmse: %.3f; val mae: %.3f...rmse: %.3f; mae: %.3f.' % (test_idx,  rmse_val, mae_val, rmse, mae))
            ####end of module addition

        selected_model_idx = np.argmin(rmses_val)
        print('%d th model is selected' %selected_model_idx )
        print('rmse and mae is %.6f and %.6f \n' % (rmses[selected_model_idx], maes[selected_model_idx]))
        rmses_tasks.append(rmses[selected_model_idx])
        maes_tasks.append(maes[selected_model_idx])

        if s['save_model']:
            with open(os.path.join(model_dir,'model_task{}_{}.pt'.format(test_idx,time_stamp)),'wb') as fout:
                torch.save(pm, fout)

        # count the module
        nmodule_dict[pm.nmodule] += 1

    print(np.mean(rmses_tasks))
    print(np.mean(maes_tasks))
    print(test_idxs)
    print(pm.show_description())
    print(nmodule_dict)
    ## record 
    s['Experiment_Description'] = pm.show_description() + s['dataset_name'] + '_' + s['method'] + '_with test_idxs' + str(test_idxs)

    text = open("Experiment_Result_Detail_2.txt","a")
    text.writelines(time_stamp+'\n')
    text.writelines(str(s)+'\n')
    text.writelines(str(rmses_tasks)+'\n')
    text.writelines(str(maes_tasks)+'\n')
    #text.writelines(str(best_configs)+'\n')
    text.writelines('\n\n')
    text.close()

    text = open("Experiment_Result_Summary_2.txt","a")
    text.writelines(time_stamp+'\n')
    text.writelines(str(s['Experiment_Description'])+'\n')

    text.writelines('RMSE:\t')
    text.writelines(str(np.mean(rmses_tasks))+'\t')
    text.writelines('\n')

    text.writelines('MAE:\t')
    text.writelines(str(np.mean(maes_tasks))+'\t')
    text.writelines('\n')
    text.writelines('\n\n')
    text.close()


if __name__ == '__main__':
    main()
import numpy as np 
from get_traffic_data import get_traffic_data
import os.path
from math import sqrt

def construct_taskdata(data_str,xlen,yhorizon):

    data_list_fr = "%s_y%d_data_list.npy" % (data_str,yhorizon)
    task_data_fr = "%s_y%d_task_data.npy" % (data_str,yhorizon)

    if os.path.isfile(data_list_fr):
        print ("File %s exist" % data_list_fr)
        data_list = np.load(data_list_fr)
        task_data = np.load(task_data_fr)

    else:
        print ("File not exist")

        if data_str == 'SHTaxi':
            X,loc = get_traffic_data('../data/traffic/')
            print(loc.shape)
            print('get Shanghai Traffic Data')
        

        if data_str == 'google_flu':
            folder = '../data/google/'
            X = np.genfromtxt(folder+'flu_2006-2015.csv',delimiter=',')
            Y = np.genfromtxt(folder+'temp_flu.csv',delimiter=',')
            X = np.stack([X.T,Y.T],2)
            loc = np.genfromtxt(folder+'flu_loc.csv',delimiter=',')
            print('get google flu data')
            #X = np.expand_dims(X.T,2)
            #Ttarget = 0
            #xlen = 15

        if data_str == 'usa_climate':
            X =  np.load('../data/climate/narr_data.npy')
            loc = np.load('../data/climate/narr_loc.npy')

            X = np.swapaxes(X,0,1)
            print('get cliate data with shape:' + str(X.shape))

        [N,T,dtype] = X.shape

        print(N)
        print(T)
        print(dtype)

        x = X.swapaxes(0,1).reshape([T,-1]) # [N,T,dtype] -> [T,N,dtype]
        data_list = []
        for t in range(xlen,T-yhorizon):
            data_list.append(x[t-xlen:t])

        data_list = np.stack(data_list) # NSample, N*dtype


        task_data = []
        for n in range(N):
            task_data_temp = []
            for t in range(xlen,T-yhorizon):
                task_data_temp.append(X[n,t:t+yhorizon,0])

            task_data.append(np.stack(task_data_temp))

        task_data = np.stack(task_data)

        np.save(data_list_fr, data_list)
        np.save(task_data_fr, task_data)

    return  data_list, task_data

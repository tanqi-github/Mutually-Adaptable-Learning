import numpy as np 
import sklearn.preprocessing as sp
from numpy import genfromtxt
import datetime
from pandas import Series

def get_traffic_data(path):
    print('get_traffic_data-b')
    
    Traffic = genfromtxt(path+'Traffic_Index_F.csv',delimiter=',').T # [T*N]
    Weather = genfromtxt(path+'Weather_Expand.csv',delimiter=',').T
    AirQ = genfromtxt(path+'AIR_Expand.csv',delimiter=',').T

    loc = genfromtxt(path+'traffic_loc.csv',delimiter=',')
    
    ## preprocess traffic data
    for n in range(Traffic.shape[1]):
        series = Series(Traffic[:,n])
        rolling = series.rolling(window=6)
        rolling_mean = rolling.mean()
        Traffic[:,n] = rolling_mean
        
    N = Traffic.shape[1]
    T = Traffic.shape[0]

    Y = Traffic

    weather = sp.scale(Weather)
    Traffic[np.isnan(Traffic)] = 0
    traffic = sp.scale(Traffic)


    X = np.expand_dims(traffic,2) # [T*N*1]

    Y_list = []
    X_list = []

    xlen = 30

    Mask = np.ones_like(Y)

    for n in range(N):
        Y_list.append(np.expand_dims(Y[xlen:,n],1) ) # [T-xlen] for each element. let nsample = T-xlen

    for n in range(N):
        x_temp = [] # [nsample*xlen]
        for t in range(xlen,T): # for each element of Y, there is x.
            x_temp.append(X[t-xlen:t,n])

        X_list.append(np.array(x_temp))

    Y_array = np.array(Y_list) # [N*nsample]
    X_array = np.array(X_list) # [N*nsample*xlen*1] 1 is the dim of features.

    return Y_array, loc

if __name__ == '__main__':
    Y_array, X_array = get_traffic_data()
    print(Y_array.shape)
    print(X_array.shape)

import numpy as np 
from get_traffic_data import get_traffic_data
import os.path
from math import sqrt




def construct_taskdata_feature(data_str):

    if os.path.isfile(data_str+'_data_description.npy'):
        print (data_str+"_data_description_list.npy File exist")

        data_description_list = np.load(data_str+'_data_description.npy')
        data_list = np.load(data_str+'_data_list.npy')
        task_description = np.load(data_str+'_task_description.npy')
        task_data = np.load(data_str+'_task_data.npy')
        loc = np.load(data_str+'_loc.npy')
    else:
        print ("File not exist")

        if data_str is 'SHTaxi':
            X,loc = get_traffic_data('../data/traffic/')
            print(loc.shape)
            print('get Shanghai Traffic Data')
        

        if data_str is 'google_flu':
            folder = '../data/google/'
            X = np.genfromtxt(folder+'flu_2006-2015.csv',delimiter=',')
            Y = np.genfromtxt(folder+'temp_flu.csv',delimiter=',')
            X = np.stack([X.T,Y.T],2)
            loc = np.genfromtxt(folder+'flu_loc.csv',delimiter=',')
            print('get google flu data')
            #X = np.expand_dims(X.T,2)

            Ttarget = 0
            xlen = 15
        if data_str is 'usa_climate':
            X =  np.load('../data/climate/narr_data.npy')
            loc = np.load('../data/climate/narr_loc.npy')

            X = np.swapaxes(X,0,1)
            print('get cliate data with shape:' + str(X.shape))

        [N,T,dtype] = X.shape
        print(T)
        print(N)
        print(dtype)

        # hyper_param
        xlen = 10
        tbegin = 101

        temporal_resolutions = [1,5]
        spatio_resolutions = [1,10] # KNN
        #time_lags = [1,10,20]

        data_description_list = []
        data_list = []

        # data description format: dtype, temporal_resolutions, spatio_resolutions, [2*spatial]
        for d in range(dtype):
            for tr in temporal_resolutions:
                for sr in spatio_resolutions:
                    #for tl in time_lags:
                    for i in range(N):

                        loc_idx = get_neighbors(loc,loc[i],sr)
                        loc_idx = np.array(loc_idx).astype(int)
   
                        _data_list = []

                        for t in range(tbegin,T):
                            _temp = []
                            for _xlen in range(xlen):
                                _temp.append( np.mean(X[loc_idx,t-(_xlen+1)*tr:t-_xlen*tr,d]) )

                            _temp = _temp[::-1]
                            _data_list.append(np.expand_dims(np.array(_temp),1)) #np.array(_temp): [xlen*]

                        data_list.append(np.stack(_data_list,0)) # np.stack(_data_list,0) [nsample*xlen]
                        data_description_list.append(np.expand_dims(np.array([d,tr,sr,loc[i][0],loc[i][1]]),0) )


        task_description = []
        task_data = []
        for d in [0]:
            for tr in [1]:
                for sr in [1]:
                    for i in range(N):
                        loc_idx = i
        
                        task_data.append(np.expand_dims(X[loc_idx,tbegin:T,d],1))
                        task_description.append(np.expand_dims(np.array([d,tr,sr,loc[i][0],loc[i][1]]),0) )

        data_description = np.concatenate(data_description_list,0)

        np.save(data_str+'_data_description.npy', data_description)
        np.save(data_str+'_data_list.npy', data_list)
        np.save(data_str+'_task_description.npy', task_description)
        np.save(data_str+'_task_data.npy', task_data)
        np.save(data_str+'_loc.npy', loc)

    return data_description_list, data_list, task_description, task_data, loc


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def get_neighbors(loc, loc_query, num_neighbors):
    distances = list()
    for i in range(loc.shape[0]):
        dist = euclidean_distance(loc_query, loc[i])
        distances.append((i, dist))

    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors

def get_nearby(task_description, data_description, k):
    dist = []
    for i in range(len(data_description)):
        dist.append(np.sum((task_description[0,] - data_description[i,])**2))

    arg = np.argsort(np.array(dist))[:k]
    return arg


def create_spatial(latlon,knn):
    from math import sin, cos, sqrt, atan2, radians
    # https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    # approximate radius of earth in km
    def distance(x1,x2):
        lat1, lon1 = x1
        lat2, lon2 = x2

        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return c

    ng = latlon.shape[0]

    mis = np.zeros((ng,ng))

    for i in range(ng):
        for j in range(i):
            mi = 1./distance(latlon[i],latlon[j])
            mis[i,j] = mi
            mis[j,i] = mi

    g = {}
    for i in range(ng):
        idx = np.argsort(mis[i])
        neighborhood = []
        for k in range(knn):
            v = idx[-k-1]
            neighborhood.append([mis[i,v],v])
        g[i] = neighborhood
    return g
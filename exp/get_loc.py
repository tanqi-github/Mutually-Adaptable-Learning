import numpy as np 
import os.path

def construct_taskdata(data_str):


    if data_str == 'SHTaxi':
        loc = np.genfromtxt('../data/traffic/traffic_loc.csv',delimiter=',')
    

    if data_str == 'google_flu':
        folder = '../data/google/'
        loc = np.genfromtxt(folder+'flu_loc.csv',delimiter=',')


    if data_str == 'usa_climate':
        loc = np.load('../data/climate/narr_loc.npy')



    return  loc

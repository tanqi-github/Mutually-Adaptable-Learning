import numpy as np
import os
import time
import torch
use_cuda = False #torch.cuda.is_available()
def V2Cuda(x):
    if use_cuda:
        return x.cuda()
    else:
        return x

def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = len(data)
    p = np.random.permutation(num)
    return data[p]

def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(data):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield data[start:end]

class early_stop():
    def __init__(self, measure_step=5, validation_stop=0.5):
        self.validation_score = [] 
        self.measure_step = measure_step
        self.validation_stop = validation_stop

    def record(self,score):
        self.validation_score.append(score)


    def check_break(self):
        if len(self.validation_score) > self.measure_step+2:
            return np.sum(np.array(self.validation_score[-self.measure_step:]) > self.validation_score[-self.measure_step-1]) > self.validation_stop*self.measure_step
        else:
            return False

    def adapt_lr(self, lr):
        if len(self.validation_score) > self.measure_step + 2:
            if self.validation_score[-1] < self.validation_score[-2]:
                lr = lr*1.3
            else:
                lr = lr*0.5

        return lr


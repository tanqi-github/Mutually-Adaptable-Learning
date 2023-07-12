import numpy as np
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
#from torch.distributions import Normal,Uniform
from sklearn.neighbors import NearestNeighbors

use_cuda = False
def V2Cuda(x):
    if use_cuda:
        return x.cuda()
    else:
        return x

def batch_generator(data, batch_size, shuffle=True):

    def shuffle_aligned_list(data):
        """Shuffle arrays in a list by shuffling each array identically."""
        num = len(data)
        p = np.random.permutation(num)
        return data[p]

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
'''
class Classifer(nn.Module):

    def __init__(self, input_dim, hidden_size = 64, embed_size = 32):
        super(Classifer, self).__init__()

        x_dim, y_dim = input_dim

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1),
            nn.Sigmoid()
        )

        self.embed_x =   nn.Linear(x_dim, embed_size)
        self.embed_y = nn.Linear(y_dim, embed_size)

    def forward(self,X,Y):
        embed_x = self.embed_x(X)
        embed_y = self.embed_y(Y)

        input_t = torch.cat([embed_x, embed_y],1)

        return self.classifier(input_t)

'''
class Classifer(nn.Module):

    def __init__(self, input_dim, hidden_size = 64, embed_size = 32):
        super(Classifer, self).__init__()

        x_dim, y_dim = input_dim

        self.classifier = nn.Sequential(
            nn.Linear(x_dim+y_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1),
            nn.Sigmoid()
        )

    def forward(self,X,Y):

        input_t = torch.cat([X, Y],1)

        return self.classifier(input_t)

class RF_CMI_MIDIF(object):

    def __init__(self,  hidden_size = 32, lr=0.001,inner_boost =2,maxiter=500):

        self.hidden_size = hidden_size
        self.lr = lr
        self.eps = 1e-6
        self.inner_boost = inner_boost
        self.maxiter = maxiter


    def batch_cmi(self,x,y,z):
        mi_hat = []
        for boost in range(self.inner_boost):
            mi_hat.append(self.get_cmi(x=x,y=y,z=z))
        return np.mean(mi_hat)


    def get_cmi(self, x, y, z):

        mi_x_yz = self.get_mi(x=x, y=torch.cat([y,z],1))
        mi_x_z = self.get_mi(x=x, y=z)

        return mi_x_yz - mi_x_z

    def get_mi(self, x, y, x_shuffle_dim=None):
        self.classifier = V2Cuda(Classifer(input_dim=( x.size()[1], y.size()[1] ), hidden_size= self.hidden_size*2, embed_size=self.hidden_size ))

        self.optimizer_order = optim.Adam(self.classifier.parameters(),lr=self.lr,weight_decay = 1e-4)

        train_partition = int(2*x.size()[0]//3)

        #data_joint = torch.cat([x,y],1)
        indx = np.random.permutation(x.size()[0])
        #data_marginal = torch.cat([x,y[indx]],1)

        xmarginal = x
        ymarginal = y[indx]

        self.data_joint_train_seq =     x[:train_partition]
        self.data_marginal_train_seq =  xmarginal[:train_partition]
        
        self.data_joint_train_vec =     y[:train_partition]
        self.data_marginal_train_vec =  ymarginal[:train_partition]

        self.data_joint_eval_seq =      x[train_partition:]
        self.data_marginal_eval_seq =   xmarginal[train_partition:]       
        
        self.data_joint_eval_vec =      y[train_partition:]
        self.data_marginal_eval_vec =   ymarginal[train_partition:]

        self._train_classifier(maxiter=self.maxiter)

        mi_hat = self._cal_cmi()
    
        return mi_hat.cpu().data.numpy() 


    def _cal_cmi(self):
        y_prob1 = self.forward(self.data_joint_eval_seq,self.data_joint_eval_vec)
        xpmeanlog = torch.mean(torch.log( (y_prob1+self.eps)/(1-y_prob1+self.eps)))

        y_prob2 = self.forward(self.data_marginal_eval_seq,self.data_marginal_eval_vec)
        xplogmean = torch.log(torch.mean( (y_prob2+self.eps)/(1-y_prob2+self.eps)))

        mi_hat = xpmeanlog-xplogmean
        return mi_hat

    def forward(self,x,y):
        prob = self.classifier(x,y)

        return prob

    def _eval(self):
        #input_bit = torch.cat([self.data_joint_eval, self.data_marginal_eval],0)
        input_b_seq = torch.cat([self.data_joint_eval_seq, self.data_marginal_eval_seq],0)
        input_b_vec = torch.cat([self.data_joint_eval_vec, self.data_marginal_eval_vec],0)

        label = V2Cuda(torch.cat([torch.ones(self.data_joint_eval_seq.size()[0]), torch.zeros(self.data_joint_eval_seq.size()[0] )],0))


        y_prob = self.forward(input_b_seq,input_b_vec)
        loss = nn.BCELoss()(y_prob, label.unsqueeze(1))
        return loss


    def _train_classifier(self,maxiter):
        ## Train classifer 
        batch_size = 64
    
        train_idx = np.arange(int(self.data_joint_train_seq.size()[0]))
        criterion = nn.BCELoss()
        val_loss = []
        measure_step = 10
        mi_hats = []


        for it in range(maxiter):
            
            batch_gen_train = batch_generator(train_idx,batch_size)
            
            for bit in range(len(train_idx)//batch_size):
                batch_idx = next(batch_gen_train)

                
                label = V2Cuda(torch.cat([torch.ones(len(batch_idx)), torch.zeros(len(batch_idx))],0))

                input_b_seq = torch.cat([self.data_joint_train_seq[batch_idx], self.data_marginal_train_seq[batch_idx]],0)
                input_b_vec = torch.cat([self.data_joint_train_vec[batch_idx], self.data_marginal_train_vec[batch_idx]],0)

                y_prob = self.forward(input_b_seq,input_b_vec)

                loss = criterion(y_prob, label.unsqueeze(1))

         
                self.optimizer_order.zero_grad()
                loss.backward()
                self.optimizer_order.step()

            val_loss.append(self._eval())
            if len(val_loss) > 5 + measure_step and torch.sum(torch.stack([val_loss[-1] > ele for ele in val_loss[-measure_step:-1]])) > 0.65*measure_step:
                break

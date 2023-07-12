import numpy as np
import sys
import argparse
import time
import datetime
import math
from copy import deepcopy

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn import Parameter
from collections import OrderedDict

use_cuda = False #torch.cuda.is_available()
np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed(777)
from method.method_utils import *
from method.cnn_enc import WaveNet_Encoder
from method.RF_CMI_CLASS import RF_CMI_CLASS
from method.RF_CMI_MIDIF import RF_CMI_MIDIF

class MLP_Encoder(nn.Module):
    def __init__(self, D, hidden_size,x_len,use_fs=False):
        ## batch_first
        super(MLP_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(x_len * D, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            )

        self.use_fs = use_fs

    def forward(self, x):
        if self.use_fs:
            raiseNotImplementedError('MLP_Encoder feature selection')

        else:
            return self.mlp(x.view(x.size(0),-1))

class RNN_Encoder(nn.Module):
    def __init__(self, D, hidden_size,x_len, nlayer = 1):
        ## batch_first
        super(RNN_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(D,hidden_size,batch_first=True) 
    
    def forward(self, x):
        _, (h_n, _) = self.rnn(x) #  output( seq_len, batch, num_directions * hidden_size) h_n (num_layers * num_directions, batch, hidden_size):
        return h_n[-1] # (batch, hidden_size)

class Encoder(nn.Module):
    def __init__(self, D, hidden_size, x_len, embeding_size=16, nlayer = 1, low_feat_emb = 0):
        super(Encoder, self).__init__()
        self.emb = nn.Linear(D+low_feat_emb,embeding_size)
        self.emodules = nn.ModuleList()
        self.mtypes = ['mlp','cnn','rnn']
        self.emodules.append(MLP_Encoder(D=embeding_size, hidden_size = hidden_size,x_len=x_len))
        self.emodules.append(WaveNet_Encoder(D=embeding_size, res_channels=16, skip_channels=hidden_size,x_len=x_len))
        self.emodules.append(RNN_Encoder(D=embeding_size, hidden_size = hidden_size, x_len=x_len))
        self.wining_count = torch.ones(3).cuda() if use_cuda else torch.ones(3)
        self.lock = False # this module is not influenced by the mtype

    def forward(self, x, mtype=None):
        x = self.emb(x)
        m = nn.Tanh()
        if mtype is None or self.lock:
            bidx = torch.argmax(self.wining_count)
            mtype = self.mtypes[bidx]

        if mtype == 'mlp':
            z = m(self.emodules[0](x))

        elif mtype == 'cnn':
            z = m(self.emodules[1](x))

        elif mtype == 'rnn':
            z = m(self.emodules[2](x))

        return z

class Linear_Decoder(nn.Module):
    def __init__(self,
        input_size, 
        output_horizon,
        encoder_hidden_size, 
        decoder_hidden_size, 
        output_size):
        super(Linear_Decoder, self).__init__()

        self.output_size = output_size

        self.mu_mlp = nn.Parameter(torch.randn((encoder_hidden_size,output_size)), requires_grad=True)
        stdv = 1. / math.sqrt(self.mu_mlp.size(1))
        self.mu_mlp.data.uniform_(-stdv, stdv)

        self.register_parameter('linear_w',self.mu_mlp)

    def forward(self, ht):
        mu = torch.matmul(ht, self.mu_mlp) # ht: BxH
        return mu, None

    def global_update(self,ht,y,beta=1e-4):
        '''
        with torch.no_grad():
            idenmat = beta * torch.eye(ht.size(1)).cuda()
            U = torch.matmul(ht.transpose(0,1), ht) + idenmat
            Uinv = torch.inverse(U)
            print(Uinv.size())
            print(ht.size())
            torch.matmul(ht.transpose(0,1),y)
            self.mu_mlp.data = torch.matmul(Uinv,torch.matmul(ht.transpose(0,1),y))
        '''
        #ht = augment(ht,ht.size(1))
        
        htn = ht.cpu().data.numpy()
        yn = y.cpu().data.numpy()
        
        idenmat = beta * np.eye(ht.size(1))
        U = np.dot(htn.T, htn) + idenmat
        Uinv = np.linalg.inv(U)
        W = np.dot(Uinv,np.dot(htn.T,yn))
        self.mu_mlp.data = torch.from_numpy(W).cuda().float()



    def global_update_poisson(self,ht,y,beta=1e-4):
        batch_size = ht.size(0)
        train_idx = np.arange(ht.size()[0])#range(ht.size(0))
        optimizer = optim.SGD(self.parameters(), lr=1e-3, weight_decay=beta)
        
        ht = ht.detach()
        y = y.detach()


        pre_nll = 1e10

        while True:
            nllsum = 0.01
            batch_gen_train = batch_generator(train_idx,batch_size)
            
            for it in range(len(train_idx)//batch_size):
                batch_idx = next(batch_gen_train)

                est = self.forward(ht[batch_idx])[0]

                NLL = torch.nn.functional.poisson_nll_loss(input=est, target=y[batch_idx], log_input=True, full=False, reduction='mean')
                loss = Variable(NLL, requires_grad = True)
                nllsum += NLL.data.cpu().numpy()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #assert nllsum > 0

            if nllsum > pre_nll - 1e-8*np.abs(pre_nll):
                break

            #print(nllsum)
            pre_nll = nllsum


    def expand(self, new_size):
        new_output = nn.Parameter(torch.randn((self.mu_mlp.data.size(0) + new_size, self.output_size)))
        stdv = 1. / math.sqrt(new_output.size(1))
        new_output.data.uniform_(-stdv, stdv)
        #print(new_output.size())
        #print(new_output.data[:self.mu_mlp.data.size(0),:].size())
        #print(self.mu_mlp.data.size())
        new_output.data[:self.mu_mlp.data.size(0),:] = self.mu_mlp.data
        self.mu_mlp = new_output

    def __delitem__(self, delete_size):
        ## only del the output of last module
        new_output = nn.Parameter(torch.randn((self.mu_mlp.data.size(0) - delete_size, self.output_size)))
        new_output.data = self.mu_mlp.data[:-delete_size]
        self.mu_mlp = new_output



class Seq2seq(nn.Module):
    
    def __init__(self, Dx, Dxf, output_horizon, x_len, hidden_size, output_size=1, nlayer = 1, likfun = 'gauss', interconnect = False, l1_lambda=0.01):
        super(Seq2seq, self).__init__()
        self.hidden_size = hidden_size
        self.Dx = Dx
        self.Dxf = Dxf
        self.x_len = x_len
        self.output_horizon = output_horizon    
        self.emb_size = 16
        self.embs = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.decoder = Linear_Decoder(
                            input_size=Dx,
                            output_horizon=output_horizon,
                            encoder_hidden_size=0, 
                            decoder_hidden_size=None, 
                            output_size=output_size)

        self.conns = torch.nn.ModuleDict()

        self.indicators = nn.ParameterList() #

        self.nmodule = 0
        
        self.s = {
            'name':'super_linear_output',
            'fs':'l1', # glubel,l1
            'mtype':[], # hete, rnn,mlp
            'l1_lambda':l1_lambda,
            'mi_lambda':10,
            'output_prob':likfun,
            'interconnect':interconnect,
            'hidden_size':hidden_size,
            'low_feat_emb':8,
            'output':'linear_output',
        }
        self.likfun = likfun
        self.mtypes = ['mlp','cnn','rnn']
        self.l2_loss_function = nn.MSELoss()
        self.lock_mtype = None
        self.gf = None
        if self.likfun is 'crossentropy':
            self.ce_criterion = nn.CrossEntropyLoss().cuda()

    def show_description(self):
        return str(self.s)

    def expand(self,mtype=None):
        self.nmodule = self.nmodule + 1 # 1,2,3,4,5,6....
        
        if self.nmodule > 1 and self.s['interconnect']:
            low_feat_emb = self.s['low_feat_emb']
            
            for i in range(self.nmodule-1):
                self.conns.add_module('conn_%d_%d' % (i, self.nmodule-1), nn.Linear(self.hidden_size, self.s['low_feat_emb']))
                
        else:
            low_feat_emb = 0 
            
        self.indicators.append(Parameter(torch.ones(self.Dx), requires_grad=True))
        self.embs.append(nn.Linear(self.Dx, self.emb_size))
        self.encoders.append(Encoder(D= self.emb_size+1, hidden_size = self.hidden_size, x_len=self.x_len, nlayer = 1,low_feat_emb=low_feat_emb))
        self.decoder.expand(self.hidden_size)
        
        if mtype is not None:
            self.lock_mtype = self.mtypes.index(mtype)
            self.encoders[-1].wining_count[self.lock_mtype] = 10
            
        if use_cuda:
            self.cuda()

    def nll(self, true, est, logvar):
        if self.likfun is 'gauss':
            #NLL =  torch.mean(logvar/2 + (true - est)**2 / (2*torch.exp(logvar))) # Negative Log Lik
            NLL =  torch.mean(0.5 * ((true - est) / logvar) ** torch.log(logvar))

        elif self.likfun is 'poiss':
            # log_input is true !
            NLL = torch.nn.functional.poisson_nll_loss(input=est, target=true, log_input=True, full=True,reduction='mean')
            # https://pytorch.org/docs/stable/nn.functional.html

        elif self.likfun is 'crossentropy':
            NLL = self.ce_criterion(est, true)

        else:
            raiseNotImplementedError('nll error')
            
        return NLL

    def get_best_module(self, xs_tra,xf_tra,y_tra):
        if self.lock_mtype is None:
            best_loss = 1e8
            best_idx = -1
            
            for i in range(3):
                mu,logvar = self.forward(xs_tra,xf_tra, self.mtypes[i])

                if self.likfun is 'poiss':
                    mu = torch.exp(mu)

                loss = self.l2_loss_function(mu,y_tra)

                if loss < best_loss:
                    best_idx = i
                    best_loss = loss

            assert best_idx > -1

        else:
            best_idx = self.lock_mtype 

        return best_idx


    def train(self, xs_tra, y_tra,xf_tra,  xs_val,  y_val, xf_val, maxiter=100, validation_stop=0.6,lr=1e-3, weight_decay=0, use_scheduler=False,
        pretrain_epoch=10,posttrain_epoch=80,measure_step=10,target_module=-1,update_linear=0):
        # update_linear: 0 no sgd update; 1 sgd update; <0 analytic update each -update_linear setep
        train_idx = np.arange(xs_tra.size()[0])
        val_idx = np.arange(xs_val.size()[0])

        l2_loss_function = nn.MSELoss()
        es = early_stop(measure_step=measure_step, validation_stop=validation_stop)

        batch_size = 64

        #D_parameters = self.parameters()
        D_parameters = [
            {'params': self.indicators[target_module]},
            {'params': self.embs[target_module].parameters()},
            {'params': self.encoders[target_module].parameters()},
        ]
        if update_linear==1:
            D_parameters.append( {'params': self.decoder.parameters()} )

        #print(update_linear) 
        #print([param for param in D_parameters[-1]['params']])# the weight is extrem large...
        #exit(0)
        if self.s['interconnect']:
            if target_module == -1:
                target_module_idx = self.nmodule-1
            
            else:
                target_module_idx = target_module

            for j in range(target_module_idx):
                D_parameters.append( {'params': self.conns['conn_%d_%d' % (j,target_module_idx)].parameters()} )


        optimizer = optim.Adam(D_parameters, lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.4*maxiter),int(0.8*maxiter)], gamma=0.4)

        for epoch in range(maxiter): 


            batch_gen_train = batch_generator(train_idx,batch_size)
            
            for it in range(len(train_idx)//batch_size):
                batch_idx = next(batch_gen_train)

                if self.lock_mtype is None:
                    if epoch < pretrain_epoch:
                        best_module_idx = it % 3

                    elif epoch > posttrain_epoch:
                        self.encoders[-1].lock = True
                        best_module_idx = 0
                    else: 
                        best_module_idx = self.get_best_module(xs_tra = xs_tra[batch_idx] ,xf_tra = xs_tra[batch_idx], y_tra = y_tra[batch_idx])
                        self.encoders[-1].wining_count[best_module_idx] += 1 

                else:
                    best_module_idx = self.lock_mtype

                mu,logvar = self.forward(xs_tra[batch_idx],xf_tra[batch_idx],self.mtypes[best_module_idx])

                #loss = l2_loss_function(mu, y_tra[batch_idx])

                if self.likfun == 'gauss': #epoch < 20 and
                    loss = l2_loss_function(mu, y_tra[batch_idx])

                else:
                    loss = self.nll(true=y_tra[batch_idx], est=mu, logvar=logvar)

                if self.s['fs'] == 'l1':
                    for i in range(len(self.indicators)):
                        loss = loss + self.s['l1_lambda']*torch.norm(self.indicators[i],1)
                        loss = loss + self.s['l1_lambda']*torch.norm(self.embs[i].weight,1)

                    ## a small typo. but anyway...
                    if self.s['interconnect']:
                        for conn in self.conns.values():
                            loss = loss + self.s['l1_lambda']*torch.norm(conn.weight,1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if update_linear < 0 and epoch % -update_linear ==0:
                self.update_output_layer(xs_tra, y_tra,beta=1)

            if use_scheduler:
                scheduler.step()

            
            RMSE,MAE = self.eval(xs_val, xf_val, y_val)
            if epoch % 10 == 0:
                print('epoch %d: loss is %.3f Val RMSE %.3f' % (epoch, loss, RMSE))

            if epoch > pretrain_epoch:
                es.record(RMSE)

            if es.check_break():
                break

        ## lock 
        self.encoders[-1].lock = True
        print('The wining module is %s.' % self.mtypes[torch.argmax(self.encoders[-1].wining_count)])
        self.s['mtype'].append(self.mtypes[torch.argmax(self.encoders[-1].wining_count)])

    def update_output_layer(self, x, y,beta=1e-4):

        with torch.no_grad():
            zs = self.forward(x, None, return_hidden=True)
            ht = torch.cat(zs,1)
            if self.likfun == 'gauss':
                self.decoder.global_update(ht,y,beta)

            else:
                self.decoder.global_update_poisson(ht,y,beta)

    def cutting_conns(self, theshold):
        for conn in self.conns.values():
            if torch.norm(conn.weight,2) < theshold:
                conn.weight = nn.Parameter(torch.zeros_like(conn.weight),requires_grad = False).cuda()
                conn.bias = nn.Parameter(torch.zeros_like(conn.bias),requires_grad = False).cuda()
                
                conn.requires_grad = False
                for param in conn.parameters():
                    param.requires_grad = False


    def eval(self,xs,xf,y):
        with torch.no_grad():
            l2_loss_function = nn.MSELoss()
            l1_loss_function = nn.L1Loss()

            MSE = 0
            MAE = 0

            batch_size = 1000
            begin_b = 0
            count_batch = 0
            while begin_b < xs.size()[0]:
                end_b = min(xs.size()[0], begin_b+batch_size)
                output,_ = self.forward(xs[begin_b:end_b],xf[begin_b:end_b])

                if self.likfun is 'poiss':
                    output = torch.exp(output)

                MSE += l2_loss_function(output, y[begin_b:end_b]).cpu().data.numpy()
                MAE += l1_loss_function(output, y[begin_b:end_b]).cpu().data.numpy()

                begin_b += batch_size
                count_batch += 1

            MSE /= count_batch
            MAE /= count_batch
            #return [np.sqrt(MSE), MAE.cpu().data.numpy()] if use_cuda else [np.sqrt(MSE.data.numpy()), MAE.data.numpy()]
            return [np.sqrt(MSE), MAE]

    def calculate_mi(self,x,xf,y,mtype=None,inner_boost =2):
        mi_est1 = RF_CMI_CLASS(inner_boost=inner_boost)
        mi_est2 = RF_CMI_MIDIF(inner_boost=inner_boost)
        zs = self.forward(x,xf,return_hidden=True)

        if self.nmodule == 1:
            mi = mi_est1.get_mi(x=y.detach(),y=z.detach())
            return mi
            
        else:
            cmi1 = mi_est1.batch_cmi(x= y.detach() ,y=zs[-1].detach(),z=torch.cat(zs[:-1],1).detach())
            cmi2 = mi_est2.batch_cmi(x= y.detach() ,y=zs[-1].detach(),z=torch.cat(zs[:-1],1).detach())
            print([cmi1,cmi2])
            cmi = np.mean([cmi1,cmi2])

            return cmi

    def calculate_hmi(self,x,xf,y,mtype=None,inner_boost =2):
        mi_est1 = RF_CMI_CLASS(inner_boost=inner_boost)
        mi_est2 = RF_CMI_MIDIF(inner_boost=inner_boost)
        zs = self.forward(x,xf,return_hidden=True)

        if self.nmodule == 1:
            return [0]

        else:
            mis = []
            for j in range(self.nmodule-1):
                mi = mi_est1.get_mi(x=zs[-1].detach(),y=zs[j].detach())
                mis.append(mi)

            return np.array(mis)


    def exam_incremental_b(self, x, xf, y,  threshold = 0.02,inner_boost=2):
        if self.nmodule > 1:
            cmi = self.calculate_mi(x,xf,y,inner_boost)
            print(cmi)
            if cmi < threshold:
                self.nmodule = self.nmodule - 1
                #self.indicators.__delitem__(self.nmodule)
                self.embs.__delitem__(self.nmodule)
                self.encoders.__delitem__(self.nmodule)
                self.decoder.__delitem__(self.hidden_size)

                return 1

            else:
                return 0

        else:
            return 0

    def forward(self, x,xf, mtype=None,return_hidden=False):
        ys = []
        logvars = []
        outputs = []
        zs = []
        relu_act = nn.ReLU()
        for i in range(self.nmodule):
            
            if self.s['fs'] == 'glubel':
                raiseNotImplementedError('glubel is not used')

            elif self.s['fs'] == 'l1':
                e = self.indicators[i].unsqueeze(0).unsqueeze(0).repeat([x.size()[0],x.size()[1],1])

            else:
                e = torch.ones_like(x)

            xe = torch.cat([self.embs[i](e*x),x[:,:,self.gf].unsqueeze(2)],2)

            if self.s['interconnect'] and i > 0:
                #z = self.conns[i](torch.cat([z]+zs,1))
                low_feat_embs = []
                for j in range(i):
                    low_feat_embs.append(relu_act(self.conns['conn_%d_%d' % (j,i)](zs[-1])))
                
                low_feat_emb = torch.sum(torch.stack(low_feat_embs,0),0) #low_feat_emb: B x lfez
                xe = torch.cat([xe, low_feat_emb.unsqueeze(1).expand([-1, xe.size(1), -1])],2)

            z = self.encoders[i](xe, mtype)
               
            zs.append(z)
        
        if return_hidden:
            return zs

        else:
            mu, _ = self.decoder(torch.cat(zs,1))
            return mu, None

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)
    
    def forward(self, inputs):
        outputs = super(CausalConv1d, self).forward(inputs)
        return outputs[:,:,:-1]

class DilatedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(DilatedConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            padding, dilation, groups, bias)
    
    def forward(self, inputs):
        outputs = super(DilatedConv1d, self).forward(inputs)
        return outputs

class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        super(ResidualBlock, self).__init__()
        self.filter_conv = DilatedConv1d(in_channels=res_channels, out_channels=res_channels, dilation=dilation)
        self.gate_conv = DilatedConv1d(in_channels=res_channels, out_channels=res_channels, dilation=dilation)
        self.skip_conv = nn.Conv1d(in_channels=res_channels, out_channels=skip_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(in_channels=res_channels, out_channels=res_channels, kernel_size=1)
        
    def forward(self,inputs):
        sigmoid_out = F.sigmoid(self.gate_conv(inputs))
        tahn_out = F.tanh(self.filter_conv(inputs))
        output = sigmoid_out * tahn_out
        #
        skip_out = self.skip_conv(output)
        res_out = self.residual_conv(output)
        res_out = res_out + inputs[:, :, -res_out.size(2):]
        # res
        return res_out , skip_out

class WaveNet_Encoder(nn.Module):
    def __init__(self, D, res_channels=32, skip_channels=16,x_len=60,dilations=None): # skip_channels = hidden_size
        super(WaveNet_Encoder, self).__init__()
        #self.dilations = [2**i for i in range(dilation_depth)] * n_repeat
        if x_len>20:
            self.dilations = [1,2,4,8,16,1,2,4,8]
        elif x_len<6:
            self.dilations = [1,2]
        else:
            self.dilations = [1,2,4]

        if dilations is not None:
            self.dilations = dilations

        self.main = nn.ModuleList([ResidualBlock(res_channels,skip_channels,dilation) for dilation in self.dilations])
        self.pre =  nn.Linear(D, res_channels)
        
        self.post = nn.Sequential(nn.ReLU(),
                                  nn.Linear(skip_channels,skip_channels),
                                  nn.ReLU(),
                                  nn.Linear(skip_channels,skip_channels))
        
    def forward(self,inputs):
        outputs = self.preprocess(inputs)
        skip_connections = []
        
        for layer in self.main:
            outputs,skip = layer(outputs)
            skip_connections.append(skip)

        outputs = sum([s[:,:,-outputs.size(2):] for s in skip_connections]).sum(2) # [B,skip_channels,outputs.size(2)] ->  [B,skip_channels]
        outputs = self.post(outputs)
        
        return outputs
    
    def preprocess(self,inputs):
        out = self.pre(inputs).transpose(1,2)
        return out

if __name__ == '__main__':
    model = WaveNet()
    x = torch.rand(3,1024,256)
    y = model(x)
    print(y.size())
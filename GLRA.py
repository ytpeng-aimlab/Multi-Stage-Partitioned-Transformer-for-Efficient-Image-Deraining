# Directional Pooling Attention

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
from torch.nn.modules.utils import _pair, _quadruple
from torchinfo import summary
from einops import rearrange

def inital_kernel(k_size, batch, channel, diagonal=False, types=False):
    if not diagonal:
        if types: #True for hor 
            kernel = (torch.ones(1, k_size)/k_size).cuda()
            m = nn.ZeroPad2d((0, 0, int(k_size//2), int(k_size//2))) # (left,right,top,bottom)
            kernel = m(kernel)
        else:
            kernel = (torch.ones(k_size, 1)/k_size).cuda()
            m = nn.ZeroPad2d((int(k_size//2), int(k_size//2), 0, 0)) # (left,right,top,bottom)
            kernel = m(kernel)
    else: # False for ani-diagonal
        kernel = torch.zeros(k_size, k_size).cuda()
        kernel = kernel.fill_diagonal_(1)/k_size
        if types:
            kernel = torch.flip(kernel, dims=[1])
    kernel = kernel.view(1, 1, k_size, k_size).repeat(batch, channel, 1, 1)
    return kernel


class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.act_fn = nn.GELU()
        self._init_weights()
        self.drop = nn.Dropout(0.)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Diagonal Strip pooling
class DSP(nn.Module):  
    def __init__(self, midplanes, k_size=9, diagonal=True, types=False):
        super(DSP, self).__init__()
        # self.norm = nn.BatchNorm2d(midplanes)
        self.kernel = inital_kernel(k_size=k_size, batch=midplanes, channel= midplanes, diagonal=diagonal, types=types)
        self.conv_1 = nn.Conv2d(midplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
    def forward(self, x):
        with torch.no_grad():
            diapool = F.conv2d(x, self.kernel, padding='same')
        # avg_result = self.norm(self.conv_1(diapool))
        avg_result = self.conv_1(diapool)

        return avg_result 

class HVSP(nn.Module):  
    def __init__(self, midplanes, k_size, types=False):
        super(HVSP, self).__init__()
        # self.norm = nn.BatchNorm2d(midplanes)
        if types == True: #True for hor; False for ver
            self.pool = nn.AvgPool2d((k_size, 1))
            self.conv_1 = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)
        else:
            self.pool = nn.AvgPool2d((1, k_size))
            self.conv_1 = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        _, _, h, w = x.size()
        # avg_result = self.norm(self.conv_1( self.pool(x)))
        avg_result = self.conv_1( self.pool(x))

        return F.interpolate(avg_result, (h, w))

class SP(nn.Module):  
    def __init__(self, midplanes, k_size, types=False):
        super(SP, self).__init__()
        # self.norm = nn.BatchNorm2d(midplanes)
        if types == True: #True for hor; False for ver
            self.pool = nn.AdaptiveAvgPool2d((None, 1))
            self.conv_1 = nn.Conv2d(midplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, None))
            self.conv_1 = nn.Conv2d(midplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
         
    def forward(self, x):
        _, _, h, w = x.size()
        # avg_result = self.norm(self.conv_1( self.pool(x)))
        avg_result = self.conv_1( self.pool(x))

        return avg_result.expand(-1, -1, h, w)# avg_result

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class DPA_v3(nn.Module):
    def __init__(self, inplanes):
        super(DPA_v3, self).__init__()

        midplanes = int(inplanes//4)
        # Input conv
        self.input_conv = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1)
        self.input_relu = nn.LeakyReLU(0.2, True)
        ks = 3

        self.SP_hor = SP(midplanes, k_size = ks, types=True)
        self.SP_ver = SP(midplanes, k_size = ks, types=False)
        self.SP_fusion = nn.Conv2d(midplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.hor = HVSP(midplanes, k_size = ks, types=True)
        self.ver = HVSP(midplanes, k_size = ks, types=False)
        self.dsp = DSP(midplanes, k_size = ks, diagonal=True, types=False)
        self.adsp = DSP(midplanes, k_size = ks, diagonal=True, types=True)
        
        self.conv = nn.Conv2d(5*midplanes, inplanes, kernel_size=3, stride = 1, padding = 1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        in_feature = self.input_relu((self.input_conv(x)))
        _, _, h, w = in_feature.size()

        x_1_h_sp = self.SP_hor(in_feature)
        x_1_w_sp = self.SP_ver(in_feature)
        sp = self.SP_fusion(x_1_h_sp*x_1_w_sp)

        x_1_h = self.hor(in_feature)
        x_1_w = self.ver(in_feature)
        x_1_d = self.dsp(in_feature)
        x_1_ad = self.adsp(in_feature)

        dpa = torch.cat([sp, x_1_h, x_1_w, x_1_d, x_1_ad], dim=1)   
        dpa = self.conv(dpa)
        dpa = self.sigmoid(dpa) * x
        return nn.LeakyReLU(0.2, True)(dpa+x)

if __name__ == '__main__':
    import numpy as np
    # inputs = torch.Tensor(16, 256, 96, 96)
    # x = torch.from_numpy(np.array([[[[1.,2.,3.],
    #                                     [4.,5.,6.],
    #                                     [7.,8.,9.]]]])).to(torch.float32)
    # x = torch.from_numpy(np.array([[[[1.,2.,3.,4.],
    #                                     [5.,6.,7.,8.],
    #                                     [9.,10.,11.,12.]]]])).to(torch.float32)
    pool = DPA(256)
    summary(pool, (16, 256, 96, 96))

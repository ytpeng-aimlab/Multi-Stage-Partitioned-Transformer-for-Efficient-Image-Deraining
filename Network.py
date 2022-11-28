
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as  F
from torch.autograd import Variable
import scipy.io as io

from BottleCSP import *

from utils import *

class MPT(nn.Module):
    def __init__(self, args):
        super(MPT, self).__init__()
        self.iter  = args.S                                        # Stage number S includes the initialization process
        self.num_M = args.num_M
        self.num_Z = args.num_Z

        # Stepsize
        self.etaM = torch.Tensor([1])                               # initialization
        self.etaB = torch.Tensor([5])                               # initialization
        self.etaM_S = self.make_eta(self.iter, self.etaM)
        self.etaB_S = self.make_eta(self.iter, self.etaB)
       
        self.to_R = nn.Conv2d(self.num_M, 3, kernel_size=9, stride=1, padding=4, groups=1)
        self.inital_FB = nn.Conv2d(3, self.num_M, kernel_size=3, stride=1, padding=1, dilation=1)
        self.to_FR_layer = nn.Conv2d(3, self.num_M, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.to_FB_layer = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dilation=1)

        # proxNet
        self.proxNet_B_0 = Bnet(args)                                 # used in initialization process
        self.proxNet_B_S = self.make_Bnet(self.iter, args)
        self.proxNet_M_S = self.make_Mnet(self.iter, args)
        self.proxNet_B_last_layer = Bnet(args)                       # fine-tune at the last

        # for sparse rain layer
        self.tau_const = torch.Tensor([1])
        self.tauR = nn.Parameter(self.tau_const, requires_grad=True)
        self.tauB = nn.Parameter(self.tau_const, requires_grad=True)

    def make_Bnet(self, iters, args):
        layers = []
        for i in range(iters):
            layers.append(Bnet(args))
        return nn.Sequential(*layers)
    def make_Mnet(self, iters, args):
        layers = []
        for i in range(iters):
            layers.append(Mnet(args))
        return nn.Sequential(*layers)
    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f, requires_grad=True)
        return eta

    def forward(self, input, labels=None):
        im_rain = input
        ListB = []
        ListR = []
        
        R_hat = input - torch.ones(input.shape).cuda()
        F_R_hat = self.to_FR_layer((1-self.etaM_S[0]/10) * R_hat + self.etaM_S[0]/10 * torch.ones(R_hat.shape).cuda()) # 3->32 conv(R+R0)
        FR_1 = self.proxNet_M_S[0](F_R_hat + torch.ones(F_R_hat.shape).cuda())
        R1 = self.to_R(FR_1) # 32->3

        B_hat = input - R1
        F_B_0 = self.inital_FB((1-self.etaB_S[0]/10) * B_hat + self.etaB_S[0]/10 * torch.ones(B_hat.shape).cuda()) # 3->3 conv(B+B0)
        input_concat = torch.cat((B_hat, F_B_0), dim=1) # 35
        B1, FB_1 = self.proxNet_B_S[0](input_concat)

        ListB.append(B1)
        ListR.append(R1)
        B = B1
        R = R1
        FB = FB_1
        FR = FR_1
        for i in range(self.iter-1):

            # GLRA for rain streaks
            R_hat = input - B 
            F_R_hat = self.to_FR_layer( (1-self.etaM_S[i+1, :]/10 * R_hat + self.etaM_S[i+1, :]/10 * R) )
            FR = self.proxNet_M_S[i+1](FR + F_R_hat)
            R = self.to_R(FR)
            ListR.append(R)
        
            # ACMLP for clean background
            B_hat = input - R
            B_mid = (1-self.etaB_S[i+1, :]/10) * B_hat + self.etaB_S[i+1, :]/10 * B
            input_concat = torch.cat((B_mid, FB), dim=1)
            B, FB = self.proxNet_B_S[i+1](input_concat) # 32
            ListB.append(B)

        B_adjust, _ = self.proxNet_B_last_layer(torch.cat((B, FB), dim=1))
        ListB.append(B_adjust)

        output = (ListB[-1],)

        loss_Bs = 0.
        loss_Rs = 0.
        if labels is not None:
            for j in range(self.iter):
                loss_Bs = float(loss_Bs) + F.mse_loss(ListB[j], labels)
                loss_Rs = float(loss_Rs) + F.mse_loss(ListR[j], im_rain - labels)
            lossB = F.mse_loss(ListB[-1], labels)
            lossR = F.mse_loss(ListR[-1], im_rain - labels)

            loss = loss_Bs + loss_Rs + lossB + lossR
            output = (loss,) + output

        return output, ListR[-1]

class Mnet(nn.Module):
    def __init__(self, args):
        super(Mnet, self).__init__()
        self.channels = args.num_M
        self.T = args.T                                           # the number of resblocks in each proxNet
        self.layer_m= self.make_resblock(self.T)

        self.tau0 = torch.Tensor([0.5])
        self.tau_const = self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1,self.channels,-1,-1)
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)  # for sparse rain map

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            if i+1 == T:
                layers.append(nn.Sequential(

                CSP_Bottleneck(self.channels, self.channels, 3, activation_type= nn.ReLU),

                CSP_GLRA(self.channels, self.channels, 1, activation_type= nn.ReLU)
                          ))
            else:
                layers.append(nn.Sequential(
                    CSP_Bottleneck(self.channels, self.channels, 3, activation_type= nn.ReLU),
                              ))

        return nn.Sequential(*layers)

    def forward(self, input):
        M = input
        for i in range(self.T):
            M = nn.ReLU(inplace=False)(M + self.layer_m[i](M))
        M = nn.ReLU(inplace=False)(M-self.tau)
        return M

class Bnet(nn.Module):
    def __init__(self, args):
        super(Bnet, self).__init__()
        self.channels = args.num_Z+3 # 3 means R,G,B channels for color image 32+3
        self.T = args.T
        self.layer_b= self.make_resblock(self.T)
        self.relu = relu()
        self.B32 = nn.Conv2d(self.channels, args.num_Z, kernel_size=3, stride=1, padding=1, dilation=1)

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            if i+1 == T:
                layers.append(nn.Sequential(

                CSP_Bottleneck(self.channels, self.channels, 3, activation_type= nn.ReLU),
                
                CSP_ACMLP(self.channels, self.channels, 1, activation_type= nn.ReLU)
                          ))
            else:
                layers.append(nn.Sequential(
                    CSP_Bottleneck(self.channels, self.channels, 3, activation_type= nn.ReLU),
                              ))

        return nn.Sequential(*layers)

    def forward(self, input):
        B = input
        for i in range(self.T):
            B = self.relu(B + self.layer_b[i](B)) 
        return B[:, :3, :, :], self.relu(self.B32(B))

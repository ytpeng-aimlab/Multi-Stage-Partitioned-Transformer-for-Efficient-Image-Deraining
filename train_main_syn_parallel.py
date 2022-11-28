#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by HongW 2020-08-05
# This is a simple coding framework. The original framework for CVPR2020 RCDNet is at https://github.com/hongwang01/RCDNet
# CVPR2020  A Model-driven Deep Neural Network for Single Image Rain Removal
from __future__ import print_function
import argparse
import os
import cv2
import random
import json
import time
import numpy as np
from math import ceil
import torch.nn as nn
import torch.nn.functional as  F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import multiprocessing
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from Network import MPT
from torch.utils.data import DataLoader
from DerainDataset import TrainDataset

from ptflops import get_model_complexity_info
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

parser = argparse.ArgumentParser() # Rain100L Rain800 Rain1400 SPA-Data_6385 Rain1200 Rain100H RainDS
parser.add_argument("--data_path",type=str, default=r"/home/wenyi_peng/MPT/data/Rain100L/train/rain/",help='path to training input data')
parser.add_argument("--gt_path",type=str, default=r"/home/wenyi_peng/MPT/data/Rain100L/train/norain/",help='path to training gt data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=32)
parser.add_argument('--batchSize', type=int, default=12, help='input batch size')
parser.add_argument('--patchSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=100, help='total number of training epochs')
parser.add_argument('--num_M', type=int, default=32, help='the number of rain maps')
parser.add_argument('--num_Z', type=int, default=32, help='the number of dual channels')
parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every CSP_ResBlock')
parser.add_argument('--S', type=int, default=20, help='the number of iterative stages in MPT')
parser.add_argument('--resume', type=int, default=-1, help='continue to train from epoch')
parser.add_argument("--milestone", type=int, default=[25,50,75], help="When to decay learning rate")
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0,1,2,3", help='GPU id')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--log_dir', default='./checkpoints/Rain100L_onlyLocal_ad/', help='tensorboard logs')
parser.add_argument('--model_dir',default='./checkpoints/Rain100L_onlyLocal_ad/',help='saving model')
parser.add_argument('--manualSeed', type=int, default='6488', help='manual seed')
opt = parser.parse_args()

torch.cuda.empty_cache()

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'

if opt.use_gpu:

    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_id
    torch.cuda.set_device(opt.local_rank)
    device = torch.device('cuda', opt.local_rank)
    
    rank = opt.local_rank  # or: rank = int(os.environ['RANK'])
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    
    torch.distributed.barrier()

try:
    os.makedirs(opt.model_dir)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True
ppp=0
def train_model(net, optimizer, lr_scheduler, datasets):
    
    train_sampler = DistributedSampler(datasets)
    data_loader = DataLoader(datasets, sampler=train_sampler, batch_size=opt.batchSize, num_workers=int(opt.workers), pin_memory=True, prefetch_factor=2)

    num_data = len(datasets)
    num_iter_epoch = ceil(num_data / opt.batchSize)
    writer = SummaryWriter(opt.log_dir)
    step = 0
    for epoch in range(opt.resume, opt.niter):
        train_sampler.set_epoch(epoch)

        mse_per_epoch = 0
        tic = time.time()
        # train 
        lr = optimizer.param_groups[0]['lr']
        for ii, data in enumerate(data_loader):
            im_rain, im_gt = [x.cuda() for x in data]
            net.train()
            optimizer.zero_grad()
            outputs, _ = net(im_rain, labels=im_gt)
            loss = outputs[0]
            out = outputs[-1]

            # back propagation
            loss.backward()
            optimizer.step()
            mse_iter = loss.item()
            mse_per_epoch = mse_per_epoch + mse_iter

            if ii % 300 == 0:
                out = torch.clamp(out, 0., 255.)
                pre_out = np.uint8(out.data.cpu().numpy().squeeze())
                gt_np = np.uint8(im_gt.data.cpu().numpy().squeeze())
                psnr=[]
                for i in range(len(pre_out)):
                    psnr.append(compare_psnr(pre_out[i].transpose(1, 2, 0), gt_np[i].transpose(1, 2, 0), data_range=255))
                print("PSNR: ",sum(psnr)/len(psnr))
                template = '[Epoch:{:>2d}/{:<2d}] {:0>5d}/{:0>5d}, Loss={:5.2e}, lr={:.2e}'
                print(template.format(epoch+1, opt.niter, ii, num_iter_epoch, mse_iter, lr))
                writer.add_scalar('train Loss Iter', mse_iter, step)
                writer.add_scalar('val PSNR Iter', sum(psnr)/len(psnr), step)
            step = step + 1
        mse_per_epoch /= (ii+1)
        print('Epoch:{:>2d}, Derain_Loss={:+.2e}'.format(epoch + 1, mse_per_epoch))
        # adjust the learning rate
        lr_scheduler.step()
        # save model
        model_prefix = 'DerainNet_state_'
        save_path_model = os.path.join(opt.model_dir, model_prefix+str(epoch+1)+'.pt')
        torch.save(net.state_dict(), save_path_model)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc-tic))
        print('-' * 100)
    writer.close()
    print('Reach the maximal epochs! Finish training')

if __name__ == '__main__':
    netDerain = DPASKNet(opt).cuda()
    if torch.cuda.device_count():
        torch.cuda.set_device(opt.local_rank)
        netDerain = nn.SyncBatchNorm.convert_sync_batchnorm(netDerain)
        netDerain = nn.parallel.DistributedDataParallel(netDerain, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)

    macs, params = get_model_complexity_info(netDerain, (3, 64, 64), as_strings=True, flops_units="GMac", param_units="M",
                                   print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # assert False
    optimizerDerain = optim.Adam(netDerain.parameters(), lr=opt.lr)
    schedulerDerain = optim.lr_scheduler.MultiStepLR(optimizerDerain, milestones=opt.milestone, gamma=0.2)  # learning rates

    # from opt.resume continue to train
    for _ in range(opt.resume):
        schedulerDerain.step()
    if opt.resume != -1:
        if opt.resume: #_pretrained
            checkpoint = torch.load(os.path.join(opt.model_dir, 'model_' + str(opt.resume)), map_location=device)
            state_dict_new = torch.load(os.path.join(opt.model_dir, 'DerainNet_state_' + str(opt.resume) + '.pt'), map_location=device)

        else:
            model_dict = netDerain.state_dict()
            pretrained_dict = torch.load(os.path.join(opt.model_dir, 'DerainNet_state_pretrained' + '.pt'), map_location=device)
            state_dict_new = dict({})
            for key in pretrained_dict.keys():
                if key in model_dict.keys():
                    if 'etaM_S' in key or 'etaB_S' in key:
                        # continue
                        # print("====================================================================")
                        # print(len(model_dict[key]), len(pretrained_dict[key]))
                        # assert False
                        model_dict[key][:len(pretrained_dict[key])] = pretrained_dict[key] # 20/19
                        # model_dict[key] = pretrained_dict[key][:len(model_dict[key])]
                        # model_dict[key] = pretrained_dict[key][:-1]
                        state_dict_new[key] = model_dict[key]
                    else:
                        state_dict_new[key] = pretrained_dict[key]

        model_dict = netDerain.state_dict()
        model_dict.update(state_dict_new)
        netDerain.load_state_dict(model_dict)

    # load dataset
    train_dataset = TrainDataset(opt.data_path, opt.gt_path, opt.patchSize, opt.batchSize * 1500 )#
    train_model(netDerain, optimizerDerain, schedulerDerain, train_dataset)

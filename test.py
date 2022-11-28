import cv2
import os
import argparse
import glob
import json
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Network import MPT
import time
from ptflops import get_model_complexity_info
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import multiprocessing
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


parser = argparse.ArgumentParser(description="RCDNet_Test") # ablation Rain100H Rain1400 Rain800 SPA-Data_6385 Rain100L Rain1200
parser.add_argument("--model_dir", type=str, default="./checkpoints/Rain100L_onlyLocal_v", help='path to model files')
parser.add_argument("--data_path", type=str, default=r"/home/wenyi_peng/MPT/data/Rain100L/val/rain/", help='path to testing data')
parser.add_argument("--gt_path", type=str, default=r"/home/wenyi_peng/MPT/data/Rain100L/val/norain/", help='path to testing data')
parser.add_argument('--num_M', type=int, default=32, help='the number of rain maps')
parser.add_argument('--num_Z', type=int, default=32, help='the number of dual channels')
parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every CSP_ResBlock')
parser.add_argument('--S', type=int, default=20, help='the number of iterative stages in MPT')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--save_path", type=str, default="./results/Rain100L_onlyLocal_v", help='path to derained results')

opt = parser.parse_args()
torch.cuda.empty_cache()
try:
    os.makedirs(opt.save_path)
except OSError:
    pass

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_id
    torch.cuda.set_device(opt.local_rank)
    device = torch.device('cuda', opt.local_rank)

    rank = opt.local_rank  # or: rank = int(os.environ['RANK'])
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    torch.distributed.barrier()

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

def main():
    
    # Build model
    print('Loading model ...\n', opt.save_path)
    model = MPT(opt).cuda()
    macs, params = get_model_complexity_info(model, (3, 64, 64), as_strings=True, flops_units="GMac", param_units="M",
                                   print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # assert False
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
        if torch.cuda.device_count():
            torch.cuda.set_device(opt.local_rank)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)

    psnrs_ = []
    for epoch in (range(80,20, -5)):
        torch.cuda.empty_cache()
        save_path = opt.save_path+'/epoch_'+str(epoch)+'/'
        try:
            os.makedirs(opt.save_path+'/epoch_'+str(epoch)+'/')
        except OSError:
            pass
        psnrs = []
        model.load_state_dict(torch.load(os.path.join(opt.model_dir, 'DerainNet_state_'+str(epoch)+'.pt'), map_location=device))
        model.eval()

        time_test = 0
        count = 0
        for img_name in (os.listdir(opt.data_path)):
            if is_image(img_name):
                img_path = os.path.join(opt.data_path, img_name) 
                # gt_img_path = os.path.join(opt.gt_path, img_name) #rain800
                gt_img_path = os.path.join(opt.gt_path, 'no'+img_name)
                # gt_img_path = os.path.join(opt.gt_path, img_name.split('jpg')[0]+'png') # rain1400

                O = cv2.imread(img_path)
                if 'Rain1200' in opt.data_path:
                    h, w, c = O.shape
                    width_cutoff = int(w/2)
                    gt = O[:, width_cutoff:, :]
                    O = O[:, :width_cutoff, :]
                else:
                    gt = cv2.imread(gt_img_path)

                b, g, r = cv2.split(O)
                O = cv2.merge([r, g, b])
                O = np.expand_dims(O.transpose(2, 0, 1), 0)
                O = Variable(torch.Tensor(O))
                if opt.use_GPU:
                    O = O.cuda()
                with torch.no_grad():
                    torch.cuda.synchronize()
                    start_time = time.time()
                    outputs,R = model(O)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    dur_time = end_time - start_time
                    time_test += dur_time
                    out = outputs[-1]

                    del O
                    out = torch.clamp(out, 0., 255.)
                    # print(img_name, ': ', dur_time)
                if opt.use_GPU:
                    save_out = np.uint8(out.data.cpu().numpy().squeeze())   #back to cpu
                else:
                    save_out = np.uint8(out.data.numpy().squeeze())
                save_out = save_out.transpose(1, 2, 0)
                b, g, r = cv2.split(save_out)
                save_out = cv2.merge([r, g, b])
                psnrs.append(compare_psnr(save_out, gt, data_range=255))
                cv2.imwrite(os.path.join(save_path, img_name), save_out)

                count += 1
        print('Avg. time:', time_test/count)
        print(epoch, " Avg. PSNR: ", sum(psnrs)/len(psnrs))
        psnrs_.append(sum(psnrs)/len(psnrs))
    print("MAX: ", max(psnrs_), np.argmax(psnrs_))
if __name__ == "__main__":
    main()


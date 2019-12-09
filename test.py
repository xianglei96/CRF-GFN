from __future__ import print_function
import argparse
import os
import time
from math import log10
from os.path import join
from torchvision import transforms
from torchvision import utils as utils
import torch
from torch.utils.data import DataLoader
from datasets.dataset_hf5 import DataValSet
import statistics
import re
import torch.nn as nn

import torch.nn.parallel


parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--gated", type=bool, default=False, help="Activated gate module")
parser.add_argument("--isTest", type=bool, default=True, help="Test or not")
parser.add_argument('--dataset', type=str, default='', help='Path of the validation dataset')
parser.add_argument("--intermediate_process", default="", type=str, help="Test on intermediate pkl (default: none)")
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--name', type=str, default='ICONIP', help='filename of the training models')
parser.add_argument("--train_step", type=int, default=0, help="Activated gate module")
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_set=[
    {'gated':False},
    {'gated':False},
    {'gated':True}
]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def is_pkl(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def which_trainingstep_epoch(resume):
    trainingstep = "".join(re.findall(r"\d", resume)[1])
    start_epoch = "".join(re.findall(r"\d", resume)[2:])
    return int(trainingstep), int(start_epoch)


def test(test_gen, model, criterion, SR_dir):
    avg_psnr = 0
    med_time = []

    with torch.no_grad():
        for iteration, batch in enumerate(test_gen, 1):
            print(iteration)
            Blur = batch[0]
            HR = batch[1]
            #Blur = Blur.to(device)
            #HR = HR.to(device)
            Blur = Blur.cuda()
            HR = HR.cuda()


            start_time = time.perf_counter()#-------------------------begin to deal with an image's time
            sr = model(Blur)
            #modify
            sr = torch.clamp(sr, min=0, max=1)
            torch.cuda.synchronize()#wait for CPU & GPU time syn
            evalation_time = time.perf_counter() - start_time#---------finish an image
            med_time.append(evalation_time)

            #resultSRDeblur = transforms.ToPILImage()(sr.cpu()[0])
            #resultSRDeblur.save(join(SR_dir, '{0:04d}_{1}.png'.format(iteration, opt.name)))
            
            mse = criterion(sr, HR)
            psnr = 10 * log10(1 / mse)
            print("Processing {}:  {}".format(iteration, psnr))
            avg_psnr += psnr

        print("Avg. SR PSNR:{:4f} dB".format(avg_psnr / iteration))
        median_time = statistics.median(med_time)
        print(median_time)
        return avg_psnr / iteration

def model_test(model):
    #model = model.to(device)
    model = model.cuda()
    if opt.gpu_ids and len(opt.gpu_ids) > 1:
            model = torch.nn.DataParallel(model, opt.gpu_ids)
    criterion = torch.nn.MSELoss(size_average=True)
    #criterion = criterion.to(device)
    criterion = criterion.cuda()
    print(opt)
    psnr = test(testloader, model, criterion, SR_dir)
    return psnr

opt = parser.parse_args()
#device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
#str_ids = opt.gpu_ids.split(',')
#torch.cuda.set_device(int(str_ids[0]))
if torch.cuda.is_available():
    opt.gpu_ids = [int(i) for i in opt.gpu_ids.split(',')]
    torch.cuda.set_device(int(opt.gpu_ids[0]))
else:
    opt.gpu_ids = None

    
root_val_dir = opt.dataset# #----------Validation path
SR_dir = join(root_val_dir, opt.name)  #--------------------------SR results save path
isexists = os.path.exists(SR_dir)
if not isexists:
    os.makedirs(SR_dir)
print("The results of testing images sotre in {}.".format(SR_dir))

testloader = DataLoader(DataValSet(root_val_dir), batch_size=1, shuffle=False, pin_memory=False)
print("===> Loading model and criterion")

if is_pkl(opt.intermediate_process):
    test_pkl = opt.intermediate_process
    if is_pkl(test_pkl):
        print("Testing model {}----------------------------------".format(opt.intermediate_process))
        train_step, epoch = which_trainingstep_epoch(opt.intermediate_process)
        model = torch.load(test_pkl, map_location=lambda storage, loc: storage).module
        print(get_n_params(model))
        #model = model.eval()
        model_test(model)
    else:
        print("It's not a pkl file. Please give a correct pkl folder on command line for example --opt.intermediate_process /models/1/GFN_epoch_25.pkl)")
else:
    test_list = [x for x in sorted(os.listdir(opt.intermediate_process)) if is_pkl(x)]
    print("Testing on the given 3-step trained model which stores in /models, and ends with pkl.")
    Results = []
    Max = {'max_psnr':0, 'max_epoch':0}
    for i in range(len(test_list)):
        print("Testing model is {}----------------------------------".format(test_list[i]))
        model = torch.load(join(opt.intermediate_process, test_list[i]), map_location=lambda storage, loc: storage).module
        print(get_n_params(model))
        model = model.eval()
        psnr = model_test(model)
        Results.append({'epoch':"".join(re.findall(r"\d", test_list[i])[:]), 'psnr':psnr})
        if psnr > Max['max_psnr']:
            Max['max_psnr'] = psnr
            Max['max_epoch'] = "".join(re.findall(r"\d", test_list[i])[:])

        for Result in Results:
            print(Result)
        print('The Best Result is : ============================>')
        print(Max)




import os
import datetime

import numpy as np
import torch

def set_gpu(opt):
    if opt.use_cuda and torch.cuda.is_available():
        print("Setting GPU")
        print("===> CUDA Available: ", torch.cuda.is_available())
        opt.use_cuda = True
        opt.device = 'cuda'
    else:
        opt.use_cuda = False
        opt.device = 'cpu'

    if opt.use_cuda and torch.cuda.device_count() > 1 and opt.multi_gpu:
        print("===> Use " + str(torch.cuda.device_count()) + " GPUs")
        opt.multi_gpu = True
    else:
        opt.multi_gpu = False

    num_gpus = torch.cuda.device_count()
    opt.gpu_ids = []
    for id in range(num_gpus):
        opt.gpu_ids.append(id)
    print("GPU IDs:", opt.gpu_ids)

    return opt

def set_checkpoint_dir(opt):
    dt = datetime.datetime.now()
    date = dt.strftime("%Y%m%d")

    dataset_name = ''
    for d in opt.train_datasets:
        dataset_name = dataset_name + d
    model_opt = dataset_name  + "-" + date + "-" + opt.model + '-patch' + str(opt.patch_size)
    
    opt.checkpoint_dir = os.path.join(opt.checkpoint_dir, model_opt)

def set_test_dir(opt):
    model_opt = os.path.basename(opt.checkpoint_dir)

    if opt.test_patches:
        test_dir_opt = model_opt + '-patch_offset' + str(opt.patch_offset)
    else:
        test_dir_opt = model_opt + "-image"

    if opt.ensemble:
        test_dir_opt = test_dir_opt + "-ensemble"

    #for linux server
    if os.name == 'posix': #linux
        test_result_dir = opt.test_result_dir.split('\\')
        '/'.join(test_result_dir)
    elif os.name == 'nt': #window
        pass

    opt.test_result_dir = os.path.join(opt.test_result_dir , test_dir_opt)

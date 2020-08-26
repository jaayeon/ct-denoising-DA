import os, time, scipy.io, shutil, glob
# import re
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from models import set_model
from utils.saver import load_model, save_checkpoint, save_config
from utils.helper import set_gpu, set_checkpoint_dir

def run_train(opt, training_dataloader, valid_dataloader):

    # check gpu setting with opt arguments
    opt = set_gpu(opt)
    
    print('Initialize networks for training')
    net = set_model(opt)
    print(net)

    # if opt.use_cuda:
    #     net = net.to(opt.device)
    
    print("Setting Optimizer")
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
        print("===> Use Adam optimizer")
    
    if opt.resume:
        opt.start_epoch, net, optimizer = load_model(opt, net, optimizer=optimizer)
    else:
        set_checkpoint_dir(opt)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    log_file = os.path.join(opt.checkpoint_dir, opt.model + "_log.csv")
    opt_file = os.path.join(opt.checkpoint_dir, opt.model + "_opt.txt")
    
    if opt.multi_gpu:
        net = nn.DataParallel(net)
    
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    # Setting loss function
    if opt.loss == 'l1':
        loss_criterion = nn.L1Loss()
    elif opt.loss == 'l2':
        loss_criterion = nn.MSELoss()
    else:
        raise ValueError("Please specify correct loss function")

    if opt.use_cuda:
        loss_criterion = loss_criterion.to(opt.device)

    # Create log file when training start
    if opt.start_epoch == 1:
        with open(log_file, mode='w') as f:
            f.write("epoch,train_loss,valid_loss\n")
        save_config(opt)

    dataloader = {
        'train': training_dataloader,
        'valid': valid_dataloader
    }
    modes = ['train', 'valid']

    mse_criterion = nn.MSELoss()
    
    if opt.use_cuda:
        mse_criterion = mse_criterion.to(opt.device)

    print('train_dir : {}\ntest_dir : {}\nimg_dir : {}\ngt_img_dir : {}'.format(opt.train_dir, opt.test_dir, opt.img_dir, opt.gt_img_dir))

    current_best_psnr = 0.0
    for epoch in range(opt.start_epoch, opt.n_epochs):
        opt.epoch_num = epoch
        for phase in modes:
            total_loss = 0.0
            total_psnr = 0.0

            if phase == 'train':
                net.train()
            else:
                net.eval()
                
            mode = "Training" if phase == 'train' else "Validation"
            print("*** %s ***"%(mode))
            start_time = time.time()
            for iteration, batch in enumerate(dataloader[phase], 1):
                x, target = batch[0], batch[1]
                
                if opt.use_cuda:
                    x = x.to(opt.device)
                    target = target.to(opt.device)

                with torch.set_grad_enabled(phase=='train'):
                    optimizer.zero_grad()

                    out = net(x)
                    loss = loss_criterion(out, target)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                    total_loss += loss.item()

                # print("max(out):", torch.max(out))
                # print("min(out):", torch.min(out))
                mse_loss = mse_criterion(out, target)
                psnr = 10 * math.log10(1 / mse_loss.item())
                total_psnr += psnr

                print("%s %.2fs => Epoch[%d/%d](%d/%d): Loss: %.10f PSNR: %.5f" %
                    (mode, time.time() - start_time, opt.epoch_num, opt.n_epochs, iteration, len(dataloader[phase]), loss.item(), psnr))

            epoch_avg_loss = total_loss / iteration
            epoch_avg_psnr = total_psnr / iteration

            if phase == 'train':
                train_loss = epoch_avg_loss
                train_psnr = epoch_avg_psnr
            else:
                valid_loss = epoch_avg_loss
                valid_psnr = epoch_avg_psnr
                scheduler.step(valid_loss)
                print("Valid LOSS avg : {:5f}\nValid PSNR avg : {:5f}".format(valid_loss, valid_psnr))

        with open(log_file, mode='a') as f:
            f.write("%d,%08f,%08f,%08f,%08f\n" % (
                epoch,
                train_loss,
                train_psnr,
                valid_loss,
                valid_psnr
            ))

        if current_best_psnr < valid_psnr : 
            save_checkpoint(opt, net, optimizer, epoch, valid_loss)
            current_best_psnr = valid_psnr


    
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
from data.mask import Masker

def run_train(opt, n2s_t_loader, n2s_v_loader):

    # check gpu setting with opt arguments
    opt = set_gpu(opt)
    
    print('Initialize networks for training_noise2self')
    net = set_model(opt)
    print(net)

    # if opt.use_cuda:
    #     net = net.to(opt.device)
    
    print("Setting Optimizer")

    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        print("===> Use Adam optimizer")

    if opt.resume:
        opt.start_epoch, net, optimizer = load_model(opt, net, optimizer=optimizer)
    else:
        set_checkpoint_dir(opt)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    log_file = os.path.join(opt.checkpoint_dir, opt.way + "_log.csv")
    opt_file = os.path.join(opt.checkpoint_dir, opt.way + "_opt.txt")
    
    if opt.multi_gpu:
        net = nn.DataParallel(net)
    
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    
    masker = Masker(width=4, mode = 'interpolate')

    # Create log file when training start
    if opt.start_epoch == 1:
        with open(log_file, mode='w') as f:
            f.write("epoch, train_loss, train_psnr, valid_loss, valid_psnr\n")
        save_config(opt)

    mse_criterion = nn.MSELoss()
    
    if opt.use_cuda:
        mse_criterion = mse_criterion.to(opt.device)

    print('train_dir : {}\ntest_dir : {}\nimg_dir : {}\ngt_img_dir : {}'.format(opt.train_dir, opt.test_dir, opt.img_dir, opt.gt_img_dir))

    current_best_psnr = 0.0

    for epoch in range(opt.start_epoch, opt.n_epochs):
        opt.epoch_num = epoch
        train_loss = 0.0
        train_psnr = 0.0

        net.train()

        print("*** Training ***")
        start_time = time.time()

        for iteration_t, batch in enumerate(n2s_t_loader, 1):
            noisy, clean = batch[0], batch[1]

            if opt.use_cuda:
                noisy = noisy.to(opt.device)
                clean = clean.to(opt.device)
            
            input, mask = masker.mask(noisy, iteration_t)
            output = net(input).to(opt.device)

            loss_n2s = mse_criterion(output*mask, noisy*mask)
                
            optimizer.zero_grad()
            loss_n2s.backward()
            optimizer.step()

            # x, target = batch[0], batch[1]  

            # with torch.set_grad_enabled(phase=='train'):
            #     optimizer.zero_grad()

            #     out = net(x)
            #     loss = loss_criterion(out, target)

            #     if phase == 'train':
            #         loss.backward()
            #         optimizer.step()
                        
            train_loss += loss_n2s

            mse_loss = mse_criterion(output, clean)
            psnr = 10 * math.log10(1 / mse_loss.item())
            train_psnr += psnr

            print("%s %.2fs => Epoch[%d/%d](%d/%d): Loss: %.7f PSNR: %.5f" %
                ('Training', time.time() - start_time, opt.epoch_num, opt.n_epochs, iteration_t, len(n2s_t_loader), loss_n2s, psnr))
        print("Mayo avg_Loss : %.5f Mayo avg_PSNR : %.5f"%(train_loss/iteration_t, train_psnr/iteration_t))
        
        valid_loss = 0.0
        valid_psnr = 0.0

        net.eval()
      
        print("***Validation***")
        start_valid = time.time()
    
        for iteration_v, batch in enumerate(n2s_v_loader, 1):
            noisy, clean = batch[0], batch[1]

            if opt.use_cuda:
                noisy = noisy.to(opt.device)
                clean = clean.to(opt.device)
            
            input, mask = masker.mask(noisy, iteration_t)
            
            with torch.no_grad():
                output = net(input).to(opt.device)
                loss_n2s = mse_criterion(output*mask, noisy*mask)
                valid_loss += loss_n2s

                mse_loss = mse_criterion(output, clean)
                nmse_loss = mse_criterion(input, clean)

                psnr = 10 * math.log10(1 / mse_loss.item())
                npsnr = 10 * math.log10(1 / nmse_loss.item())
                valid_psnr += psnr

            print("%s %.2fs => Epoch[%d/%d](%d/%d): valid_loss : %.5f noise PSNR : %.5f PSNR : %.5f "%(
                'Validation', time.time()-start_valid, epoch, opt.n_epochs, iteration_v, len(n2s_v_loader), loss_n2s, npsnr, psnr))
    
        print("Mayo avg_Loss : %.5f Mayo avg_PSNR : %.5f"%(valid_loss/iteration_v, valid_psnr/iteration_v))
        
        train_loss = train_loss/iteration_t
        train_psnr = train_psnr/iteration_t
        valid_loss = valid_loss/iteration_v
        valid_psnr = valid_psnr/iteration_v

        with open(log_file, mode='a') as f:
            f.write("%d,%08f,%08f,%08f,%08f"%(
                epoch,
                train_loss,
                train_psnr,
                valid_loss,
                valid_psnr
            ))

        if current_best_psnr < valid_psnr : 
            save_checkpoint(opt, net, optimizer, epoch, valid_loss)
            current_best_psnr = valid_psnr

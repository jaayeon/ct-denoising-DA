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

def run_train(opt, n2v_t_loader,n2v_v_loader):

    # check gpu setting with opt arguments
    opt = set_gpu(opt)
    
    print('Initialize networks for training')
    net = set_model(opt)
    print(net)

    # if opt.use_cuda:
    #     net = net.to(opt.device)
    
    print("Setting Optimizer")

    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0004, betas=(0.5, 0.999))
        print("===> Use Adam optimizer")
    
    if opt.resume:
        #not possible
        opt.start_epoch, net, optimizer = load_model(opt, net, optimizer=optimizer)
        # _, net_D, optimizer_d = load_model(opt, net_D, optimizer_g=optimizer_d)
    else:
        set_checkpoint_dir(opt)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    log_file = os.path.join(opt.checkpoint_dir, opt.model + "_log.csv")
    opt_file = os.path.join(opt.checkpoint_dir, opt.model + "_opt.txt")
    
    if opt.multi_gpu:
        net = nn.DataParallel(net)

    
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')
    # scheduler = StepLR(optimizer_g, step_size=50, gamma=0.5)

    # Create log file when training start
    if opt.start_epoch == 1:
        with open(log_file, mode='w') as f:
            f.write("epoch, gloss_t, pxloss_t, ploss_t, fgloss_t, dloss_t, gploss_t, advloss_t, dmgploss_t, psnr_t, gloss_v, pxloss_v, ploss_v, fgloss_v, dloss_v, gploss_v, advloss_v, dmgploss_v, psnr_v\n")
        save_config(opt)

    mse_criterion = nn.MSELoss()
    fn_REG = nn.MSELoss()
    
    if opt.use_cuda:
        mse_criterion = mse_criterion.to(opt.device)

    print('train_dir : {}\ntest_dir : {}\nimg_dir : {}\ngt_img_dir : {}'.format(opt.train_dir, opt.test_dir, opt.img_dir, opt.gt_img_dir))

    current_best_psnr = 0.0
    for epoch in range(opt.start_epoch, opt.n_epochs):
        opt.epoch_num = epoch
        train_loss = 0.0
        valid_loss = 0.0
        train_psnr = 0.0

        net.train()

        print("*** Training ***")
        start_time = time.time()
        for iteration_t, batch in enumerate(n2v_t_loader, 1):
            img,label,mask,lbl = batch['input'], batch['label'], batch['mask'], batch['lbl']

            if opt.use_cuda:
                img,label,mask,lbl = img.to(opt.device),label.to(opt.device),mask.to(opt.device),lbl.to(opt.device)

            #discriminator
            optimizer.zero_grad()
           
            out = net(img).cuda()

            loss_G = fn_REG(out * (1 - mask), label * (1 - mask))

            loss_G.backward()
            optimizer.step()
    
            train_loss += loss_G
    
            # print("max(out):", torch.max(out))
            # print("min(out):", torch.min(out))
            mse_loss = mse_criterion(out, lbl)
            psnr = 10 * math.log10(1 / mse_loss.item())
            train_psnr += psnr

            print("%s %.2fs => Epoch[%d/%d](%d/%d): Loss: %.7f PSNR: %.5f" %
                ('Training', time.time() - start_time, opt.epoch_num, opt.n_epochs, iteration_t, len(n2v_t_loader), loss_G, psnr))
        print("Mayo avg_Loss : %.5f Mayo avg_PSNR : %.5f"%(train_loss/iteration_t, train_psnr/iteration_t))
        
        valid_psnr = 0.0
        valid_loss = 0.0
        start_valid = time.time()
        print("***Validation***")
    
        for iteration_v, batch in enumerate(n2v_v_loader, 1):
            img,label,mask,lbl = batch['input'], batch['label'], batch['mask'], batch['lbl']

            if opt.use_cuda:
                img,label,mask,lbl = img.to(opt.device),label.to(opt.device),mask.to(opt.device),lbl.to(opt.device)
            
            with torch.no_grad():
                net.eval()
                out = net(img)
                loss_G = fn_REG(out * (1 - mask), label * (1 - mask))
                valid_loss += loss_G
                mse_loss = mse_criterion(out, lbl)
                nmse_loss = mse_criterion(img, lbl)
                psnr = 10 * math.log10(1 / mse_loss.item())
                npsnr = 10 * math.log10(1 / nmse_loss.item())
                valid_psnr += psnr

            print("%s %.2fs => Epoch[%d/%d](%d/%d): valid_loss : %.5f noise PSNR : %.5f PSNR : %.5f "%(
                'Validation', time.time()-start_valid, epoch, opt.n_epochs, iteration_v, len(n2v_v_loader), loss_G, npsnr, psnr))
    
        print("Mayo avg_Loss : %.5f Mayo avg_PSNR : %.5f"%(valid_loss/iteration_v, valid_psnr/iteration_v))
        

        train_psnr = train_psnr/iteration_t
        train_loss = train_loss/iteration_t
        valid_psnr = valid_psnr/iteration_v
        valid_loss = valid_loss/iteration_v

        with open(log_file, mode='a') as f:
            f.write("%d,%08f,%08f,%08f,%08f"%(
                epoch,
                train_loss,
                valid_loss,
                train_psnr,
                valid_psnr
            ))

        if current_best_psnr < valid_psnr : 
            save_checkpoint(opt, net, optimizer, epoch, valid_loss)
            current_best_psnr = valid_psnr
            
            #gloss, ploss, fgloss, dloss, gploss, advloss, domain_gploss
            valid_loss += valid_loss
            # print("max(out):", torch.max(out))
            # print("min(out):", torch.min(out))

            print("%s %.2fs => Epoch[%d/%d]: train_PSNR: %.5f train_loss: %.5f valid_loss: %.5f valid_PSNR: %.5f " %('Validation', 
            time.time() - start_time, opt.epoch_num, opt.n_epochs, train_psnr, train_loss, valid_psnr, valid_loss ))



    
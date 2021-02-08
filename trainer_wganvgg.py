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
from utils.loader import load_model
from utils.saver import Record
from utils.helper import set_gpu, set_checkpoint_dir

def run_train(opt, training_dataloader, valid_dataloader):

    # check gpu setting with opt arguments
    opt = set_gpu(opt)
    
    print('Initialize networks for training')
    net = set_model(opt)
    print(net)

    print("Setting Optimizer")
    if opt.optimizer == 'adam':
        optimizer_g = optim.Adam(net.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
        optimizer_d = optim.Adam(net.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
        print("===> Use Adam optimizer_g")
    
    if opt.resume:
        #not possible
        opt.start_epoch, net, optimizer_g = load_model(opt, net, optimizer=optimizer_g)
    else:
        set_checkpoint_dir(opt)
    
    if opt.multi_gpu:
        net = nn.DataParallel(net)
    
    scheduler = ReduceLROnPlateau(optimizer_g, factor=0.5, patience=5, mode='min')
    # scheduler = StepLR(optimizer_g, step_size=50, gamma=0.5)

    # Create log file when training start
    if opt.start_epoch == 1:
        keys = ['gloss', 'pxloss', 'ploss', 'dloss', 'gploss', 'psnr']
        record = Record(opt, train_length=len(training_dataloader), valid_length=len(valid_dataloader), keys=keys)
    
    dataloader = {
        'train': training_dataloader,
        'valid': valid_dataloader
    }
    modes = ['train', 'valid']

    mse_criterion = nn.MSELoss()
    if opt.use_cuda:
        mse_criterion = mse_criterion.to(opt.device)

    print('train_dir : {}\ntest_dir : {}\nimg_dir : {}\ngt_img_dir : {}'.format(opt.train_dir, opt.test_dir, opt.img_dir, opt.gt_img_dir))

    for epoch in range(opt.start_epoch, opt.n_epochs):
        opt.epoch_num = epoch
        for phase in modes:

            if phase == 'train':
                net.generator.train()
                net.discriminator.train()
            else:
                net.generator.eval()
                net.discriminator.eval()
                
            mode = "Training" if phase == 'train' else "Validation"
            print("*** %s ***"%(mode))
            start_time = time.time()
            for iteration, batch in enumerate(dataloader[phase], 1):
                x, target = batch[0], batch[1]
                if opt.use_cuda:
                    x = x.to(opt.device)
                    target = target.to(opt.device)
                
                if phase == 'train':
                    #discriminator
                    optimizer_d.zero_grad()
                    net.discriminator.zero_grad()
                    for _ in range(opt.n_d_train):
                        d_loss, gp_loss = net.d_loss(x,target,gp=True, return_gp=True)
                        d_loss.backward()
                        optimizer_d.step()
                    #generator, perceptual loss
                    optimizer_g.zero_grad()
                    net.generator.zero_grad()
                    g_loss, px_loss, p_loss, _ = net.g_loss(x,target,perceptual=True, return_p=True, pixel_wise=True)
                    g_loss.backward()
                    optimizer_g.step()

                elif phase == 'valid':
                    d_loss, gp_loss = net.d_loss(x,target,gp=True,return_gp=True)
                    g_loss, px_loss, p_loss, _ = net.g_loss(x,target,perceptual=True,return_p=True, pixel_wise=True)

                out = net.fake
                # print("max(out):", torch.max(out))
                # print("min(out):", torch.min(out))
                mse_loss = mse_criterion(out, target)
                psnr = 10 * math.log10(1 / mse_loss.item())

                #generator loss, perceptual loss, discriminator loss, gradient penalty loss
                status = [g_loss.item()-p_loss.item()-px_loss.item(), px_loss.item(), p_loss.item(), d_loss.item()-gp_loss.item(), gp_loss.item(), psnr]
                record.update_status(status, mode=phase)
                record.print_buffer(mode=phase)
            record.print_average(mode=phase)

        scheduler.step(mse_loss)
        record.save_checkpoint(net, optimizer_g, save_criterion='psnr')
        record.write_log()
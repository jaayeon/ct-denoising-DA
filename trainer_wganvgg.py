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
        optimizer_g = optim.Adam(net.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
        optimizer_d = optim.Adam(net.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
        print("===> Use Adam optimizer_g")
    
    if opt.resume:
        #not possible
        opt.start_epoch, net, optimizer_g = load_model(opt, net, optimizer=optimizer_g)
        # _, net_D, optimizer_d = load_model(opt, net_D, optimizer_g=optimizer_d)
    else:
        set_checkpoint_dir(opt)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    log_file = os.path.join(opt.checkpoint_dir, opt.model + "_log.csv")
    opt_file = os.path.join(opt.checkpoint_dir, opt.model + "_opt.txt")
    
    if opt.multi_gpu:
        net = nn.DataParallel(net)
    
    scheduler = ReduceLROnPlateau(optimizer_g, factor=0.5, patience=5, mode='min')
    # scheduler = StepLR(optimizer_g, step_size=50, gamma=0.5)

    # Create log file when training start
    if opt.start_epoch == 1:
        with open(log_file, mode='w') as f:
            f.write("epoch, gloss_t, ploss_t, dloss_t, gploss_t, psnr_t, gloss_v, ploss_v, dloss_v, gploss_v, psnr_v\n")
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
            total_losses = np.array([0.0,0.0,0.0,0.0])
            total_psnr = 0.0

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
                    g_loss, p_loss, _ = net.g_loss(x,target,perceptual=True, return_p=True)
                    g_loss.backward()
                    optimizer_g.step()

                    raise KeyboardInterrupt
                elif phase == 'valid':
                    d_loss, gp_loss = net.d_loss(x,target,gp=True,return_gp=True)
                    g_loss, p_loss, _ = net.g_loss(x,target,perceptual=True,return_p=True)

                out = net.fake
                #generator loss, perceptual loss, discriminator loss, gradient penalty loss
                total_losses += [g_loss.item()-p_loss.item()*0.1, p_loss.item(), d_loss.item()-gp_loss.item(), gp_loss.item()]

                # print("max(out):", torch.max(out))
                # print("min(out):", torch.min(out))
                mse_loss = mse_criterion(out, target)
                psnr = 10 * math.log10(1 / mse_loss.item())
                total_psnr += psnr

                print("%s %.2fs => Epoch[%d/%d](%d/%d): gLoss: %.10f pLoss: %.10f dLoss: %.10f gpLoss: %.10f PSNR: %.5f" %
                    (mode, time.time() - start_time, opt.epoch_num, opt.n_epochs, iteration, len(dataloader[phase]), g_loss.item()-p_loss.item()*0.1, p_loss.item(), d_loss.item()-gp_loss.item(), gp_loss.item(), psnr))

            epoch_avg_loss = total_losses / iteration
            epoch_avg_psnr = total_psnr / iteration

            if phase == 'train':
                gloss_t = epoch_avg_loss[0]
                ploss_t = epoch_avg_loss[1]
                dloss_t = epoch_avg_loss[2]
                gploss_t = epoch_avg_loss[3]
                train_psnr = epoch_avg_psnr
            else:
                gloss_v = epoch_avg_loss[0]
                ploss_v = epoch_avg_loss[1]
                dloss_v = epoch_avg_loss[2]
                gploss_v = epoch_avg_loss[3]
                valid_psnr = epoch_avg_psnr
                scheduler.step(mse_loss)
                print("Valid LOSS avg : gLoss: {:5f} pLoss: {:5f} dLoss: {:5f} gpLoss: {:5f}\nValid PSNR avg : {:5f}".format(gloss_v, ploss_v, dloss_v, gploss_v, valid_psnr))

        with open(log_file, mode='a') as f:
            f.write("%d,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f\n" % (
                epoch,
                gloss_t,
                ploss_t,
                dloss_t,
                gploss_t,
                train_psnr,
                gloss_v,
                ploss_v,
                dloss_v,
                gploss_v,
                valid_psnr
            ))

        if current_best_psnr < valid_psnr : 
            save_checkpoint(opt, net, optimizer_g, epoch, valid_psnr)
            current_best_psnr = valid_psnr


    
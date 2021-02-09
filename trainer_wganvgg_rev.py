import os, time
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import DataLoader

from models import set_model
from utils.saver import Record
from utils.loader import load_model
from utils.helper import set_gpu, set_checkpoint_dir

def run_train(opt, src_t_loader, src_v_loader, trg_t_loader, trg_v_loader):
    # check gpu setting with opt arguments
    opt = set_gpu(opt)
    print('Initialize networks for training')
    net = set_model(opt)
    print(net)

    print("Setting Optimizer")
    if opt.optimizer == 'adam':
        optimizer_g = optim.Adam(net.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=opt.weight_decay)
        optimizer_d = optim.Adam(net.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=opt.weight_decay)
        optimizer_dc = optim.Adam(net.domain_discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=opt.weight_decay_dc)
        optimizer_rev = optim.Adam(net.generator.style_params(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=opt.weight_decay)
        print("===> Use Adam optimizer")
    elif opt.optimizer == 'rms':
        optimizer_g = optim.RMSprop(net.generator.parameters(), lr=opt.lr, eps=1e-8, weight_decay=opt.weight_decay, centered=False)
        optimizer_d = optim.RMSprop(net.discriminator.parameters(), lr=opt.lr, eps=1e-8, weight_decay=opt.weight_decay, centered=False)
        optimizer_dc = optim.RMSprop(net.domain_discriminator.parameters(), lr=opt.lr, eps=1e-8, weight_decay=opt.weight_decay_dc, centered=False)
        optimizer_rev = optim.RMSprop(net.generator.style_params(), lr=opt.lr, eps=1e-8, weight_decay=opt.weight_decay_dc, centered=False)
        print("===> Use RMSprop optimizer")
    
    if opt.resume:
        #not possible
        opt.start_epoch, net, optimizer_g = load_model(opt, net, optimizer=optimizer_g)
        raise NotImplementedError("can't load optimizer, you have to save all different optimizers")
    else:
        set_checkpoint_dir(opt)
    
    if opt.multi_gpu:
        net.generator = nn.DataParallel(net.generator)
        net.discriminator = nn.DataParallel(net.discriminator)
        net.domain_discriminator = nn.DataParallel(net.domain_discriminator)
        net.feature_extractor = nn.DataParallel(net.feature_extractor)
    
    scheduler_g = ReduceLROnPlateau(optimizer_g, factor=0.5, patience=5, mode='min')
    scheduler_d = ReduceLROnPlateau(optimizer_d, factor=0.5, patience=5, mode='min')
    scheduler_dc = ReduceLROnPlateau(optimizer_dc, factor=0.5, patience=5, mode='min')
    scheduler_rev = ReduceLROnPlateau(optimizer_rev, factor=0.5, patience=5, mode='min')
    # scheduler = StepLR(optimizer_g, step_size=50, gamma=0.5)

    # Create log file when training start
    if opt.start_epoch == 1:
        keys=['gloss', 'advloss', 'lloss', 'ploss', 'revloss', 'dcloss', 'dloss', 'src_psnr', 'nsrc_psnr', 'trg_psnr', 'ntrg_psnr']
        record = Record(opt, train_length=len(src_t_loader), valid_length=len(src_v_loader), keys=keys)

    mse_criterion = nn.MSELoss()
    if opt.use_cuda:
        mse_criterion = mse_criterion.to(opt.device)

    print('train_dir : {}\ntest_dir : {}\nimg_dir : {}\ngt_img_dir : {}'.format(opt.train_dir, opt.test_dir, opt.img_dir, opt.gt_img_dir))

    for epoch in range(opt.start_epoch, opt.n_epochs):
        opt.epoch_num = epoch

        for param_group in optimizer_g.param_groups:
            print('optim_g lr : ', param_group['lr'])
        for param_group in optimizer_d.param_groups:
            print("optim_d lr : ", param_group['lr'])
        for param_group in optimizer_dc.param_groups:
            print("optim_dc lr : ", param_group['lr'])
        for param_group in optimizer_rev.param_groups:
            print("optim_rev lr : ", param_group['lr'])

        net.generator.train()
        net.discriminator.train()
        net.domain_discriminator.train()

        print("*** Training ***")
        for batch in zip(src_t_loader, trg_t_loader):
            src_img, src_lbl = batch[0][0], batch[0][1]
            trg_img, trg_lbl = batch[1][0], batch[1][1]

            if opt.use_cuda:
                src_img, src_lbl = src_img.to(opt.device), src_lbl.to(opt.device)
                trg_img, trg_lbl = trg_img.to(opt.device), trg_lbl.to(opt.device)

            #discriminator & domain classifier
            optimizer_d.zero_grad()
            optimizer_dc.zero_grad()
            net.discriminator.zero_grad()
            net.domain_discriminator.zero_grad()
            for _ in range(opt.n_d_train):
                d_loss, gp_loss = net.d_loss(src_img, src_lbl, gp=True, return_losses=True)
                d_loss.backward()
                optimizer_d.step()

                dc_loss, dcgp_loss = net.dc_loss(src_img, src_lbl, trg_img, gp=True, return_losses=True)
                dc_loss.backward()
                optimizer_dc.step()

            #generator & reversal 
            optimizer_g.zero_grad()
            optimizer_rev.zero_grad()
            net.generator.zero_grad()
            g_loss, adv_loss, l_loss, p_loss= net.g_loss(src_img, src_lbl, perceptual=True, pixel_wise=True, return_losses=True)
            rev_loss = net.rev_loss(src_img, src_lbl)
            g_loss.backward()
            rev_loss.backward()
            optimizer_g.step()
            optimizer_rev.step()

            #calculate psnr
            src_out = net.src_out
            trg_out = net.trg_out
            mse_loss = mse_criterion(src_out, src_lbl)
            spsnr = 10 * math.log10(1 / mse_loss.item())
            nmse_loss = mse_criterion(src_img, src_lbl)
            nspsnr = 10 * math.log10(1 / nmse_loss.item())
            mse_loss = mse_criterion(trg_out, trg_lbl)
            tpsnr = 10 * math.log10(1 / mse_loss.item())
            nmse_loss = mse_criterion(trg_img, trg_lbl)
            ntpsnr = 10 * math.log10(1 / nmse_loss.item())

            #update status
            status = [g_loss.item(), adv_loss.item(), l_loss.item(), p_loss.item(), rev_loss.item(), dc_loss.item(), d_loss.item(), spsnr, nspsnr, tpsnr, ntpsnr]
            record.update_status(status, mode='train')
            record.print_buffer(mode='train')

        record.print_average(mode='train')

        net.generator.eval()
        net.discriminator.eval()
        net.domain_discriminator.eval()
        print("***Validation***")
        for batch in zip(src_v_loader, trg_v_loader):
            src_img, src_lbl = batch[0][0], batch[0][1]
            trg_img, trg_lbl = batch[1][0], batch[1][1]

            if opt.use_cuda:
                src_img, src_lbl = src_img.to(opt.device), src_lbl.to(opt.device)
                trg_img, trg_lbl = trg_img.to(opt.device), trg_lbl.to(opt.device)

            d_loss, gp_loss = net.d_loss(src_img,src_lbl,gp=True,return_losses=True)
            dc_loss, dcgp_loss = net.dc_loss(src_img, src_lbl, trg_img, gp=True, return_losses=True)
            g_loss, adv_loss, l_loss, p_loss = net.g_loss(src_img,src_lbl,perceptual=True, pixel_wise=True, return_losses=True)
            rev_loss = net.rev_loss(src_img, src_lbl)

            #calculate psnr
            src_out = net.src_out
            trg_out = net.trg_out
            mse_loss = mse_criterion(src_out, src_lbl)
            spsnr = 10 * math.log10(1 / mse_loss.item())
            nmse_loss = mse_criterion(src_img, src_lbl)
            nspsnr = 10 * math.log10(1 / nmse_loss.item())
            mse_loss = mse_criterion(trg_out, trg_lbl)
            tpsnr = 10 * math.log10(1 / mse_loss.item())
            nmse_loss = mse_criterion(trg_img, trg_lbl)
            ntpsnr = 10 * math.log10(1 / nmse_loss.item())

            #update status
            status = [g_loss.item(), adv_loss.item(), l_loss.item(), p_loss.item(), rev_loss.item(), dc_loss.item(), d_loss.item(), spsnr, nspsnr, tpsnr, ntpsnr]
            record.update_status(status, mode='valid')
            record.print_buffer(mode='valid')

        scheduler_g.step(mse_loss)
        scheduler_d.step(mse_loss)
        scheduler_dc.step(mse_loss)
        scheduler_rev.step(mse_loss)

        record.print_average(mode='valid')
        record.save_checkpoint(net, optimizer_g, save_criterion='trg_psnr')
        record.write_log()

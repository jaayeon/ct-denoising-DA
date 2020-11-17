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

def run_train(opt, src_t_loader, src_v_loader, trg_t_loader, trg_v_loader):

    # check gpu setting with opt arguments
    opt = set_gpu(opt)
    
    print('Initialize networks for training')
    net = set_model(opt)
    print(net)

    # if opt.use_cuda:
    #     net = net.to(opt.device)
    
    print("Setting Optimizer")
    if opt.optimizer == 'adam':
        optimizer_g = optim.Adam(net.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0.001)
        optimizer_d = optim.Adam(net.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0.001)
        optimizer_dmd = optim.Adam(net.domain_discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=opt.weight_decay)
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
        net.generator = nn.DataParallel(net.generator)
        net.discriminator = nn.DataParallel(net.discriminator)
        net.domain_discriminator = nn.DataParallel(net.domain_discriminator)
        net.feature_extractor = nn.DataParallel(net.feature_extractor)
    
    scheduler = ReduceLROnPlateau(optimizer_g, factor=0.5, patience=5, mode='min')
    # scheduler = StepLR(optimizer_g, step_size=50, gamma=0.5)

    # Create log file when training start
    if opt.start_epoch == 1:
        with open(log_file, mode='w') as f:
            f.write("epoch, gloss_t, pxloss_t, ploss_t, fgloss_t, dloss_t, gploss_t, advloss_t, dmgploss_t, psnr_t, gloss_v, pxloss_v, ploss_v, fgloss_v, dloss_v, gploss_v, advloss_v, dmgploss_v, psnr_v\n")
        save_config(opt)

    mse_criterion = nn.MSELoss()
    
    if opt.use_cuda:
        mse_criterion = mse_criterion.to(opt.device)

    print('train_dir : {}\ntest_dir : {}\nimg_dir : {}\ngt_img_dir : {}'.format(opt.train_dir, opt.test_dir, opt.img_dir, opt.gt_img_dir))

    current_best_psnr = 0.0
    for epoch in range(opt.start_epoch, opt.n_epochs):
        opt.epoch_num = epoch

        for param_group in optimizer_g.param_groups:
            print('optim_g lr : ', param_group['lr'])
        for param_group in optimizer_d.param_groups:
            print("optim_d lr : ", param_group['lr'])
        for param_group in optimizer_dmd.param_groups:
            print("optim_dmd lr : ", param_group['lr'])

        train_losses = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        valid_losses = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        train_psnr = 0.0
        valid_spsnr = 0.0
        valid_tpsnr = 0.0

        net.generator.train()
        net.discriminator.train()
        net.domain_discriminator.train()

        print("*** Training ***")
        start_time = time.time()
        for iteration_t, batch in enumerate(zip(src_t_loader, trg_t_loader), 1):
            src_img, src_lbl = batch[0][0], batch[0][1]
            trg_img, trg_lbl = batch[1][0], batch[1][1]

            if opt.use_cuda:
                src_img, src_lbl = src_img.to(opt.device), src_lbl.to(opt.device)
                trg_img, trg_lbl = trg_img.to(opt.device), trg_lbl.to(opt.device)

            #discriminator
            optimizer_d.zero_grad()
            optimizer_dmd.zero_grad()
            net.discriminator.zero_grad()
            net.domain_discriminator.zero_grad()
            for _ in range(opt.n_d_train):
                d_loss, gp_loss = net.d_loss(src_img, src_lbl, gp=True, return_gp=True)
                d_loss.backward()
                optimizer_d.step()

                adv_loss, dmgp_loss = net.adv_loss(src_img, trg_img, gp=True, return_gp=True)
                adv_loss.backward()
                optimizer_dmd.step()
            #generator, perceptual loss
            optimizer_g.zero_grad()
            net.generator.zero_grad()
            g_loss, px_loss, p_loss, fg_loss = net.g_loss(src_img, src_lbl, perceptual=True, return_p=True, pixel_wise=True, adv=True)
            g_loss.backward()
            optimizer_g.step()

            out = net.fake
            #gloss, ploss, fgloss, dloss, gploss, advloss, domain_gploss
            train_loss = [g_loss.item()-px_loss.item()-p_loss.item()-fg_loss.item(), px_loss.item(), p_loss.item(), fg_loss.item(), d_loss.item()-gp_loss.item(), gp_loss.item(), adv_loss.item()-dmgp_loss.item(), dmgp_loss.item()]
            train_losses += train_loss

            # print("max(out):", torch.max(out))
            # print("min(out):", torch.min(out))
            mse_loss = mse_criterion(out, src_lbl)
            psnr = 10 * math.log10(1 / mse_loss.item())
            train_psnr += psnr

            print("%s %.2fs => Epoch[%d/%d](%d/%d): gLoss: %.7f pxLoss: %.7f pLoss: %.7f fgLoss: %.7f dLoss: %.7f gpLoss: %.7f advLoss: %.7f domain_gpLoss: %.7f PSNR: %.5f" %
                ('Training', time.time() - start_time, opt.epoch_num, opt.n_epochs, iteration_t, len(src_t_loader), train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5], train_loss[6], train_loss[7], psnr))


        net.generator.eval()
        net.discriminator.eval()
        net.domain_discriminator.eval()
        print("***Validation***")
        for iteration_v, batch in enumerate(zip(src_v_loader, trg_v_loader), 1):
            src_img, src_lbl = batch[0][0], batch[0][1]
            trg_img, trg_lbl = batch[1][0], batch[1][1]

            if opt.use_cuda:
                src_img, src_lbl = src_img.to(opt.device), src_lbl.to(opt.device)
                trg_img, trg_lbl = trg_img.to(opt.device), trg_lbl.to(opt.device)

            d_loss, gp_loss = net.d_loss(src_img,src_lbl,gp=True,return_gp=True)
            adv_loss, dmgp_loss = net.adv_loss(src_img, trg_img, gp=True, return_gp=True)
            g_loss, px_loss, p_loss, fg_loss = net.g_loss(src_img,src_lbl,perceptual=True,return_p=True,pixel_wise=True)

            src_out = net.fake
            _ = net.g_loss(trg_img, trg_lbl, perceptual=False)
            trg_out = net.fake
            
            #gloss, ploss, fgloss, dloss, gploss, advloss, domain_gploss
            valid_loss = [g_loss.item()-px_loss.item()-p_loss.item()-fg_loss.item(), px_loss.item(), p_loss.item(), fg_loss.item(), d_loss.item()-gp_loss.item(), gp_loss.item(), adv_loss.item()-dmgp_loss.item(), dmgp_loss.item()]
            valid_losses += valid_loss

            # print("max(out):", torch.max(out))
            # print("min(out):", torch.min(out))
            smse_loss = mse_criterion(src_out, src_lbl)
            nsmse_loss = mse_criterion(src_out, src_img)
            tmse_loss = mse_criterion(trg_out, trg_lbl)
            ntmse_loss = mse_criterion(trg_out, trg_img)
            spsnr = 10 * math.log10(1 / smse_loss.item())
            nspsnr = 10 * math.log10(1 / nsmse_loss.item())
            tpsnr = 10 * math.log10(1 / tmse_loss.item())
            ntpsnr = 10 * math.log10(1 / ntmse_loss.item())
            valid_spsnr += spsnr
            valid_tpsnr += tpsnr

            print("%s %.2fs => Epoch[%d/%d](%d/%d): gLoss: %.7f pxLoss: %.7f pLoss: %.7f fgLoss: %.7f dLoss: %.7f gpLoss: %.7f advLoss: %.7f domain_gpLoss: %.7f noise_srcPSNR: %.5f srcPSNR: %.5f noise_trgPSNR: %.5f trgPSNR: %.5f" %('Validation', 
            time.time() - start_time, opt.epoch_num, opt.n_epochs, iteration_t, len(src_t_loader), valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5], valid_loss[6], valid_loss[7], nspsnr, spsnr, ntpsnr, tpsnr))


        epoch_avg_train_loss = train_losses / iteration_t
        epoch_avg_valid_loss = valid_losses / iteration_v
        epoch_avg_train_psnr = train_psnr / iteration_t
        epoch_avg_valid_spsnr = valid_spsnr / iteration_v
        epoch_avg_valid_tpsnr = valid_tpsnr / iteration_v

        scheduler.step(smse_loss)
        print("Valid LOSS avg : gLoss: %.7f pLoss: %.7f pxLoss: %.7f fgLoss: %.7f dLoss: %.7f gpLoss: %.7f advLoss: %.7f domain_gpLoss: %.7f srcPSNR: %.5f trgPSNR: %.5f"%(epoch_avg_valid_loss[0], 
                epoch_avg_valid_loss[1], epoch_avg_valid_loss[2], epoch_avg_valid_loss[3], epoch_avg_valid_loss[4], epoch_avg_valid_loss[5], epoch_avg_valid_loss[6], epoch_avg_valid_loss[7], epoch_avg_valid_spsnr, epoch_avg_valid_tpsnr))

        with open(log_file, mode='a') as f:
            f.write("%d,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f,%08f\n"%(
                epoch,
                epoch_avg_train_loss[0],
                epoch_avg_train_loss[1],
                epoch_avg_train_loss[2],
                epoch_avg_train_loss[3],
                epoch_avg_train_loss[4],
                epoch_avg_train_loss[5],
                epoch_avg_train_loss[6],
                epoch_avg_train_loss[7],
                epoch_avg_train_psnr,
                epoch_avg_valid_loss[0],
                epoch_avg_valid_loss[1],
                epoch_avg_valid_loss[2],
                epoch_avg_valid_loss[3],
                epoch_avg_valid_loss[4],
                epoch_avg_valid_loss[5],
                epoch_avg_valid_loss[6],
                epoch_avg_valid_loss[7],
                epoch_avg_valid_tpsnr
            ))

        if current_best_psnr < epoch_avg_valid_tpsnr : 
            save_checkpoint(opt, net, optimizer_g, epoch, epoch_avg_valid_tpsnr)
            current_best_psnr = epoch_avg_valid_tpsnr


    
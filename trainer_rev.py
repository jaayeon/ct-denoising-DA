import os, time, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from utils.helper import set_checkpoint_dir, set_gpu
from utils.loader import load_model
from utils.saver import Record
from models import set_model

def run_train(opt, src_t_loader, src_v_loader, trg_t_loader, trg_v_loader):
    opt= set_gpu(opt)
    print('Initialize networks for training')
    net = set_model(opt)
    print(net)
    
    print("Setting Optimizer")
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(net.denoiser.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=opt.weight_decay)
        optimizer_dc = optim.Adam(net.domain_discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=opt.weight_decay_dc)
        print("===> Use Adam optimizer")
    elif opt.optimizer == 'rms':
        optimizer = optim.RMSprop(net.denoiser.parameters(), lr=opt.lr, eps=1e-8, weight_decay=opt.weight_decay)
        optimizer_dc = optim.RMSprop(net.domain_discriminator.parameters(), lr=opt.lr, eps=1e-8, weight_decay=opt.weight_decay_dc)
        print("===> Use RMSprop optimizer")

    if opt.resume:
        print("Choose Model checkpoint")
        opt.start_epoch, net, optimizers = load_model(opt, net, optimizer=[optimizer, optimizer_dc])
        optimizer, optimizer_dc = optimizers[0], optimizers[1]
    elif opt.pretrained : 
        net, _ = load_model(opt, net)
        set_checkpoint_dir(opt)
    else : 
        set_checkpoint_dir(opt)

    if opt.multi_gpu:
        net.denoiser = nn.DataParallel(net.denoiser)
        net.domain_discriminator = nn.DataParallel(net.domain_discriminator)

    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=20, mode='min')
    scheduler_dc = ReduceLROnPlateau(optimizer_dc, factor=0.5, patience=20, mode='min')

    # if opt.start_epoch == 1:
    keys = ['loss', 'lloss', 'ploss', 'revloss', 'dcloss', 'src_psnr', 'nsrc_psnr', 'trg_psnr', 'ntrg_psnr']
    record = Record(opt, train_length=len(src_t_loader), valid_length=len(src_v_loader), keys=keys)
    if opt.pretrained:
        save_criterion = 'trg_psnr'
    else : 
        save_criterion = 'src_psnr'

    mse_criterion = nn.MSELoss()
    if opt.use_cuda:
        mse_criterion = mse_criterion.to(opt.device)

    print('train_dir : {}\ntest_dir : {}\nimg_dir : {}\ngt_img_dir : {}'.format(opt.train_dir, opt.test_dir, opt.img_dir, opt.gt_img_dir))

    for epoch in range(opt.start_epoch, opt.n_epochs):
        opt.epoch_num = epoch

        for param_group in optimizer.param_groups:
            print('optim lr : ', param_group['lr'])
        for param_group in optimizer_dc.param_groups:
            print("optim D lr : ", param_group['lr'])

        net.denoiser.train()
        net.domain_discriminator.train()

        print("***Training***")
        for batch in zip(src_t_loader, trg_t_loader):
            src_img, src_lbl = batch[0][0], batch[0][1]
            trg_img, trg_lbl = batch[1][0], batch[1][1]
            trg_noise = batch[1][2] if opt.noise else None

            if opt.use_cuda:
                src_img, src_lbl = src_img.to(opt.device), src_lbl.to(opt.device)
                trg_img, trg_lbl = trg_img.to(opt.device), trg_lbl.to(opt.device)
                trg_noise = trg_noise.to(opt.device) if opt.noise else None

            optimizer.zero_grad()
            net.denoiser.zero_grad()
            if opt.pretrained: 
                #2nd step
                dc_loss = net.dc_loss(src_img, src_lbl, trg_img, ntrg=trg_noise)
                loss, l_loss, p_loss, rev_loss = net.g_loss(src_img, trg_img, src_lbl, perceptual=True, trg_noise=trg_noise, rev=opt.rev, saliency=opt.saliency, return_losses=True)
            else : 
                #1st step 
                for _ in range(opt.n_d_train):
                    optimizer_dc.zero_grad()
                    net.domain_discriminator.zero_grad()
                    dc_loss = net.dc_loss(src_img, src_lbl, trg_img, ntrg=trg_noise)
                    dc_loss.backward()
                    optimizer_dc.step()
                loss, l_loss, p_loss, rev_loss = net.g_loss(src_img, trg_img, src_lbl, perceptual=True, trg_noise=None, rev=opt.rev, saliency=opt.saliency, return_losses=True)
            loss.backward()
            optimizer.step()

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
            status = [loss.item(), l_loss.item(), p_loss.item(), rev_loss.item(), dc_loss.item(), spsnr, nspsnr, tpsnr, ntpsnr]
            record.update_status(status, mode='train')
            record.print_buffer(mode='train')
        record.print_average(mode='train')   

        net.denoiser.eval()
        net.domain_discriminator.eval()
        print("***Validation***")
        for batch in zip(src_v_loader, trg_v_loader):
            src_img, src_lbl = batch[0][0], batch[0][1]
            trg_img, trg_lbl = batch[1][0], batch[1][1]
            trg_noise = batch[1][2] if opt.noise else None

            if opt.use_cuda:
                src_img, src_lbl = src_img.to(opt.device), src_lbl.to(opt.device)
                trg_img, trg_lbl = trg_img.to(opt.device), trg_lbl.to(opt.device)
                trg_noise = trg_noise.to(opt.device) if opt.noise else None

            with torch.no_grad():
                if opt.pretrained:
                    dc_loss = net.dc_loss(src_img, src_lbl, trg_img, ntrg=trg_noise)
                    loss, l_loss, p_loss, rev_loss = net.g_loss(src_img, trg_img, src_lbl, perceptual=True, trg_noise=trg_noise, saliency=opt.saliency, return_losses=True)
                else: #1st step
                    dc_loss = net.dc_loss(src_img, src_lbl, trg_img, ntrg=trg_noise)
                    loss, l_loss, p_loss, rev_loss = net.g_loss(src_img, trg_img, src_lbl, perceptual=True, trg_noise=None, saliency=opt.saliency, return_losses=True)

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
            status = [loss.item(), l_loss.item(), p_loss.item(), rev_loss.item(), dc_loss.item(), spsnr, nspsnr, tpsnr, ntpsnr]
            record.update_status(status, mode='valid')
            record.print_buffer(mode='valid')
        
        scheduler.step(mse_loss)
        scheduler_dc.step(mse_loss)

        record.print_average(mode='valid')
        record.save_checkpoint(net, [optimizer, optimizer_dc], save_criterion = save_criterion)
        record.write_log()




import os, time, math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from utils.helper import set_checkpoint_dir, set_gpu
from utils.saver import load_model, save_checkpoint, save_config
from models import set_model
from models.losses import ssim_loss
from models import autoencoder, make_noise


def run_train(opt, self_t_loader, self_v_loader):
    opt= set_gpu(opt)
    print('Initialize self-supervised networks for training')

    net = set_model(opt)
    print(net)
    
    print("Setting Optimizer")
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
        print("===> Use Adam optimizer")

    if opt.resume:
        print("Choose Model checkpoint")
        opt.start_epoch, net, optimizer = load_model(opt, net, optimizer=optimizer)
    else : 
        set_checkpoint_dir(opt)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    
    log_file = os.path.join(opt.checkpoint_dir, opt.model + "_log.csv")
    opt_file = os.path.join(opt.checkpoint_dir, opt.model + "_opt.txt")

    if opt.multi_gpu:
        net = nn.DataParallel(net)

    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')

    # Setting loss function
    loss = nn.MSELoss()
    
    if opt.start_epoch == 1:
        with open(log_file, mode='w') as f:
            f.write("epoch,train_loss,valid_loss, train_psnr_,valid_psnr\n")
        save_config(opt)

    self_loader = {
        'train': self_t_loader,
        'valid': self_v_loader
    }

    models = ['train', 'valid']

    mse_criterion = nn.MSELoss()

    if opt.use_cuda:
        mse_criterion = mse_criterion.to(opt.device)
        

    print('train_dir : {}\ntest_dir : {}\nimg_dir : {}\ngt_img_dir : {}'.format(opt.train_dir, opt.test_dir, opt.img_dir, opt.gt_img_dir))

    current_best_psnr = 0.0
    #loss = ['loss_src_M', 'loss_trg_m', 'loss_trg_D_fake', 'loss_src_D', 'loss_trg_D_real']
    for epoch in range(opt.start_epoch, opt.n_epochs):
        opt.epoch_num = epoch
        train_psnr = 0.0
        train_loss = 0.0

        start_train = time.time()
        print("***Training***")
        for iteration_t, img in enumerate(self_t_loader, 1):
            self_lbl, self_real = img
            
            if opt.noise == True:
                self_img = make_noise(self_img, noise_typ = opt.noise_typ)


            if opt.use_cuda:
                self_img, self_lbl = self_img.to(opt.device), self_lbl.to(opt.device)

            
            net.train()
            optimizer.zero_grad()
        
            self_out = net(self_img)
        
            train_loss = loss(self_out,self_img)
            train_loss.backward()
            optimizer.step()

            train_loss += train_loss
            mse_loss = mse_criterion(self_out, self_lbl)
            psnr = 10* math.log10(1 / mse_loss.item())
            train_psnr += psnr

            print("%s %.2fs => Epoch[%d/%d](%d/%d): \ntrain_loss : %.5f trian_psnr : %.5f "%(
                'Training', time.time()-start_train, epoch, opt.n_epochs, iteration_t, len(self_t_loader), train_loss, train_psnr))
            print("PSNR : %.5f avg_PSNR : %.5f avg_LOSS : %.5f "%(psnr, train_psnr/iteration_t, train_loss/iteration_t))
        

        valid_psnr = 0.0
        valid_loss = 0.0
        valid_loss_D = 0.0
        start_valid = time.time()
        print("***Validation***")

        for iteration_v, img in enumerate(self_v_loader, 1):
            self_lbl, self_real = img
            
            if opt.noise == True:
                self_img = make_noise(self_img, noise_typ = opt.noise_typ)

            
            if opt.use_cuda:
                self_img, self_lbl = self_img.to(opt.device), self_lbl.to(opt.device)

            with torch.no_grad():
                net.eval()

                self_out = net(self_img)
                self_val_loss = loss(self_out,self_img)


                valid_loss += self_val_loss
    
                mse_loss = mse_criterion(self_out, self_lbl)
                psnr = 10 * math.log10(1 / mse_loss.item())
                valid_psnr += psnr
            print("%s %.2fs => Epoch[%d/%d](%d/%d): \n valid_loss: %.5f"%(
                'Validation', time.time()-start_valid, epoch, opt.n_epochs, iteration_v, len(self_v_loader), valid_loss))
            print("piglet PSNR : %.5f piglet avg_PSNR : %.5f avg_LOSS : %.5f avg_LOSS_D : %.5f"%(psnr, valid_psnr/iteration_v, valid_loss/iteration_v, valid_loss_D/iteration_v))
        

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


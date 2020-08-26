import os, time, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from utils.helper import set_checkpoint_dir, set_gpu
from utils.saver import load_model, save_checkpoint, save_config
from models import set_model, set_model_D
from models.losses import wgan_loss, ssim_loss


def run_train(opt, src_t_loader, src_v_loader, trg_t_loader, trg_v_loader):
    opt= set_gpu(opt)
    print('Initialize networks for training')

    net = set_model(opt)
    net_D = set_model_D(opt)
    print(net)
    print(net_D)
    
    print("Setting Optimizer")
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
        optimizer_D = optim.Adam(net_D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
        print("===> Use Adam optimizer")

    if opt.resume:
        print("Choose Model checkpoint")
        opt.start_epoch, net, optimizer = load_model(opt, net, optimizer=optimizer)
        print("Choose Discriminator checkpoint")
        _, net_D, optimizer_D = load_model(opt, net_D, optimizer=optimizer_D)
    else : 
        set_checkpoint_dir(opt)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    if not os.path.exists(opt.checkpoint_dir_D):
        os.makedirs(opt.checkpoint_dir_D)

    log_file = os.path.join(opt.checkpoint_dir, opt.model + "_log.csv")
    opt_file = os.path.join(opt.checkpoint_dir, opt.model + "_opt.txt")

    if opt.multi_gpu:
        net = nn.DataParallel(net)
        net_D = nn.DataParallel(net_D)

    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')

    # Setting loss function
    if opt.loss == 'l1':
        loss_criterion = nn.L1Loss()
    elif opt.loss == 'l2':
        loss_criterion = nn.MSELoss()
    else:
        raise ValueError("Please specify correct loss function")

    if opt.start_epoch == 1:
        with open(log_file, mode='w') as f:
            f.write("epoch,train_loss,valid_loss,train_loss_D,valid_loss_D,train_psnr,valid_psnr\n")
        save_config(opt)

    srcloader = {
        'train': src_t_loader,
        'valid': src_v_loader
    }
    trgloader = {
        'train': trg_t_loader,
        'valid': trg_v_loader
    }
    modes = ['train', 'valid']

    mse_criterion = nn.MSELoss()

    if opt.use_cuda:
        mse_criterion = mse_criterion.to(opt.device)

    print('train_dir : {}\ntest_dir : {}\nimg_dir : {}\ngt_img_dir : {}'.format(opt.train_dir, opt.test_dir, opt.img_dir, opt.gt_img_dir))

    current_best_psnr = 0.0
    loss = ['loss_src_M', 'loss_trg_m', 'loss_trg_D_fake', 'loss_src_D', 'loss_trg_D_real']
    for epoch in range(opt.start_epoch, opt.n_epochs):
        opt.epoch_num = epoch
        train_psnr = 0.0
        train_loss = 0.0
        train_loss_D = 0.0
        train_loss_s = 0.0
        train_loss_t = 0.0
        ssimloss = 0.0
        start_train = time.time()
        print("***Training***")
        for iteration_t, batch in enumerate(zip(src_t_loader, trg_t_loader), 1):
            src_img, src_lbl = batch[0][0], batch[0][1]
            trg_img, trg_lbl = batch[1][0], batch[1][1]

            if opt.use_cuda:
                src_img, src_lbl = src_img.to(opt.device), src_lbl.to(opt.device)
                trg_img, trg_lbl = trg_img.to(opt.device), trg_lbl.to(opt.device)

            net.train()
            net_D.train()

            optimizer.zero_grad()
            optimizer_D.zero_grad()

            # M with source Denoising
            src_out = net(src_img, src_lbl)
            loss_src_M = net.loss

            if opt.ssim_loss == True:
                ssimloss = ssim_loss.SSIM(window_size = 11)
                loss_src_M += ssimloss(src_img, src_lbl) 
            
            # M with target Denoising
            trg_out = net(trg_img, trg_lbl)

            ''' for param in net_D.parameters():
                param.requires_grad=False '''
            
            net_D.eval()

            src_out, trg_out = src_out.detach(), trg_out.detach()
            # source domain images
            src_outD = net_D(src_out,0)
            # target domain images
            trg_outD = net_D(trg_out,1)

            if opt.wgan_loss == True:
                if iteration_t % 10 == 0:
                    loss_src_M += -torch.mean(src_outD)
                    print('====================================================')
                    print('Adding gan loss:',loss_src_M)            
            
            loss_src_M.backward()
            optimizer.step()
            
            # D with source, target classification
            ''' for param in net_D.parameters():
                param.requires_grad = True '''
            
            net_D.train()

            # Gradient penalty
            gradient_penalty = wgan_loss.compute_gradient_penalty(net_D, src_out.data, trg_out.data)
            # Adversarial loss
            loss_trg_D = torch.mean(src_outD) - torch.mean(trg_outD) + opt.lambda_gp * gradient_penalty
            loss_trg_D.backward() #update M with adversarial loss 
            optimizer_D.step()

            train_loss_D += loss_trg_D
            train_loss += loss_src_M
            mse_loss = mse_criterion(src_out, src_lbl)
            psnr = 10* math.log10(1 / mse_loss.item())
            train_psnr += psnr

            print("%s %.2fs => Epoch[%d/%d](%d/%d): \nloss_src_M : %.5f loss_trg_D : %.5f"%(
                'Training', time.time()-start_train, epoch, opt.n_epochs, iteration_t, len(src_t_loader), loss_src_M, loss_trg_D))
            print("PSNR : %.5f avg_PSNR : %.5f avg_LOSS : %.5f avg_LOSS_D : %.5f"%(psnr, train_psnr/iteration_t, train_loss/iteration_t, train_loss_D/iteration_t))
        

        valid_psnr = 0.0
        valid_loss = 0.0
        valid_loss_D = 0.0
        start_valid = time.time()
        print("***Validation***")
        for iteration_v, batch in enumerate(zip(src_v_loader, trg_v_loader), 1):
            src_img, src_lbl = batch[0][0], batch[0][1]
            trg_img, trg_lbl = batch[1][0], batch[1][1]

            if opt.use_cuda:
                src_img, src_lbl = src_img.to(opt.device), src_lbl.to(opt.device)
                trg_img, trg_lbl = trg_img.to(opt.device), trg_lbl.to(opt.device)

            with torch.no_grad():
                net.eval()
                net_D.eval()

                src_out = net(src_img, src_lbl)
                loss_src_M = net.loss

                trg_out = net(trg_img, trg_lbl)
                loss_trg_m = net.loss

                trg_outD = net_D(trg_out, 0)
                loss_trg_D_fake = net_D.loss  

                src_outD = net_D(src_out,0)
                loss_src_D = net_D.loss / 2

                trg_outD = net_D(trg_out, 1)
                loss_trg_D_real = net_D.loss / 2

                valid_loss += loss_src_M
                valid_loss_D += loss_src_D
                valid_loss_D += loss_trg_D_real
                mse_loss = mse_criterion(trg_out, trg_lbl)
                psnr = 10 * math.log10(1 / mse_loss.item())
                valid_psnr += psnr
            print("%s %.2fs => Epoch[%d/%d](%d/%d): \nloss_src_M : %.5f loss_trg_m : %.5f loss_trg_D_fake : %.5f loss_src_D : %.5f loss_trg_D_real : %.5f"%(
                'Validation', time.time()-start_valid, epoch, opt.n_epochs, iteration_v, len(src_v_loader), loss_src_M, loss_trg_m, loss_trg_D_fake, loss_src_D, loss_trg_D_real))
            print("piglet PSNR : %.5f piglet avg_PSNR : %.5f avg_LOSS : %.5f avg_LOSS_D : %.5f"%(psnr, valid_psnr/iteration_v, valid_loss/iteration_v, valid_loss_D/iteration_v))
        

        train_psnr = train_psnr/iteration_t
        train_loss = train_loss/iteration_t
        valid_psnr = valid_psnr/iteration_v
        valid_loss = valid_loss/iteration_v
        train_loss_D = train_loss_D/iteration_t
        valid_loss_D = valid_loss_D/iteration_v

        with open(log_file, mode='a') as f:
            f.write("%d,%08f,%08f,%08f,%08f,%08f,%08f"%(
                epoch,
                train_loss,
                valid_loss,
                train_loss_D,
                valid_loss_D,
                train_psnr,
                valid_psnr
            ))

        if current_best_psnr < valid_psnr : 
            save_checkpoint(opt, net, optimizer, epoch, valid_loss)
            save_checkpoint(opt, net_D, optimizer_D, epoch, valid_loss_D, 'D')
            current_best_psnr = valid_psnr






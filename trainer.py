import os, time, glob, math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from utils.helper import set_gpu, set_checkpoint_dir
from utils.loader import load_model
from utils.saver import Record
from models import set_model

def run_train(opt, training_dataloader, valid_dataloader):
    # check gpu setting with opt arguments
    opt = set_gpu(opt)
    print('Initialize networks for training')
    net = set_model(opt)
    print(net)

    print("Setting Optimizer")
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=opt.weight_decay)
        print("===> Use Adam optimizer")
    
    if opt.resume:
        opt.start_epoch, net, optimizers = load_model(opt, net, optimizer=[optimizer])
        optimizer = optimizers[0]
    else:
        set_checkpoint_dir(opt)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    if opt.multi_gpu:
        net.denoiser = nn.DataParallel(net.denoiser)
        net.feature_extractor = nn.DataParallel(net.feature_extractor)
    
    if opt.multi_gpu:
        net = nn.DataParallel(net)
    
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    # Create log file when training start
    keys = ['loss', 'lloss', 'ploss', 'psnr', 'npsnr']
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
                net.denoiser.train()
            else:
                net.denoiser.eval()
                
            mode = "Training" if phase == 'train' else "Validation"
            print("*** %s ***"%(mode))
            for batch in dataloader[phase]:
                img, lbl = batch[0], batch[1]
                
                if opt.use_cuda:
                    img = img.to(opt.device)
                    lbl = lbl.to(opt.device)

                optimizer.zero_grad()
                net.denoiser.zero_grad()
                loss, l_loss, p_loss = net.g_loss(img, lbl, perceptual=True, return_losses=True)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                        
                out = net.out
                mse_loss = mse_criterion(out, lbl)
                psnr = 10 * math.log10(1 / mse_loss.item())
                nmse_loss = mse_criterion(img, lbl)
                npsnr = 10 * math.log10(1 / nmse_loss.item())

                status = [loss.item(), l_loss.item(), p_loss.item(), psnr, npsnr]
                record.update_status(status, mode=phase)
                record.print_buffer(mode=phase)
            record.print_average(mode=phase)

        scheduler.step(mse_loss)
        record.save_checkpoint(net, [optimizer], save_criterion='psnr')
        record.write_log()


    
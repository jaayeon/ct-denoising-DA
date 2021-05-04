import time
import torch
import random
import numpy as np

from utils.tester import test_net_by_tensor_patches


def train_net(opt, model, dataloader, train=True):
    if train:
        phase = 'Traning'
        model.train()
    else:
        phase = 'Validation'
        model.eval()

    print('*** {} phase ***'.format(phase))
    avg_loss = 0.0
    avg_psnr = 0.0

    len_dataloader =  len(dataloader)
    start_time = time.time()
    
    for i, batch in enumerate(dataloader, 1):
        src, trg = batch
        
        epoch = opt.epoch
        p = float(i + epoch * len_dataloader) / opt.n_epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        input = {
            'src': src,
            'trg': trg,
            'alpha': alpha
        }

        model.set_input(input)
        if train:
            model.optimize_parameters()
            end_time = time.time()
            model.log_loss(opt, phase, end_time - start_time, i, len(dataloader))
            batch_loss, batch_psnr = model.get_batch_loss_psnr()
        else:
            model.set_input(input)
            model.test()

                # out and target should be detach()
            out = model.out.detach()
            target = trg[1].detach()

            # x = x.to(opt.device).detach()
            out = out.to(opt.device).detach()
            target = target.to(opt.device).detach()

            batch_loss, batch_psnr = calc_loss_psnr(out, target)
        
        avg_loss += batch_loss
        avg_psnr += batch_psnr

    avg_loss, avg_psnr = avg_loss / i, avg_psnr / i
    print("===> {} avg_loss: {:.8f}, avg_psnr: {:.8f}".format(phase, avg_loss, avg_psnr))
    return avg_loss, avg_psnr

def calc_loss_psnr(out, target):
    mse_criterion = torch.nn.MSELoss()
    mse_loss = mse_criterion(out, target)
    psnr = 10 * torch.log10(1 / mse_loss)
    return mse_loss, psnr

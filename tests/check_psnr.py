import torch
import numpy as np
import sys
from models import set_model
from utils.saver import select_checkpoint_dir, load_model
from options import args

from skimage.external.tifffile import imsave, imread

def calc_metrics(x, out, target):

    out[out>1.0] = 1.0
    out[out<0.0] = 0.0

    mse_criterion = torch.nn.MSELoss()
    
    noise_loss = mse_criterion(x, target)
    noise_psnr = 10 * torch.log10(1 / noise_loss)

    out_loss = mse_criterion(out, target)
    out_psnr = 10 * torch.log10(1 / out_loss)

    return noise_loss, noise_psnr, out_loss, out_psnr


if __name__ == "__main__":
    target = 'D:/data/denoising/test/lp-mayo/full/C267/simens_full_C267_000.tiff'
    x = 'D:/data/denoising/test/lp-mayo/low/C267/simens_low_C267_000.tiff'
    target = imread(target)
    x = imread(x)
    target = torch.from_numpy(target.reshape(1,1,target.shape[0], target.shape[1])).type(torch.FloatTensor)
    x = torch.from_numpy(x.reshape(1,1,x.shape[0], x.shape[1])).type(torch.FloatTensor)
    opt = args
    net = set_model(opt)
    _, net, _ = load_model(opt, net)

    out = net(x)
    print(x.shape)
    print(target.shape)

    nl, np, ol, op = calc_metrics(x, out, target)
    print('nl : {:.8f}, np : {:.8f}, ol : {:.8f}, op : {:.8f}'.format(nl, np, ol, op))
    pass
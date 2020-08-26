import torch
import numpy as np
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

def calc_metrics(x, out, target, data_range=1, window_size=11, channel=1, size_average=True):

    out[out>1.0] = 1.0
    out[out<0.0] = 0.0

    #compute psnr
    noise_loss, noise_psnr = compute_LOSS_PSNR(x, target)
    out_loss, out_psnr = compute_LOSS_PSNR(out, target)

    #compute ssim
    noise_ssim = compute_SSIM(x, target)
    out_ssim = compute_SSIM(out, target)

    return noise_loss, noise_psnr, noise_ssim, out_loss, out_psnr, out_ssim


def compute_LOSS_PSNR(img1, img2):
    mse_criterion = torch.nn.MSELoss()
    loss = mse_criterion(img1, img2)
    psnr = 10 * torch.log10(1 / loss)

    return loss, psnr
    

def compute_SSIM(img1, img2, data_range=1, window_size=11, channel=1, size_average=True):
    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    #C1, C2 = 0.01**2, 0.03**2

    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def forward_ensemble(input_img, net, device): #1, c, h, w
    out_img = np.zeors(input_img.shape, dtype=np.float32)
    for f in range(2):
        for r in range(4):
            aug_input = augment(input_img, flip=f, rot90=r)
            aug_output = net(aug_input)
            out_img += de_augment(aug_output, rot90=4-r, flip=f)
    
    out_img = out_img/8.0

    return out_img
            

def augment(img, flip=0, rot90=0):
    if flip : img = img[:, :, :, ::-1]
    if rot90 : img = np.rot90(img, rot90, (2,3))

    return img


def de_augment(img, rot90=0, flip=0):
    if rot90 : img = np.rot90(img, rot90, (2,3))
    if flip : img = img[:, :, :, ::-1]

    return img

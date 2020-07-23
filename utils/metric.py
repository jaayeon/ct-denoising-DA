import torch
import numpy as np

def calc_metrics(x, out, target):

    out[out>1.0] = 1.0
    out[out<0.0] = 0.0

    mse_criterion = torch.nn.MSELoss()
    
    noise_loss = mse_criterion(x, target)
    noise_psnr = 10 * torch.log10(1 / noise_loss)

    out_loss = mse_criterion(out, target)
    out_psnr = 10 * torch.log10(1 / out_loss)

    return noise_loss, noise_psnr, out_loss, out_psnr

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

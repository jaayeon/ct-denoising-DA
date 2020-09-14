import numpy as np
import os
import cv2
import torch

def make_noisy(img, noise_type = 'guass'):  #50,1,80,80
    if noise_type == 'guass':
        bt,ch,row,col= img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,bt))

        gauss = gauss.reshape(bt,row,col)
        gauss = torch.from_numpy(gauss)
        gauss = gauss.unsqueeze(1).float()
        #print(gauss.shape)
        noisy = img + gauss
        #print(noisy.shape)

        return noisy


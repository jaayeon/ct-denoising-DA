import os
import torch
import torch.nn as nn

def make_model(opt):
    return DnCNN(opt)


class DnCNN(nn.Module):
    def __init__(self, opt, num_of_layers = 17):
        super(DnCNN,self).__init__()
        kernel_size = 3
        padding = 1
        #default features of dncnn is 64
        features = 64
        # num_of_layers = opt.n_conv
        layers = []

        n_channels = opt.n_channels
        self.Loss = nn.L1Loss()

        layers.append(nn.Conv2d(in_channels=n_channels, out_channels = features, kernel_size = kernel_size, padding = padding, bias = False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        
        y = x
        out = self.dncnn(x)
        # self.loss = self.Loss(y-out, lbl)

        return y - out

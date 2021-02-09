import os
import torch
import torch.nn as nn

def make_model(opt):
    return DnCNN(opt)


class DnCNN(nn.Module):
    def __init__(self, opt, num_of_layers = 17):
        super(DnCNN,self).__init__()
        self.rev = False
        kernel_size = 3
        padding = 1
        features = 64
        before_layers = []
        after_layers = []
        n_channels = opt.n_channels

        before_layers.append(nn.Conv2d(in_channels=n_channels, out_channels = features, kernel_size = kernel_size, padding = padding, bias = False))
        before_layers.append(nn.ReLU(inplace=True))

        style_stage = int((num_of_layers-2)*opt.style_stage/6)
        for _ in range(style_stage):
            before_layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            before_layers.append(nn.BatchNorm2d(features))
            before_layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_of_layers-style_stage):
            after_layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            after_layers.append(nn.BatchNorm2d(features))
            after_layers.append(nn.ReLU(inplace=True))
        after_layers.append(nn.Conv2d(in_channels=features, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.body1 = nn.Sequential(*before_layers)
        self.body2 = nn.Sequential(*after_layers)

    def forward(self, x):
        
        global_res = x
        feature=self.body1(x)
        out = self.body2(feature)

        out = global_res-out
        if self.rev : 
            return out, feature
        else : 
            return out

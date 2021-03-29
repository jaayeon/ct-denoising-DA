from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import random

def set_model(opt):

    if opt.way == 'base':
        module_name = 'models.networks_base'
    elif opt.way == 'rev':
        module_name = 'models.networks_rev'
    elif opt.way == 'wgan':
        module_name = 'models.wganvgg_base'
    elif opt.way == 'wganrev':
        module_name = 'models.wganvgg_rev'

    module = import_module(module_name)
    model = module.make_model(opt)

    if opt.use_cuda:
        model = model.to(opt.device)

    return model

def get_base_model(opt):
    if opt.model == 'dncnn':
        module_name = 'models.dncnn'
    elif opt.model == 'unet':
        module_name = 'models.unet'
    elif opt.model == 'edsr':
        module_name = 'models.edsr'   
    else :     
        raise ValueError("Need to specify model {}".format(opt.model))
    
    module = import_module(module_name)
    model = module.make_model(opt)

    return model


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35]).eval()

    def forward(self, x):
        x = self.normalize(x)
        out = self.feature_extractor(x)
        return out

    def normalize(self, x):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        for i, (m,s) in enumerate(zip(mean, std)):
            x[:,i:i+1,:,:] = (x[:,i:i+1,:,:] - m)/s
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, input_channels, class_num=1):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        def conv_output_size(input_size, kernel_size_list, stride_list):
            n=input_size
            for k, s in zip(kernel_size_list, stride_list):
                # n = (n - k) // s + 1
                n = (n - k + 2*1) // s + 1
            return n

        def add_block(layers, ch_in, ch_out, stride):
            # layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 0))
            layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 1))
            layers.append(nn.LeakyReLU())
            return layers

        layers = []
        # ch_stride_set = [(input_channels,64,1),(64,64,2),(64,128,1),(128,128,2),(128,256,1),(256,256,2)]
        ch_stride_set = [(input_channels,64,1),(64,128,2),(128,256,1)]
        for ch_in, ch_out, stride in ch_stride_set:
            add_block(layers, ch_in, ch_out, stride)

        # self.output_size = conv_output_size(input_size, [3]*6, [1,2]*3)
        self.output_size = conv_output_size(input_size, [3]*3, [1,2,1])
        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(256*self.output_size*self.output_size, 1024)
        self.fc2 = nn.Linear(1024, class_num)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.random_crop(x)
        out = self.net(x)
        out = out.view(-1, 256*self.output_size*self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)
        return out

    def random_crop(self, x):
        size = x.size()[-1] #b,c,h,w
        if size == self.input_size:
            return x
        rh = random.randint(0, size-self.input_size)
        rw = random.randint(0, size-self.input_size)
        return x[:,:,rh:rh+self.input_size, rw:rw+self.input_size]
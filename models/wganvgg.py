import os
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import vgg19
from . import unet

def make_model(opt):
    return WGAN_VGG(opt)

#wganvgg original generator
""" 
class WGAN_VGG_generator(nn.Module):
    def __init__(self):
        super(WGAN_VGG_generator, self).__init__()
        layers = [nn.Conv2d(1,32,3,1,1), nn.ReLU()]
        for i in range(2, 8):
            layers.append(nn.Conv2d(32,32,3,1,1))
            layers.append(nn.ReLU())
        layers.extend([nn.Conv2d(32,1,3,1,1), nn.ReLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out
 """

class WGAN_VGG_generator(nn.Module):
    def __init__(self,opt):
        super(WGAN_VGG_generator, self).__init__()
        self.unet = unet.UNet(opt)

    def forward(self, x):
        out = self.unet(x)
        return out


class WGAN_VGG_discriminator(nn.Module):
    def __init__(self, input_size):
        super(WGAN_VGG_discriminator, self).__init__()
        def conv_output_size(input_size, kernel_size_list, stride_list):
            n = (input_size - kernel_size_list[0]) // stride_list[0] + 1
            for k, s in zip(kernel_size_list[1:], stride_list[1:]):
                n = (n - k) // s + 1
            return n

        def add_block(layers, ch_in, ch_out, stride):
            layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 0))
            layers.append(nn.LeakyReLU())
            return layers

        layers = []
        ch_stride_set = [(1,64,1),(64,64,2),(64,128,1),(128,128,2),(128,256,1),(256,256,2)]
        for ch_in, ch_out, stride in ch_stride_set:
            add_block(layers, ch_in, ch_out, stride)

        self.output_size = conv_output_size(input_size, [3]*6, [1,2]*3)
        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(256*self.output_size*self.output_size, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, 256*self.output_size*self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)
        return out


class WGAN_VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(WGAN_VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35]).eval()

    def forward(self, x):
        out = self.feature_extractor(x)
        return out


class WGAN_VGG(nn.Module):
    # referred from https://github.com/kuc2477/pytorch-wgan-gp
    def __init__(self, opt):
        input_size = opt.patch_size
        super(WGAN_VGG, self).__init__()
        self.generator = WGAN_VGG_generator(opt)
        self.discriminator = WGAN_VGG_discriminator(input_size)
        self.domain_discriminator = WGAN_VGG_discriminator(input_size)
        self.feature_extractor = WGAN_VGG_FeatureExtractor()
        self.p_criterion = nn.L1Loss()

    def d_loss(self, x, y, gp=True, return_gp=False):
        self.generator.eval()
        self.discriminator.train()

        fake = self.generator(x)
        d_real = self.discriminator(y)
        d_fake = self.discriminator(fake)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = torch.from_numpy(np.array(0.0))
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss
    
    def adv_loss(self, src, trg, gp=True, return_gp=False):
        self.generator.eval()
        self.domain_discriminator.train()

        src_out = self.generator(src)
        trg_out = self.generator(trg)
        # d_src = self.domain_discriminator(src_out.detach())
        # d_trg = self.domain_discriminator(trg_out.detach())
        d_src = self.domain_discriminator(src_out.detach()-src)
        d_trg = self.domain_discriminator(trg_out.detach()-trg)
        adv_loss = -torch.mean(d_trg) + torch.mean(d_src)
        if gp:
            gp_loss = self.gp(src_out.detach()-src, trg_out.detach()-trg)
            loss = adv_loss + gp_loss
        else : 
            gp_loss = torch.from_numpy(np.array(0.0))
            loss = adv_loss
        return (loss, gp_loss) if return_gp else loss
   

    def g_loss(self, x, y, perceptual=True, return_p=False, adv=False):
        self.generator.train()
        self.discriminator.eval()
        self.domain_discriminator.eval()

        self.fake = self.generator(x)
        d_fake = self.discriminator(self.fake) 
        g_loss = -torch.mean(d_fake) 
        if perceptual:
            p_loss = self.p_loss(x, y)
            loss = g_loss + (0.1 * p_loss)
        else:
            p_loss = torch.from_numpy(np.array(0.0))
            loss = g_loss
        if adv:
            fg_loss = -0.001 * torch.mean(self.domain_discriminator(self.fake))
            loss = loss + fg_loss
        else : 
            fg_loss = torch.from_numpy(np.array(0.0))
        return (loss, p_loss, fg_loss) if (return_p or adv) else loss

    def p_loss(self, x, y):
        fake = self.generator(x).repeat(1,3,1,1)
        real = y.repeat(1,3,1,1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10):
        y, fake = self.size_align(y, fake)
        assert y.size() == fake.size()
        a = torch.cuda.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.discriminator(interp)
        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty

    def size_align(self, x, y):
        if x.size(0) == y.size(0) : 
            pass
        elif x.size(0) > y.size(0):
            x = x[y.size(0), :, :, :]
        elif x.size(0) < y.size(0) : 
            y = y[x.size(0), :, :, :]
        return x,y
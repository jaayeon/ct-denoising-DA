import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from . import Discriminator, FeatureExtractor, get_base_model

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

class WGAN_VGG(nn.Module):
    # referred from https://github.com/kuc2477/pytorch-wgan-gp
    def __init__(self, opt):
        input_size = opt.patch_size
        super(WGAN_VGG, self).__init__()
        self.generator = get_base_model(opt)
        self.discriminator = Discriminator(input_size, opt.n_channels)
        self.feature_extractor = FeatureExtractor()

        self.p_criterion = nn.L1Loss() #perceptual loss
        self.l_criterion = nn.L1Loss() #l1 pixelwise loss

        self.vgg_weight = opt.vgg_weight #perceptual loss weight
        self.l_weight = opt.l_weight #l1 pixelwise loss weight

    def d_loss(self, x, y, gp=True, return_losses=False):
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
        return (loss, gp_loss) if return_losses else loss


    def g_loss(self, x, y, perceptual=True, pixel_wise=False, return_losses=True):
        self.generator.train()
        self.discriminator.eval()

        self.fake = self.generator(x)
        d_fake = self.discriminator(self.fake) 
        adv_loss = -torch.mean(d_fake) 
        if perceptual:
            p_loss = self.vgg_weight *self.p_loss(x, y)
            loss = adv_loss + p_loss
        else:
            p_loss = torch.from_numpy(np.array(0.0))
            loss = adv_loss
        if pixel_wise:
            px_loss =  self.l_weight * self.l_criterion(self.fake, y)
            loss = loss + px_loss
        else : 
            px_loss = torch.from_numpy(np.array(0.0))

        return (loss, adv_loss, px_loss, p_loss) if return_losses else loss

    def p_loss(self, x, y):
        # fake = self.generator(x)[0].repeat(1,3,1,1)
        fake = x.repeat(1,3,1,1)
        real = y.repeat(1,3,1,1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10):
        y, fake = self.align_size(y, fake)
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

    def align_size(self, x, y):
        if x.size(0) == y.size(0) : 
            pass
        elif x.size(0) > y.size(0):
            x = x[0:y.size(0), :, :, :]
        elif x.size(0) < y.size(0) : 
            y = y[0:x.size(0), :, :, :]
        return x,y

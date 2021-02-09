import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19
import torch.nn.functional as F
from . import Discriminator, FeatureExtractor, get_base_model

def make_model(opt):
    return WGAN_VGG_rev(opt)


class WGAN_VGG_rev(nn.Module):
    # referred from https://github.com/kuc2477/pytorch-wgan-gp
    def __init__(self, opt):
        input_size = opt.patch_size
        super(WGAN_VGG_rev, self).__init__()
        self.dc_input = opt.dc_input
        self.change_contents = opt.content_randomization

        self.generator = get_base_model(opt)
        self.discriminator = Discriminator(input_size, opt.n_channels)
        if self.dc_input =='c_img' or self.dc_input == 'c_noise':
            self.dc_channel = 2*opt.n_channels
        elif self.dc_input == 'feature':
            self.dc_channel = 64*2**(opt.style_stage if opt.style_stage<4 else 6-opt.style_stage) #128 256 512 256 128 64
            input_size = (opt.patch_size//8)*2**(opt.style_stage-3 if opt.style_stage>3 else 3-opt.style_stage) #40 20 10 20 40 80 
        else :
            self.dc_channel = opt.n_channels
        self.domain_discriminator = Discriminator(input_size, self.dc_channel)
        self.feature_extractor = FeatureExtractor()

        self.p_criterion = nn.L1Loss() #perceptual loss
        self.l_criterion = nn.L1Loss() #l1 pixelwise loss
        self.vgg_weight = opt.vgg_weight #perceptual loss weight
        self.rev_weight = opt.rev_weight #reversal gradient loss weight
        self.l_weight = opt.l_weight #l1 pixelwise loss weight

    def d_loss(self, x, y, gp=True, return_losses=False):
        self.generator.eval()
        self.discriminator.train()

        fake,_ = self.generator(x)
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
    
    def dc_loss(self, src, src_lbl, trg, gp=True, return_losses=False):
        dc_input = self.dc_input
        self.generator.eval()
        self.domain_discriminator.train()

        self.src_out, self.src_feature = self.generator(src)
        self.trg_out, self.trg_feature = self.generator(trg)

        # d_src = self.domain_discriminator(src_out.detach())
        # d_trg = self.domain_discriminator(trg_out.detach())
        if self.change_contents:
            src_out, trg_out = self.content_randomization(self.src_out, self.trg_out)
            src_feature, trg_feature = self.content_randomization(self.src_feature, self.trg_feature)
        else : 
            src_out, trg_out = self.src_out, self.trg_out
            src_feature, trg_feature = self.src_feature, self.trg_feature

        if dc_input == 'img':
            d_src = self.domain_discriminator(src_out.detach())
            d_trg = self.domain_discriminator(trg_out.detach())
            gp_loss = self.gp(src_out.detach(), trg_out.detach(), net='domain_discriminator') if gp else 0
        elif dc_input == 'noise': #src_out
            d_src = self.domain_discriminator(src_out.detach()-src)
            d_trg = self.domain_discriminator(trg_out.detach()-trg)
            gp_loss = self.gp(src_out.detach()-src, trg_out.detach()-trg, net='domain_discriminator') if gp else 0
        elif dc_input == 'feature':
            d_src = self.domain_discriminator(src_feature.detach())
            d_trg = self.domain_discriminator(trg_feature.detach())
            gp_loss = self.gp(src_feature.detach(), trg_feature.detach(), net='domain_discriminator') if gp else 0
        elif dc_input == 'c_img': #concat2
            d_src = self.domain_discriminator(torch.cat((src_lbl, src_out.detach()), 1))
            d_trg = self.domain_discriminator(torch.cat((trg_out.detach(), trg_out.detach()), 1))
            gp_loss = self.gp(torch.cat((src_lbl, src_out.detach()), 1), torch.cat((trg_out.detach(), trg_out.detach()),1), net='domain_discriminator') if gp else 0
        elif dc_input == 'c_noise': #concat
            d_src = self.domain_discriminator(torch.cat((src_lbl-src, src_out.detach()-src), 1))
            d_trg = self.domain_discriminator(torch.cat((trg_out.detach()-trg, trg_out.detach()-trg), 1))
            gp_loss = self.gp(torch.cat((src_lbl-src, src_out.detach()-src), 1), torch.cat((trg_out.detach()-trg, trg_out.detach()-trg),1), net='domain_discriminator') if gp else 0
        elif dc_input == 'c_feature': 
            raise NotImplementedError('you have to implement concat_feature')
        else:
            raise ValueError("Need to specify domain classifier input")

        adv_loss = -torch.mean(d_trg) + torch.mean(d_src)
        
        if not gp:
            gp_loss = torch.from_numpy(np.array(0.0))

        loss = adv_loss + gp_loss
        return (loss, gp_loss) if return_losses else loss
   

    def g_loss(self, x, y, perceptual=True, pixel_wise=False, return_losses=True):
        self.generator.train()
        self.discriminator.eval()
        self.domain_discriminator.eval()
    
        self.src_out, self.src_feature  = self.generator(x)
        d_fake = self.discriminator(self.src_out) 
        adv_loss = -torch.mean(d_fake) 
        if perceptual:
            p_loss = self.vgg_weight * self.p_loss(self.src_out, y)
            loss = adv_loss + p_loss
        else:
            p_loss = torch.from_numpy(np.array(0.0))
            loss = adv_loss
        if pixel_wise:
            px_loss = self.l_weight * self.l_criterion(self.src_out, y)
            loss = loss + px_loss
        else : 
            px_loss = torch.from_numpy(np.array(0.0))

        return (loss, adv_loss, px_loss, p_loss) if return_losses else loss

    def rev_loss(self, src, src_lbl):
        self.generator.train()
        self.discriminator.eval()
        self.domain_discriminator.eval()

        self.src_out, self.src_feature  = self.generator(src)

        if self.dc_input == 'img':
            rev_loss = -self.rev_weight * torch.mean(self.domain_discriminator(self.src_out))
        elif self.dc_input == 'noise':
            rev_loss = -self.rev_weight * torch.mean(self.domain_discriminator(self.src_out-src))
        elif self.dc_input == 'feature':
            rev_loss = -self.rev_weight * torch.mean(self.domain_discriminator(self.src_feature))
        elif self.dc_input == 'c_img':
            rev_loss = -self.rev_weight * torch.mean(self.domain_discriminator(torch.cat((self.src_out, src_lbl), 1)))
        elif self.dc_input == 'c_noise':
            rev_loss = -self.rev_weight * torch.mean(self.domain_discriminator(torch.cat((self.src_out-src, src_lbl-src), 1)))
        elif self.dc_input == 'c_feature':
            raise NotImplementedError('you have to implement concat_feature')

        return rev_loss

    def p_loss(self, x, y):
        # fake = self.generator(x)[0].repeat(1,3,1,1)
        fake = x.repeat(1,3,1,1)
        real = y.repeat(1,3,1,1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10, net='discriminator'):
        y, fake = self.align_size(y, fake)
        assert y.size() == fake.size()
        a = torch.cuda.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        if net == 'discriminator':
            d_interp = self.discriminator(interp)
        elif net == 'domain_discriminator':
            d_interp = self.domain_discriminator(interp)
        else : 
            raise KeyError('net name must be [discriminator or domain_discriminator]')
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

    def content_randomization(self, src, trg):
        eps = 1e-5
        x=torch.cat((src,trg),0)
        n,c,h,w=x.size()
        x=x.view(n,c,-1)
        mean=x.mean(-1,keepdim=True)
        var=x.var(-1,keepdim=True)

        x=(x-mean)/(var+eps).sqrt()

        idx_swap=torch.randperm(n)
        x=x[idx_swap].detach()

        x=x*(var+eps).sqrt()+mean
        x=x.view(n,c,h,w)

        return x[:int(n/2)], x[int(n/2):]

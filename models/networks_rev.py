import torch
import torch.nn as nn
import torch.nn.functional as F
from . import get_base_model, FeatureExtractor, Discriminator

from models.convs import common

def make_model(opt):
    return Networks_rev(opt)
    

class Networks_rev(nn.Module):
    def __init__(self, opt):
        super(Networks_rev, self).__init__()
        self.rev = True
        self.change_contents = opt.content_randomization
        self.denoiser = get_base_model(opt)
        self.denoiser.rev = self.rev
        self.dc_input = opt.dc_input
        input_size = opt.patch_size

        if self.dc_input =='c_img' or self.dc_input == 'c_noise':
            self.dc_channel = 2*opt.n_channels
        elif self.dc_input == 'feature' and opt.model == 'unet':
            self.dc_channel = 64*2**(opt.style_stage if opt.style_stage<4 else 6-opt.style_stage) #128 256 512 256 128 64
            input_size = (opt.patch_size//8)*2**(opt.style_stage-3 if opt.style_stage>3 else 3-opt.style_stage) #40 20 10 20 40 80 
        elif self.dc_input == 'feature' and opt.model == 'edsr':
            #clarrify self.dc_channel & input_size 
            raise NotImplementedError('you have to clarrify self.dc_channel and input_size of feature')
        else :
            self.dc_channel = opt.n_channels
        self.domain_discriminator = Discriminator(input_size, self.dc_channel)
        self.feature_extractor = FeatureExtractor()

        self.p_criterion = nn.L1Loss() #perceptual loss
        self.l_criterion = nn.L1Loss() #l1 pixelwise loss
        self.dc_criterion = nn.MSELoss() #domain discriminator loss

        self.vgg_weight = opt.vgg_weight #perceptual loss weight
        self.l_weight = opt.l_weight #l1 pixelwise loss weight
        self.rev_weight = opt.rev_weight #reversal gradient loss weight

    def dc_loss(self, src, src_lbl, trg):
        dc_input = self.dc_input
        self.denoiser.eval()
        self.domain_discriminator.train()

        self.src_out, self.src_feature = self.denoiser(src)
        self.trg_out, self.trg_feature = self.denoiser(trg)

        if self.change_contents:
            src_out, trg_out = self.content_randomization(self.src_out, self.trg_out)
            src_feature, trg_feature = self.content_randomization(self.src_feature, self.trg_feature)
        else : 
            src_out, trg_out = self.src_out, self.trg_out
            src_feature, trg_feature = self.src_feature, self.trg_feature

        if dc_input == 'img':
            d_src = self.domain_discriminator(src_out.detach())
            d_trg = self.domain_discriminator(trg_out.detach())
        elif dc_input == 'noise': #src_out
            d_src = self.domain_discriminator(src_out.detach()-src)
            d_trg = self.domain_discriminator(trg_out.detach()-trg)
        elif dc_input == 'feature':
            d_src = self.domain_discriminator(src_feature.detach())
            d_trg = self.domain_discriminator(trg_feature.detach())
        elif dc_input == 'c_img': #concat2
            d_src = self.domain_discriminator(torch.cat((src_lbl, src_out.detach()), 1))
            d_trg = self.domain_discriminator(torch.cat((trg_out.detach(), trg_out.detach()), 1))
        elif dc_input == 'c_noise': #concat
            d_src = self.domain_discriminator(torch.cat((src_lbl-src, src_out.detach()-src), 1))
            d_trg = self.domain_discriminator(torch.cat((trg_out.detach()-trg, trg_out.detach()-trg), 1))
        elif dc_input == 'c_feature': 
            raise NotImplementedError('you have to implement concat_feature')
        else:
            raise ValueError("Need to specify domain classifier input")
        
        trg_class = self.get_target_tensor(d_trg, True)
        src_class = self.get_target_tensor(d_src, False)
        loss = (self.dc_criterion(d_trg, trg_class) + self.dc_criterion(d_src, src_class))*0.5

        return loss

    def p_loss(self, x, y):
        # fake = self.generator(x)[0].repeat(1,3,1,1)
        fake = x.repeat(1,3,1,1)
        real = y.repeat(1,3,1,1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss


    def g_loss(self, src, src_lbl, perceptual=True, rev=True, return_losses=True):
        self.denoiser.train()
        self.domain_discriminator.eval()
    
        self.src_out, self.src_feature  = self.denoiser(src)
        l_loss = self.l_weight * self.l_criterion(self.src_out, src_lbl)
        if perceptual:
            p_loss = self.vgg_weight * self.p_loss(self.src_out, src_lbl)
        else:
            p_loss = torch.from_numpy(np.array(0.0))
        
        if self.dc_input == 'img':
            d_src = self.domain_discriminator(self.src_out)
        elif self.dc_input == 'noise':
            d_src = self.domain_discriminator(self.src_out-src)
        elif self.dc_input == 'feature':
            d_src = self.domain_discriminator(self.src_feature)
        elif self.dc_input == 'c_img':
            d_src = self.domain_discriminator(torch.cat((self.src_out, src_lbl), 1))
        elif self.dc_input == 'c_noise':
            d_src = self.domain_discriminator(torch.cat((self.src_out-src, src_lbl-src), 1))
        else:
            raise NotImplementedError('you have to implement dc_input {}'.format(self.dc_input))

        if rev : 
            src_class = self.get_target_tensor(d_src, True)
            rev_loss = self.rev_weight * self.dc_criterion(d_src, src_class)
        else : 
            rev_loss = torch.from_numpy(np.array(0.0))
        loss = l_loss + p_loss + rev_loss

        return (loss, l_loss, p_loss, rev_loss) if return_losses else loss


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
    
    def get_target_tensor(self, prediction, real):
        if real :
            target_tensor = torch.ones(prediction.size())
        else : 
            target_tensor = torch.zeros(prediction.size())

        if prediction.is_cuda : 
            target_tensor = target_tensor.cuda() 

        return target_tensor


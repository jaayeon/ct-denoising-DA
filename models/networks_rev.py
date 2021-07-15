import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from . import get_base_model, FeatureExtractor, Discriminator, NLayerDiscriminator

from models.convs import common

def make_model(opt):
    return Networks_rev(opt)

class Identity(nn.Module):
    def forward(self, x):
        return x

class Networks_rev(nn.Module):
    def __init__(self, opt):
        super(Networks_rev, self).__init__()
        self.change_contents = opt.content_randomization
        self.rev = True
        self.denoiser = get_base_model(opt)
        self.denoiser.rev = self.rev
        self.dc_input = opt.dc_input
        self.dc_mode = opt.dc_mode
        self.opt = opt
        input_size = opt.patch_size

        self.src_param = [0.250, 0.327] #phantom ge chest, pelvis's mean, std
        self.trg_param = [0.377, 0.338] #mayo 1,3mm's mean, std

        if self.dc_input =='c_img' or self.dc_input == 'c_noise':
            self.dc_channel = 2*opt.n_channels
        elif self.dc_input == 'feature' and opt.model == 'unet':
            self.dc_channel = 64*2**(opt.style_stage if opt.style_stage<4 else 6-opt.style_stage) #128 256 512 256 128 64
            input_size = (opt.patch_size//8)*2**(opt.style_stage-3 if opt.style_stage>3 else 3-opt.style_stage) #40 20 10 20 40 80 
        elif self.dc_input == 'feature' and opt.model == 'edsr':
            self.dc_channel = 96
        else :
            self.dc_channel = opt.n_channels

        self.p_criterion = nn.L1Loss() #perceptual loss
        self.l_criterion = nn.L1Loss() #l1 pixelwise loss
        if self.dc_mode == 'mse':
            self.dc_criterion = nn.MSELoss() #domain discriminator loss
            class_num = 1
        elif self.dc_mode == 'bce':
            self.dc_criterion = nn.BCEWithLogitsLoss()
            class_num = 2
        elif self.dc_mode == 'ce':
            self.dc_criterion == nn.CrossEntropyLoss()
            class_num = 2
        elif self.dc_mode == 'wss':
            class_num = 1
            pass
        
        norm_layer = self.get_norm_layer(norm_type=opt.norm)
        if opt.model_d=='fc':
            self.domain_discriminator = Discriminator(input_size, self.dc_channel, class_num=class_num, norm_layer=norm_layer, norm=opt.input_norm)
        elif opt.model_d=='conv':
            self.domain_discriminator = NLayerDiscriminator(self.dc_channel, norm_layer=norm_layer, norm=opt.input_norm)
        else:
            raise NotImplementedError('model_d should be one of [fc, conv] options')
        self.feature_extractor = FeatureExtractor()

        self.vgg_weight = opt.vgg_weight #perceptual loss weight
        self.sl_weight = opt.sl_weight #l1 pixelwise loss weight in src img
        self.tl_weight = opt.tl_weight #l1 pixelwise loss weight in trg img
        self.rev_weight = opt.rev_weight #reversal gradient loss weight

        self.sl_weight_iter = 0

    def dc_loss(self, src, src_lbl, trg, ntrg=None):
        self.set_requires_grad(self.denoiser, requires_grad=False)
        self.set_requires_grad(self.domain_discriminator, requires_grad=True)

        # if self.change_contents:
        #     src_out, trg_out, idx_swap = self.content_randomization(self.src_out, self.trg_out, return_idx=True)
        #     src_lbl, trg_out = self.content_randomization(src_lbl, self.trg_out, idx_swap=idx_swap)
        #     src_feature, trg_feature = self.content_randomization(self.src_feature, self.trg_feature)
        #     src, trg = self.content_randomization(src, trg, idx_swap=idx_swap)
        # else : 
        #     src_out, trg_out = self.src_out, self.trg_out
        #     src_feature, trg_feature = self.src_feature, self.trg_feature

        if self.dc_input == 'img':
            d_src = self.domain_discriminator(src_out.detach())
            d_trg = self.domain_discriminator(trg_out.detach())
            gp_loss = self.gp(src_out.detach(), trg_out.detach()) if self.dc_mode=='wss' else 0
        elif self.dc_input == 'origin':
            input = torch.cat((src, trg), 0)
            d_input = self.domain_discriminator(input)
        elif self.dc_input == 'noise': #src_out
            d_src = self.domain_discriminator(src_out.detach()-src)
            d_trg = self.domain_discriminator(trg_out.detach()-trg)
            gp_loss = self.gp(src_out.detach()-src, trg_out.detach()-trg) if self.dc_mode=='wss' else 0
        elif self.dc_input == 'feature':
            d_src = self.domain_discriminator(src_feature.detach())
            d_src_out = self.domain_discriminator(self.src_out_feature.detach())
            d_src_ref = self.domain_discriminator(self.src_lbl_feature.detach())
            d_trg = self.domain_discriminator(trg_feature.detach())

            self.ntrg_out, self.ntrg_feature = self.denoiser(ntrg, param=self.trg_param) if ntrg != None else [None, None]
            d_ntrg = self.domain_discriminator(self.ntrg_feature.detach()) if ntrg != None else torch.ones(d_trg.size()).to(self.opt.device)
            gp_loss = self.gp(src_feature.detach(), trg_feature.detach()) if self.dc_mode=='wss' else 0
        else:
            raise ValueError("Need to specify domain classifier input")
        
        if self.dc_mode in ['mse', 'bce']:
            dim = d_input.size()[0]
            d_src = d_input[:int(dim/2),...]
            d_trg = d_input[int(dim/2):,...]
            src_class = self.get_tensor(d_src, 0, loss=self.dc_mode)
            trg_class = self.get_tensor(d_trg, 1, loss=self.dc_mode)
            loss = 0.5*(self.dc_criterion(d_trg, trg_class) + self.dc_criterion(d_src, src_class))
        elif self.dc_mode == 'ce':
            src_class = self.get_tensor(d_src, 0, loss=self.dc_mode)
            trg_class = self.get_tensor(d_trg, 1, loss=self.dc_mode)
            ntrg_class = self.get_tensor(d_ntrg, 2, loss=self.dc_mode)
            loss = 1/3 * (self.dc_criterion(d_src, src_class) + self.dc_criterion(d_trg, trg_class) + self.dc_criterion(d_ntrg, ntrg_class))
        elif self.dc_mode == 'wss':
            loss = -torch.mean(d_trg) + torch.mean(d_src) + gp_loss

        return loss

    def p_loss(self, x, y):
        # fake = self.generator(x)[0].repeat(1,3,1,1)
        fake = x.repeat(1,3,1,1)
        real = y.repeat(1,3,1,1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    #new version 05.06
    def g_loss(self, src, trg, src_lbl, trg_noise=None, rev=True, saliency=False, return_losses=True):
        '''
        step 1 : Supervised Learning with Denoiser L(D(src), src*) ... trg_noise=None, rev=False
        step 2 : Supervised Learning with Denoiser L(D(src|n_trg), src*|trg), Domain Classifier Adversarial Loss for Denoiser with L(DC(trg'),0) ... trg_noise=arr, rev=True
        one step : Supervised Learning with Denoiser L(D(src|n_trg), src*|trg), Domain Classifier Adversarial Loss for Denoiser with L(DC(trg'),0) ... trg_noise=arr, rev=True
        '''

        if not self.opt.pretrained or not self.dc_input == 'feature' : # when in the 1st step OR in the 2nd step without feature dc_input 
            self.set_requires_grad(self.denoiser, requires_grad=True)
            self.set_requires_grad(self.domain_discriminator, requires_grad=False)
        elif self.opt.pretrained and self.dc_input=='feature' and self.opt.model == 'edsr': # when in the 2nd step with feature dc_input (default model is edsr)
            self.set_requires_grad([self.denoiser.head, self.denoiser.body1], requires_grad=True)
            self.set_requires_grad([self.denoiser.body2, self.denoiser.tail, self.domain_discriminator], requires_grad=False)
        else : 
            raise NotImplementedError('if dc_input is feature and you are using other networks, not edsr, you have to specify network modules of required gradient')

        # self.src_out, self.src_feature = self.denoiser(src, param=self.src_param)
        self.trg_out, self.trg_feature = self.denoiser(trg, param=self.trg_param)

        self.sl_weight_scheduler(gamma=0.1, max_iter=10, step=10000)

        if not trg_noise==None:
            batch_size = src.size()[0]
            input = torch.cat((src, trg_noise),0)
            out, _ = self.denoiser(input)
            self.src_out = out[:batch_size,...]
            self.n_trg_out = out[batch_size:,...]

            ntrg_loss = self.tl_weight*self.l_criterion(self.n_trg_out, trg)
        else:
            self.src_out, _ = self.denoiser(src)

            ntrg_loss = torch.zeros(1, dtype=torch.float, device=self.opt.device)

        if saliency:
            saliency_mask = self.get_saliency_map(self.domain_discriminator, src, loss=self.dc_mode, cls_idx=1, norm_param=self.src_param)
            src_loss = self.sl_weight*self.l_criterion(saliency_mask*self.src_out, saliency_mask*src_lbl)
        else :
            src_loss = self.sl_weight*self.l_criterion(self.src_out, src_lbl)

        p_src_loss = torch.zeros(1, dtype=torch.float, device=self.opt.device)
        p_ntrg_loss = torch.zeros(1, dtype=torch.float, device=self.opt.device)

        l_loss = src_loss + ntrg_loss
        #perceptual loss
        p_loss = p_src_loss + p_ntrg_loss
        
        #domain classifier loss
        if rev and (self.dc_input == 'img' or self.dc_input == 'origin'):
            # d_trg = self.domain_discriminator(self.trg_out)
            # d_src = self.domain_discriminator(self.src_out)
            d_trg = self.domain_discriminator(self.trg_out, param=self.trg_param)
            d_src = self.domain_discriminator(self.src_out, param=self.src_param)
        elif rev and self.dc_input == 'noise':
            d_trg = self.domain_discriminator(self.trg_out-trg)
            d_src = self.domain_discriminator(self.src_out-src)
        elif rev and self.dc_input == 'feature':
            d_trg = self.domain_discriminator(self.trg_feature)
            d_src = self.domain_discriminator(self.src_feature)
        else:
            pass

        if rev and self.dc_mode in ['mse', 'bce'] : 
            trg_class = self.get_tensor(d_trg, 1)
            src_class = self.get_tensor(d_src, 1)
            # domain_loss = self.rev_weight * self.dc_criterion(d_trg, trg_class)
            domain_loss = self.rev_weight * 0.5*(self.dc_criterion(d_trg, trg_class)+self.dc_criterion(d_src, src_class))
        elif rev : #wss
            domain_loss = -self.rev_weight * torch.mean(d_trg) #not sure,, changed src->trg
        else : 
            # no rev
            domain_loss = torch.zeros(1, dtype=torch.float, device=self.opt.device)

        #weighted sum  
        loss = l_loss + p_loss + domain_loss

        return (loss, l_loss, p_loss, domain_loss) if return_losses else loss

    def gp(self, y, fake, lambda_=10):
        y, fake = self.align_size(y, fake)
        assert y.size() == fake.size()
        a = torch.cuda.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.domain_discriminator(interp)
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

    def content_randomization(self, src, trg, idx_swap=None, return_idx=False):
        eps = 1e-5
        x=torch.cat((src,trg),0)
        n,c,h,w=x.size()
        x=x.view(n,c,-1)
        mean=x.mean(-1,keepdim=True)
        var=x.var(-1,keepdim=True)

        x=(x-mean)/(var+eps).sqrt()
        if idx_swap == None: #if there's no designated idx_swap, give new idx_swap
            idx_swap=torch.randperm(n)
        x=x[idx_swap].detach()

        x=x*(var+eps).sqrt()+mean 
        x=x.view(n,c,h,w)

        if return_idx:
            return x[:int(n/2)], x[int(n/2):], idx_swap
        else : 
            return x[:int(n/2)], x[int(n/2):]

    def get_tensor(self, prediction, cls_idx, loss='mse'):
        if loss in ['mse', 'bce']:  
            target_tensor = cls_idx*torch.ones(prediction.size())
        elif loss=='ce':
            target_tensor = cls_idx*torch.ones(prediction.size()[0])

        if prediction.is_cuda : 
            target_tensor = target_tensor.cuda() 

        return target_tensor

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_saliency_map(self, net, img, loss='mse', cls_idx=0, norm_param=None):
        img.requires_grad_()
        for param in net.parameters():
            param.requires_grad=True
        output = net(img, param=norm_param)
        if loss == 'mse':
            mse=torch.nn.MSELoss()
            target_tensor = cls_idx*torch.ones(output.size()).to(self.opt.device) #maximize the loss
            loss = mse(output, target_tensor)
            loss.backward()
        else : #bce, ce
            out_idx = output.argmax()
            output_max=sum(output[:, out_idx])
            output_max.backward()
        saliency = img.grad.data.abs()

        #reverse--> get pixel which is not important in domain decision making
        saliency[saliency<1e-8]=1e-8
        reverse_saliency = 1/saliency
        max_s = torch.max(reverse_saliency)
        rev_norm_saliency = reverse_saliency/max_s

        # rev_norm_saliency = torch.exp(-saliency)

        rev_norm_saliency.requires_grad_(False)
        for param in net.parameters():
            param.requires_grad=False
        return rev_norm_saliency

    def get_norm_layer(self, norm_type='instance'):
        """Return a normalization layer

        Parameters:
            norm_type (str) -- the name of the normalization layer: batch | instance | none

        For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
        For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
        """
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
        elif norm_type == 'none':
            def norm_layer(x): return Identity()
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
        return norm_layer

    def sl_weight_scheduler(self, gamma=0.1, max_iter=10, step=10000, type='exp'):
        if self.sl_weight_iter == max_iter:
            self.sl_weight = 0
            return
        else :
            self.sl_weight_iter += 1
        if type=='exp':
            self.sl_weight = self.sl_weight*gamma
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19
import torch.nn.functional as F

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
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False):
        super().__init__()
        
        conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        batch_norm = nn.BatchNorm2d(out_ch)
        relu = nn.ReLU(inplace=True)
        if bn : 
            self.layer = nn.Sequential(conv, batch_norm, relu)
        else : 
            self.layer = nn.Sequential(conv, relu)

    def forward(self, x):
        out = self.layer(x)
        return out

#sagnet : for batch normalizatin parameter reversal
class WGAN_VGG_generator(nn.Module):
    def __init__(self,opt,sagnet=False,block=BasicBlock):
        super(WGAN_VGG_generator, self).__init__()

        self.nc = opt.n_channels
        self.style_stage = opt.style_stage
        self.output_size = opt.patch_size
        self.sagnet = sagnet
        self.inc = down(block,self.nc,64,2,downsample=False,bn=sagnet)

        self.down1 = down(block,64,128,3,bn=sagnet)
        self.down2 = down(block,128,256,3,bn=sagnet)
        self.down3 = down(block,256,512,6,bn=sagnet)
        self.up1 = up(block,512,256,3,bn=sagnet)
        self.up2 = up(block,256,128,3,bn=sagnet)
        self.up3 = up(block,128,64,3,bn=sagnet)

        self.outc = nn.Conv2d(64,self.nc,1)


    def forward(self, inx):
        x = self.inc(inx)

        self.down = []
        for i, layer in enumerate([self.down1, self.down2, self.down3, self.up1, self.up2, self.up3]):
            if i<3: #down
                self.down.append(x) #down=[d1,d2,d3]
                x = layer(x)
            else: #up 345-->210
                x = layer(x,self.down[5-i]) #i:[3,4,5]-->down:[d3,d2,d1]
            if i+1==self.style_stage:
                feature=x
        
        out = self.outc(x)
        out = out + inx

        return out, feature
    
    def style_params(self):
        params=[]
        layers=[self.inc, self.down1, self.down2, self.down3, self.up1, self.up2, self.up3]
        for i, layer in enumerate(layers):
            if i <=  self.style_stage:
                for m in layer.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        params += [p for p in m.parameters()]
            else : 
                pass
        return params
            


class down(nn.Module):
    def __init__(self, block, in_ch, out_ch, rep, downsample=True, bn=False):
        super(down, self).__init__()
        layers = []
        if downsample: 
            layers.append(nn.AvgPool2d(2))
        layers.append(block(in_ch, out_ch, bn=bn))

        for _ in range(rep-1):
            layers.append(block(out_ch, out_ch, bn=bn))

        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        return out

class up(nn.Module):
    def __init__(self, block, in_ch, out_ch, rep, bn=False):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        layers = []
        for _ in range(rep):
            layers.append(block(out_ch,out_ch,bn=bn))

        self.conv = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        x = x2 + x1
        out = self.conv(x)
        return out

class WGAN_VGG_discriminator(nn.Module):
    def __init__(self, input_size, input_channels):
        super(WGAN_VGG_discriminator, self).__init__()
        def conv_output_size(input_size, kernel_size_list, stride_list):
            n = (input_size - kernel_size_list[0]) // stride_list[0] + 1
            for k, s in zip(kernel_size_list[1:], stride_list[1:]):
                n = (n - k) // s + 1
                # n = (n - k + 2*1) // s + 1
            return n

        def add_block(layers, ch_in, ch_out, stride):
            layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 0))
            # layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 1))
            layers.append(nn.LeakyReLU())
            return layers

        layers = []
        ch_stride_set = [(input_channels,64,1),(64,64,2),(64,128,1),(128,128,2),(128,256,1),(256,256,2)]
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
        x = self.normalize(x)
        out = self.feature_extractor(x)
        return out

    def normalize(self, x):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        for i, (m,s) in enumerate(zip(mean, std)):
            x[:,i:i+1,:,:] = (x[:,i:i+1,:,:] - m)/s
        return x


class WGAN_VGG(nn.Module):
    # referred from https://github.com/kuc2477/pytorch-wgan-gp
    def __init__(self, opt):
        input_size = opt.patch_size
        super(WGAN_VGG, self).__init__()
        self.generator = WGAN_VGG_generator(opt, sagnet=opt.sagnet)
        self.discriminator = WGAN_VGG_discriminator(input_size, opt.n_channels)
        if opt.dc_input =='concat' or opt.dc_input == 'concat2':
            self.dc_channel = 2*opt.n_channels
        elif opt.dc_input == 'feature':
            self.dc_channel = 64*2**(opt.style_stage if opt.style_stage<4 else 6-opt.style_stage) #128 256 512 256 128 64
            input_size = (opt.patch_size//8)*2**(opt.style_stage-3 if opt.style_stage>3 else 3-opt.style_stage) #40 20 10 20 40 80 
        else :
            self.dc_channel = opt.n_channels
        self.domain_discriminator = WGAN_VGG_discriminator(input_size, self.dc_channel)

        self.feature_extractor = WGAN_VGG_FeatureExtractor()
        self.p_criterion = nn.L1Loss() #perceptual loss
        self.l_criterion = nn.L1Loss() #l1 pixelwise loss
        self.vgg_weight = opt.vgg_weight #perceptual loss weight
        self.rev_weight = opt.rev_weight #reversal gradient loss weight
        self.l_weight = opt.l_weight #l1 pixelwise loss weight
        self.dc_input = opt.dc_input

    def d_loss(self, x, y, gp=True, return_gp=False):
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
        return (loss, gp_loss) if return_gp else loss
    
    def adv_loss(self, src, src_lbl, trg, gp=True, return_gp=False):
        dc_input = self.dc_input
        self.generator.eval()
        self.domain_discriminator.train()

        src_out, src_feature = self.generator(src)
        trg_out, trg_feature = self.generator(trg)

        # d_src = self.domain_discriminator(src_out.detach())
        # d_trg = self.domain_discriminator(trg_out.detach())
        src_out, trg_out = self.content_randomization(src_out, trg_out)
        src_feature, trg_feature = self.content_randomization(src_feature, trg_feature)

        if dc_input == 'src_out':
            # (source'-source)
            d_src = self.domain_discriminator(src_out.detach()-src)
        elif dc_input == 'feature':
            d_src = self.domain_discriminator(src_feature.detach())
        elif dc_input == 'src_lbl':
            # (source*-source)
            d_src = self.domain_discriminator(src_lbl-src)
        elif dc_input == 'sum_lbl_out':
            # (0.5(source'-source)-0.5(source*-source))
            d_src = self.domain_discriminator(0.5*(src_lbl-src) + 0.5*(src_out.detach()-src))
        elif dc_input == 'sum_lbl_out2':
            d_src_lbl = self.domain_discriminator(src_lbl-src)
            d_src_out = self.domain_discriminator(src_out.detach()-src)
            d_src = 0.5*(d_src_lbl + d_src_out)
        elif dc_input == 'concat':
            # print(src.shape) [32, 1, 80, 80]
            # check_dimension = torch.cat((src_lbl-src, src_out.detach()-src), 1)
            # print(check_dimension.shape) [32, 2, 80, 80]
            d_src = self.domain_discriminator(torch.cat((src_lbl-src, src_out.detach()-src), 1))
        elif dc_input == 'concat2':
            d_src = self.domain_discriminator(torch.cat((src_lbl, src_out.detach()), 1))
        else:
            raise ValueError("Need to specify domain classifier input")
        
        if dc_input == 'concat':
            d_trg = self.domain_discriminator(torch.cat((trg_out.detach()-trg, trg_out.detach()-trg), 1))
        elif dc_input == 'feature':
            d_trg = self.domain_discriminator(trg_feature.detach())
        elif dc_input == 'concat2':
            d_trg = self.domain_discriminator(torch.cat((trg_out.detach(), trg_out.detach()), 1))
        else:
            d_trg = self.domain_discriminator(trg_out.detach()-trg)

        adv_loss = -torch.mean(d_trg) + torch.mean(d_src)
        
        if gp:
            # gp_loss = self.gp(src_out.detach(), trg_out.detach())
            
            if dc_input == 'src_out':
                gp_loss = self.gp(src_out.detach()-src, trg_out.detach()-trg, net='domain_discriminator')
            elif dc_input == 'feature':
                gp_loss = self.gp(src_feature.detach(), trg_feature.detach(), net='domain_discriminator')
            elif dc_input == 'src_lbl':
                # (source*-source)
                gp_loss = self.gp(src_lbl-src, trg_out.detach()-trg, net='domain_discriminator')
            elif dc_input == 'sum_lbl_out':
                # (0.5(source'-source)-0.5(source*-source))
                gp_loss = self.gp(0.5*(src_lbl-src) + 0.5*(src_out.detach()-src), trg_out.detach()-trg, net='domain_discriminator')
            elif dc_input == 'sum_lbl_out2':
                gp_loss = 0.5*(self.gp(src_lbl-src, trg_out.detach()-trg, net='domain_discriminator') + self.gp(src_out.detach()-src, trg_out.detach()-trg, net='domain_discriminator'))
            elif dc_input == 'concat':
                # concat source*-source and source'-source
                gp_loss = self.gp(torch.cat((src_lbl-src, src_out.detach()-src), 1), torch.cat((trg_out.detach()-trg, trg_out.detach()-trg),1), net='domain_discriminator')
            elif dc_input == 'concat2':
                # concat source* and source'
                gp_loss = self.gp(torch.cat((src_lbl, src_out.detach()), 1), torch.cat((trg_out.detach(), trg_out.detach()),1), net='domain_discriminator')
            loss = adv_loss + gp_loss
        else : 
            gp_loss = torch.from_numpy(np.array(0.0))
            loss = adv_loss
        return (loss, gp_loss) if return_gp else loss
   

    def g_loss(self, x, y, perceptual=True, return_p=False, pixel_wise=False, adv=False):
        self.generator.train()
        self.discriminator.eval()
        self.domain_discriminator.eval()
    
        self.fake,src_feature  = self.generator(x)
        d_fake = self.discriminator(self.fake) 
        g_loss = -torch.mean(d_fake) 
        if perceptual:
            p_loss = self.vgg_weight * self.p_loss(x, y)
            loss = g_loss + p_loss
        else:
            p_loss = torch.from_numpy(np.array(0.0))
            loss = g_loss
        if pixel_wise:
            px_loss = self.l_weight * self.l_criterion(self.fake, y)
            loss = loss + px_loss
        else : 
            px_loss = torch.from_numpy(np.array(0.0))
        if adv:
            if self.dc_input == 'concat' or self.dc_input == 'concat2':
                fg_loss = -self.rev_weight * torch.mean(self.domain_discriminator(torch.cat((self.fake, self.fake), 1)))
            elif self.dc_input == 'feature':
                fg_loss = -self.rev_weight * torch.mean(self.domain_discriminator(src_feature))
            else:
                fg_loss = -self.rev_weight * torch.mean(self.domain_discriminator(self.fake))
            loss = loss + fg_loss
        else : 
            fg_loss = torch.from_numpy(np.array(0.0))
        return (loss, px_loss, p_loss, fg_loss) if (return_p or adv) else loss

    def p_loss(self, x, y):
        fake = self.generator(x)[0].repeat(1,3,1,1)
        real = y.repeat(1,3,1,1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10, net='discriminator'):
        dc_input = self.dc_input
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

'''
class WGAN_VGG(nn.Module):
    # referred from https://github.com/kuc2477/pytorch-wgan-gp
    def __init__(self, opt):
        input_size = opt.patch_size
        super(WGAN_VGG, self).__init__()
        self.generator = WGAN_VGG_generator(opt)
        self.discriminator = WGAN_VGG_discriminator(input_size)
        self.domain_discriminator = WGAN_VGG_discriminator(input_size)
        self.feature_extractor = WGAN_VGG_FeatureExtractor()
        self.p_criterion = nn.L1Loss() #perceptual loss
        self.l_criterion = nn.L1Loss() #l1 pixelwise loss
        self.vgg_weight = opt.vgg_weight #perceptual loss weight
        self.rev_weight = opt.rev_weight #reversal gradient loss weight
        self.l_weight = opt.l_weight #l1 pixelwise loss weight

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
            # gp_loss = self.gp(src_out.detach(), trg_out.detach())
            gp_loss = self.gp(src_out.detach()-src, trg_out.detach()-trg)
            loss = adv_loss + gp_loss
        else : 
            gp_loss = torch.from_numpy(np.array(0.0))
            loss = adv_loss
        return (loss, gp_loss) if return_gp else loss
   

    def g_loss(self, x, y, perceptual=True, return_p=False, pixel_wise=False, adv=False):
        self.generator.train()
        self.discriminator.eval()
        self.domain_discriminator.eval()

        self.fake = self.generator(x)
        d_fake = self.discriminator(self.fake) 
        g_loss = -torch.mean(d_fake) 
        if perceptual:
            p_loss = self.vgg_weight * self.p_loss(x, y)
            loss = g_loss + p_loss
        else:
            p_loss = torch.from_numpy(np.array(0.0))
            loss = g_loss
        if pixel_wise:
            px_loss = self.l_weight * self.l_criterion(self.fake, y)
            loss = loss + px_loss
        else : 
            px_loss = torch.from_numpy(np.array(0.0))
        if adv:
            fg_loss = -self.rev_weight * torch.mean(self.domain_discriminator(self.fake))
            loss = loss + fg_loss
        else : 
            fg_loss = torch.from_numpy(np.array(0.0))
        return (loss, px_loss, p_loss, fg_loss) if (return_p or adv) else loss

    def p_loss(self, x, y):
        fake = self.generator(x).repeat(1,3,1,1)
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
'''
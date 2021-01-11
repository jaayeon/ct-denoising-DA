import os
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import vgg19
import torchvision.transforms as transforms

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
        self.unet = UNet(opt)

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
        self.generator = WGAN_VGG_generator(opt)
        self.discriminator = WGAN_VGG_discriminator(input_size)
        # self.domain_discriminator = WGAN_VGG_discriminator(input_size)
        self.feature_extractor = WGAN_VGG_FeatureExtractor()
        self.p_criterion = nn.L1Loss() #perceptual loss
        self.l_criterion = nn.L1Loss() #l1 pixelwise loss
        # self.p_weight = opt.p_weight #perceptual loss weight
        # self.rev_weight = opt.rev_weight #reversal gradient loss weight
        # self.l_weight = opt.l_weight #l1 pixelwise loss weight

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
        # self.domain_discriminator.train()

        src_out = self.generator(src)
        trg_out = self.generator(trg)
        # d_src = self.domain_discriminator(src_out.detach())
        # d_trg = self.domain_discriminator(trg_out.detach())
        # d_src = self.domain_discriminator(src_out.detach()-src)
        # d_trg = self.domain_discriminator(trg_out.detach()-trg)
        d_src = torch.Tensor(0.0)
        d_trg = torch.Tensor(0.0)
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
        # self.domain_discriminator.eval()

        self.fake = self.generator(x)
        d_fake = self.discriminator(self.fake) 
        g_loss = -torch.mean(d_fake) 
        if perceptual:
            p_loss = self.p_loss(x, y)
            loss = g_loss + p_loss
        else:
            p_loss = torch.from_numpy(np.array(0.0))
            loss = g_loss
        if pixel_wise:
            px_loss =  self.l_criterion(self.fake, y)
            loss = loss + px_loss
        else : 
            px_loss = torch.from_numpy(np.array(0.0))
        if adv:
            # fg_loss = -self.rev_weight * torch.mean(self.domain_discriminator(self.fake))
            fg_loss = 0.0
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


class UNet(nn.Module):
    def __init__(self, opt):
        super(UNet, self).__init__()

        c = opt.n_channels
        self.Loss = nn.L1Loss()
        
        self.inc = nn.Sequential(
            single_conv(c, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, c)

    def forward(self, x, lbl=None):

        #x = standarize_coeffs(x, ch_mean=self.ch_mean, ch_std=self.ch_std)
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        out = out + x
        #out = unstandarize_coeffs(out, ch_mean=self.ch_mean, ch_std=self.ch_std)
        if lbl == None:
            pass
        else : 
            self.loss = self.Loss(out, lbl)
        
        return out
        # return out, conv1,conv2,conv3,conv4


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
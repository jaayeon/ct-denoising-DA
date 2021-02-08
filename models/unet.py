import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common
from torchvision.models import vgg19

def make_model(opt):
    return UNet(opt)

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

class UNet_denoiser(nn.Module):
    def __init__(self, opt, rev):
        super(UNet_denoiser, self).__init__()

        block = BasicBlock
        self.nc = opt.n_channels
        self.rev = rev
        self.style_stage = opt.style_stage
        self.bn = opt.bn

        self.inc = down(block,self.nc,64,2,downsample=False,bn=self.bn)

        self.down1 = down(block,64,128,3,bn=self.bn)
        self.down2 = down(block,128,256,3,bn=self.bn)
        self.down3 = down(block,256,512,6,bn=self.bn)
        self.up1 = up(block,512,256,3,bn=self.bn)
        self.up2 = up(block,256,128,3,bn=self.bn)
        self.up3 = up(block,128,64,3,bn=self.bn)

        # self.down1 = down(block,64,128,3,bn=self.bn)
        # self.down2 = down(block,128,256,6,bn=self.bn)
        # self.up1 = up(block,256,128,3,bn=self.bn)
        # self.up2 = up(block,128,64,3,bn=self.bn)

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
        
        if self.rev : 
            return out, feature
        else : 
            return out

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

class discriminator(nn.Module):
    def __init__(self, input_size, input_channels):
        super(discriminator, self).__init__()
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
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, 256*self.output_size*self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)
        return out


class UNet(nn.Module):
    def __init__(self, opt):
        self.rev = True if opt.way=='rev' else False
        self.change_contents = opt.content_randomization
        self.denoiser = UNet_denoiser(opt, self.rev)
        self.dc_input = opt.dc_input

        if self.dc_input =='concat' or self.dc_input == 'concat2':
            self.dc_channel = 2*opt.n_channels
            input_size = opt.patch_size
        elif self.dc_input == 'feature':
            self.dc_channel = 64*2**(opt.style_stage if opt.style_stage<4 else 6-opt.style_stage) #128 256 512 256 128 64
            input_size = (opt.patch_size//8)*2**(opt.style_stage-3 if opt.style_stage>3 else 3-opt.style_stage) #40 20 10 20 40 80 
        else :
            self.dc_channel = opt.n_channels
            input_size = opt.patch_size
        self.domain_discriminator = discriminator(input_size, self.dc_channel)
        self.feature_extractor = FeatureExtractor()

        self.p_criterion = nn.L1Loss() #perceptual loss
        self.l_criterion = nn.L1Loss() #l1 pixelwise loss
        self.vgg_weight = opt.vgg_weight #perceptual loss weight
        self.rev_weight = opt.rev_weight #reversal gradient loss weight
        self.l_weight = opt.l_weight #l1 pixelwise loss weight

    def dc_loss(self, src, src_lbl, trg, gp=True, return_losses=False):
        dc_input = self.dc_input
        self.denoiser.eval()
        self.domain_discriminator.train()

        self.src_out, self.src_feature = self.denoiser(src)
        self.trg_out, self.trg_feature = self.denoiser(trg)

        # d_src = self.domain_discriminator(src_out.detach())
        # d_trg = self.domain_discriminator(trg_out.detach())
        if self.change_contents:
            src_out, trg_out = self.content_randomization(self.src_out, self.trg_out)
            src_feature, trg_feature = self.content_randomization(self.src_feature, self.trg_feature)
        else : 
            src_out, trg_out = self.src_out, self.trg_out
            src_feature, trg_feature = self.src_feature, self.trg_feature

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
        return (loss, gp_loss) if return_losses else loss
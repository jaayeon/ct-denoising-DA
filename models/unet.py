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

class UNet(nn.Module):
    def __init__(self, opt, block=BasicBlock):
        super(UNet, self).__init__()
        self.rev = False
        self.nc = opt.n_channels
        self.style_stage = opt.style_stage
        self.bn = opt.bn or opt.sagnet
        self.sagnet = opt.sagnet
        self.dc_input = opt.dc_input

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

    def style_params(self):
        params=[]
        layers=[self.inc, self.down1, self.down2, self.down3, self.up1, self.up2, self.up3, self.outc]
        if self.dc_input == 'feature':
            for i, layer in enumerate(layers):
                if i <= self.style_stage and self.sagnet:
                    for m in layer.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            params += [p for p in m.parameters()]
                elif i <= self.style_stage: 
                    params += [p for p in layer.parameters()]
                else : 
                    pass
        else : #all layers
            for i, layer in enumerate(layers):
                params += [p for p in layer.parameters()]

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

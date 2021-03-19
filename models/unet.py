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

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            raise ValueError

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
    
    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer,)


class UNet(nn.Module):
    def __init__(self, opt, block=BasicBlock, nlblock=NONLocalBlock2D):
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

        ##### you have to add non-local block here! #####
        self.nlblock = NONLocalBlock2D(64, 64, bn_layer=True)

        # self.down1 = down(block,64,128,3,bn=self.bn)
        # self.down2 = down(block,128,256,6,bn=self.bn)
        # self.up1 = up(block,256,128,3,bn=self.bn)
        # self.up2 = up(block,128,64,3,bn=self.bn)

        self.outc = nn.Conv2d(64,self.nc,1)


    def forward(self, inx):

        x = self.inc(inx)

        self.down = []
        for i, layer in enumerate([self.down1, self.down2, self.down3, self.up1, self.up2, self.up3, self.nlblock]):
            if i<3: #down
                self.down.append(x) #down=[d1,d2,d3]
                x = layer(x)
            elif i == 6:
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
        layers=[self.inc, self.down1, self.down2, self.down3, self.up1, self.up2, self.up3, self.nlblock, self.outc]
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

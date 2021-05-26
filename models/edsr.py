import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common

def make_model(opt):
    return EDSR(opt)

class EDSR(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(EDSR, self).__init__()
        self.rev = False
        self.norm = opt.norm
        n_resblocks = opt.n_resblocks
        n_feats = 96
        kernel_size = 3 
        self.n_channels = opt.n_channels
        bn = opt.bn
        bias = not bn
        act = nn.ReLU(True)

        style_stage = int(n_resblocks * opt.style_stage / 6)
        self.sub_mean = common.MeanShift(pixel_range=1, n_channels=1)
        self.add_mean = common.MeanShift(pixel_range=1, n_channels=1, sign=1)

        # define head module
        m_head = [conv(self.n_channels, n_feats, kernel_size)]

        # define body module
        m_body1 = [
            common.ResBlock(
                conv, n_feats, kernel_size, bias=bias, bn=bn, act=act, res_scale=opt.res_scale
            ) for _ in range(style_stage)
        ]
        m_body2 = [
            common.ResBlock(
                conv, n_feats, kernel_size, bias=bias, bn=bn, act=act, res_scale=opt.res_scale
            ) for _ in range(n_resblocks-style_stage)
        ]
        m_body2.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
                conv(n_feats, self.n_channels, kernel_size)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body1 = nn.Sequential(*m_body1)
        self.body2 = nn.Sequential(*m_body2)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        if self.norm: 
            x = self.sub_mean(x)
        global_res = x
        
        x = self.head(x)
        feature = self.body1(x)
        res = self.body2(feature)
        res += x
        out = self.tail(res)
        
        out += global_res
        if self.norm:
            out = self.add_mean(out)
        if self.rev : 
            return out, feature
        else : 
            return out

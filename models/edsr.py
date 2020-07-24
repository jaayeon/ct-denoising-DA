import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common

def make_model(opt):
    return EDSR(opt)

class EDSR(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(EDSR, self).__init__()

        # n_resblocks = opt.n_resblocks
        n_resblocks = 16
        # n_feats = opt.n_feats
        n_feats = 96
        kernel_size = 3 
        self.n_channels = opt.n_channels
        # bn = opt.bn
        bn = False
        # bias = not bn
        bias = True
        act = nn.ReLU(True)

        
        # pix_range = opt.pixel_range
        # self.shift_mean = opt.shift_mean
        # self.sub_mean = common.MeanShift(pix_range, n_channels=self.n_channels)
        # self.add_mean = common.MeanShift(pix_range, n_channels=self.n_channels, sign=1)

        # define head module
        m_head = [conv(self.n_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, bias=bias, bn=bn, act=act, res_scale=opt.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
                conv(n_feats, self.n_channels, kernel_size)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)

        global_res = x
        
        x = self.head(x)
        
        res = x
        x = self.body(x)
        # x += res

        x = self.tail(x)
        
        x += global_res

        out = x
        # out = self.add_mean(x)

        return out

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

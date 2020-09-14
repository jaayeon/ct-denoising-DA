import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import functools

def make_model(opt):
    return FCDiscriminator(opt)
    # return NLayerDiscriminator(opt)


def get_norm_layer(norm_type='instance'):
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


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    """Refer the following link to calculate receptive field"""

    def __init__(self, opt, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = get_norm_layer(norm_type='batch')
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        if use_sigmoid :
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x, lbl):
        """Standard forward."""
        output = self.model(x)
        self.loss = self.mse_loss(x, Variable(torch.FloatTensor(x.data.size()).fill_(lbl)).cuda())

        return output



class FCDiscriminator(nn.Module):

    def __init__(self, opt, ndf = 64):
        super(FCDiscriminator, self).__init__()

        num_classes = opt.n_channels
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ndf*2, affine=True)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ndf*4, affine=True)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(ndf*8, affine=True)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bce_loss = nn.MSELoss()


    def forward(self, x, lbl):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.bn4(x)
        x = self.classifier(x)
        self.loss = self.bce_loss(x, Variable(torch.FloatTensor(x.data.size()).fill_(lbl)).cuda())

        return x

    def adjust_learning_rate(self, args, optimizer, i):
        if args.model == 'DeepLab':
            lr = args.learning_rate_D * ((1 - float(i) / args.num_steps) ** (args.power))
            optimizer.param_groups[0]['lr'] = lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = lr * 10 
        else:
            optimizer.param_groups[0]['lr'] = args.learning_rate_D * (0.1**(int(i/50000)))
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = args.learning_rate_D * (0.1**(int(i/50000))) * 2  			

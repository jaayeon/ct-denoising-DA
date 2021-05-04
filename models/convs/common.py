import torch
import torch.nn as nn
from torch.autograd import Function


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def dilated_conv(in_channels, out_channels, kernel_size, bias=True, dilation=2):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size+(dilation-1)*2)//2,
        dilation = dilation,
        bias = True
    )


class MeanShift(nn.Module):
    def __init__(self, pixel_range, n_channels, rgb_mean=None, rgb_std=None, sign=-1):
        super(MeanShift, self).__init__()

        if rgb_mean is None and rgb_std is None:
            if n_channels == 1:
                rgb_mean = [0.5]
                rgb_std =[1.0]
            elif n_channels == 3:
                rgb_mean = (0.4488, 0.4371, 0.4040)
                rgb_std = (1.0, 1.0, 1.0)

        self.shifter = nn.Conv2d(n_channels, n_channels, 1, 1, 0)
        std = torch.Tensor(rgb_std)
        self.shifter.weight.data = torch.eye(n_channels).view(n_channels, n_channels, 1, 1) / std.view(n_channels, 1, 1, 1)
        self.shifter.bias.data = sign * pixel_range * torch.Tensor(rgb_mean) / std

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
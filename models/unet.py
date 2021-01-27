import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common
#from models.convs import attention

def make_model(opt):
    return UNet(opt)

class discriminator(nn.Module):
    def __init__(self, input_size):
        super(discriminator, self).__init__()
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
        self.mse_loss = nn.MSELoss()

    def forward(self, x, lbl=None):
        out = self.net(x)
        out = out.view(-1, 256*self.output_size*self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)

        self.loss = self.mse_loss(out, Variable(torch.FloatTensor(x.data.size()).fill_(lbl)).cuda())
        return out


class UNet(nn.Module):
    def __init__(self, opt):
        super(UNet, self).__init__()
        self.discriminator = discriminator(opt.patch_size)

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
        return out, conv2


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
import torch
import torch.nn as nn


# WGAN-VGG Discriminator

def create_domain_classifier(domain_classifier, input_size, input_channels, class_num):
    if domain_classifier == 'WG':
        return WGDiscriminator(input_size, input_channels, class_num)
    elif domain_classifier == 'scnn':
        return SCNN(input_size, input_channels, class_num)
    elif domain_classifier == 'dc':
        return Discriminator(input_size, input_channels, class_num)


class Discriminator(nn.Module):
    def __init__(self, input_size, input_channels, class_num=1):
        super(Discriminator, self).__init__()
        self.input_size = input_size
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
        self.fc2 = nn.Linear(1024, class_num)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.random_crop(x)
        out = self.net(x)
        out = out.view(-1, 256*self.output_size*self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)
        return out

    def random_crop(self, x):
        size = x.size()[-1] #b,c,h,w
        if size == self.input_size:
            return x
        rh = random.randint(0, size-self.input_size)
        rw = random.randint(0, size-self.input_size)
        return x[:,:,rh:rh+self.input_size, rw:rw+self.input_size]

        

class WGDiscriminator(nn.Module):
    def __init__(self, input_size, input_channels, class_num=1):
        """Declare all needed layers."""
        super(WGDiscriminator, self).__init__()
        n_class = 2

        input_size = opt.patch_size
        n_channels = opt.n_feats
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
        ch_stride_set = [
            (n_channels, 64, 1),
            (64, 64, 2),
            (64, 128, 1),
            (128, 128, 2),
            (128, 256, 1),
            (256, 256, 2)]
        for ch_in, ch_out, stride in ch_stride_set:
            add_block(layers, ch_in, ch_out, stride)

        self.output_size = conv_output_size(input_size, [3]*6, [1,2]*3)
        # print('output_size:', self.output_size)
        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(256*self.output_size*self.output_size, 1024)
        self.fc2 = nn.Linear(1024, n_class)
        self.lrelu = nn.LeakyReLU()

        # pix_range = 1.0
        # self.sub_mean = common.MeanShift(pix_range, n_channels=n_channels)

    def forward(self, x):
        # print('x.shape:', x.shape)
        # x = self.sub_mean(x)
        out = self.net(x)
        out = out.view(-1, 256*self.output_size*self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)
        # print('out.shape:', out.shape)
        return out

def weight_init(net): 
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


# NRQM SCNN classifier
class SCNN(nn.Module):
    def __init__(self, input_size, input_channels, class_num=2):
        """Declare all needed layers."""
        super(SCNN, self).__init__()
        # Linear classifier.
        self.num_class = class_num
        
        self.nc = input_channels

        input_size = input_size

        def conv_output_size(input_size, kernel_size_list, stride_list):
            n = (input_size + 2 - kernel_size_list[0]) // stride_list[0] + 1
            for k, s in zip(kernel_size_list[1:], stride_list[1:]):
                n = (n + 2 - k) // s + 1
            
            # print('n:', n)
            # averaging pooling
            n = n - 14 + 1
            return n

        kernel_list = [3] * 9
        stride_list = [1, 2, 1, 2, 1, 2, 1, 1, 2]
        self.output_size = conv_output_size(input_size, kernel_list, stride_list)
        # print('output_size:', self.output_size)

        self.features = nn.Sequential(nn.Conv2d(self.nc, 48, 3, 1, 1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                                      nn.Conv2d(48, 48, 3, 2, 1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                                      nn.Conv2d(48, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        weight_init(self.features)
        self.pooling = nn.AvgPool2d(14, 1)
        self.projection = nn.Sequential(nn.Conv2d(128, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        weight_init(self.projection)    
        self.classifier = nn.Linear(256*self.output_size*self.output_size, self.num_class)
        weight_init(self.classifier)

    def forward(self, x):
        # print('in_x.shape:',x.shape)
        n = x.size()[0]
        # x = self.sub_mean(x)

        # assert x.size() == (n, 3, 224, 224)
        x = self.features(x)
        # print('x.shape:', x.shape)
        # assert x.size() == (n, 128, 14, 14)
        x = self.pooling(x)
        # assert x.size() == (n, 128, 1, 1)
        x = self.projection(x)
        # print('x.shape:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # assert x.size() == (n, self.num_class)
        return x

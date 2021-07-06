import torch
import torch.nn as nn
import torch.nn.functional as F
from . import get_base_model, FeatureExtractor

from models.convs import common

def make_model(opt):
    return Networks(opt)
    

class Networks(nn.Module):
    def __init__(self, opt):
        super(Networks, self).__init__()
        self.denoiser = get_base_model(opt)
        self.feature_extractor = FeatureExtractor()

        self.p_criterion = nn.L1Loss() #perceptual loss
        self.l_criterion = nn.L1Loss() #l1 pixelwise loss

        self.vgg_weight = opt.vgg_weight #perceptual loss weight
        self.l_weight = opt.sl_weight #l1 pixelwise loss weight


    def p_loss(self, x, y):
        # fake = self.generator(x)[0].repeat(1,3,1,1)
        fake = x.repeat(1,3,1,1)
        real = y.repeat(1,3,1,1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss


    def g_loss(self, x, y, perceptual=True, return_losses=True):
        # self.denoiser.train()
    
        self.out  = self.denoiser(x)
        l_loss = self.l_weight * self.l_criterion(self.out, y)
        if perceptual:
            p_loss = self.vgg_weight * self.p_loss(self.out, y)
        else:
            p_loss = torch.from_numpy(np.array(0.0))

        loss = l_loss + p_loss 

        return (loss, l_loss, p_loss) if return_losses else loss

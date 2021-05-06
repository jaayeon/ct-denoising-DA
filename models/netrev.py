"""
WGAN-VGG: Low-dose CT image denoising using a generative adversarial network with Wasserstein distance and perceptual loss

https://github.com/SSinyu/WGAN_VGG

"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import vgg19
import torch.nn.functional as F

from models.convs import common
from .base_model import BaseModel

from models.losses.perceptual_loss import parse_perceptual_loss, PerceptualLoss
from models.common.unet import make_model as create_unet
from models.common.edsr import make_model as create_edsr
from .common.classifiers import create_domain_classifier

# url = {
#     'data-mayof32g32b8l16': 'https://www.dropbox.com/s/q5bjbvm0tzdy4vk/epoch_best_n0124_loss0.00014028_psnr39.5489.pth?dl=1'
# }

class NetRev(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        
        # Network parameters
        parser.add_argument('--domain_classifier', type=str, default='dc',
            help='specify domain classifier')

        parser.add_argument('--rev_weight', type=float, default=0.001,
            help='domain classifier reversal loss')
        parser.add_argument('--dc_mode', type=str, default='mse',
            choices=['mse', 'bce', 'wss'], 
            help='domain classifier loss mode')
        parser.add_argument('--dc_input', type=str, default='c_img',
            choices=['img', 'noise', 'feature', 'c_img', 'c_noise', 'c_feature'],
            help = 'domain classifier input')
        parser.add_argument('--style_stage', type=int, default=4, choices=[1,2,3,4,5,6],
            help='stage for feature which is extracted from generator to domain classifier input')
        parser.add_argument('--content_randomization', default=False, action='store_true')
        parser.add_argument('--sagnet', default=False, action='store_true', 
            help='only update batch normalization parameters in rev_loss')
        parser.add_argument('--rev', default=True, 
            help='reversal gradient back propagation')

        parser.add_argument('--n_d_train', type=int, default=1,
            help='number of discriminator training')
        parser.add_argument('--generator', type=str, default='unet',
            help='generator model [unet | edsr]')
        # parser.add_argument('--perceptual', dest='perceptual', action='store_true',
        #     help='use perceptual loss')
        # parser.add_argument('--mse', dest='perceptual', action='store_false',
        #     help='use MSE loss')

        parser.add_argument('--perceptual_loss', type=str, default=None,
            choices=['srgan', 'wavelet_transfer', 'perceptual_loss'],
            help='specity loss_type')
        
        if is_train:
            # parser.set_defaults(perceptual=True)
            parser = parse_perceptual_loss(parser)
            parser.set_defaults(perceptual_loss='srgan')
            parser.set_defaults(lr=1e-4)
            parser.set_defaults(b1=0.5)
            parser.set_defaults(b2=0.999)


        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # Create model
        # self. = create_model(opt).to(self.device)

        if opt.generator == 'unet':
            self.net_G = create_unet(opt).to(self.device)
        elif opt.generator == 'edsr':
            self.net_G = create_edsr(opt).to(self.device)
        else:
            raise ValueError("Need to specify model {}".format(opt.generator))

        self.mse_loss_criterion = nn.MSELoss()

        self.change_contents = opt.content_randomization

        if opt.perceptual_loss is not None:
            self.perceptual_loss = True
            self.loss_type = opt.perceptual_loss
        else:
            self.perceptual_loss = False

        # Define losses and optimizers
        if self.is_train:
            # self.feature_extractor = create_vgg().to(self.device)
            self.model_names = ['net_G', 'net_D']
            
            self.p_criterion = nn.L1Loss() #perceptual loss
            self.l_criterion = nn.L1Loss() #l1 pixelwise loss

            self.dc_input = opt.dc_input
            self.dc_mode = opt.dc_mode
            input_size = opt.patch_size
            self.rev_weight = opt.rev_weight

            if self.dc_input =='c_img' or self.dc_input == 'c_noise':
                self.dc_channel = 2*opt.n_channels
            elif self.dc_input == 'feature' and opt.model == 'unet':
                self.dc_channel = 64*2**(opt.style_stage if opt.style_stage<4 else 6-opt.style_stage) #128 256 512 256 128 64
                input_size = (opt.patch_size//8)*2**(opt.style_stage-3 if opt.style_stage>3 else 3-opt.style_stage) #40 20 10 20 40 80 
            elif self.dc_input == 'feature' and opt.model == 'edsr':
                self.dc_channel = 96
            else:
                self.dc_channel = opt.n_channels
                
            if self.dc_mode == 'mse':
                self.dc_criterion = nn.MSELoss() #domain discriminator loss
                class_num = 1
            elif self.dc_mode == 'bce':
                self.dc_criterion = nn.BCEWithLogitsLoss()
                class_num = 2
            elif self.dc_mode == 'wss':
                class_num = 1
                pass

            self.net_D = create_domain_classifier(
                opt.domain_classifier,
                input_size, self.dc_channel,
                class_num
            ).to(self.device)

            self.optimizer_names = ['optimizer_G', 'optimizer_D']
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.n_d_train = opt.n_d_train
            # self.gp = True
            # self.perceptual = opt.perceptual

            if self.perceptual_loss:
                self.perceptual_loss_criterion = PerceptualLoss(opt)
        else:
            self.model_names = ['net_G']
            


    def set_input(self, input):
        self.src = input['src']
        self.src_x = self.src[0].to(self.device)
        self.src_target = self.src[1].to(self.device)
        self.src_domain_label = self.src[3].to(self.device)

        self.trg = input['trg']
        if self.trg is not None:
            self.trg_x = self.trg[0].to(self.device)
            self.trg_target = self.trg[1].to(self.device)
            self.trg_domain_label = self.trg[3].to(self.device)

    def forward_src(self):
        self.src_out, self.src_feature = self.net_G(self.src_x)

    def forward_trg(self):
        self.trg_out, self.trg_feature = self.net_G(self.trg_x)

    def forward(self):
        self.out = self.net_G(self.trg_x)

    def dc_loss(self):
        if self.change_contents:
            src_out, trg_out, idx_swap = self.content_randomization(self.src_out, self.trg_out)
            src_lbl, trg_out, _ = self.content_randomization(self.src_target, self.trg_out, idx_swap=idx_swap)
            src_feature, trg_feature, _ = self.content_randomization(self.src_feature, self.trg_feature)
            src, trg, _ = self.content_randomization(self.src_x, self.trg_x, idx_swap=idx_swap)
        else: 
            src_out, trg_out = self.src_out, self.trg_out
            src_feature, trg_feature = self.src_feature, self.trg_feature

        if self.dc_input == 'img':
            self.src_domain_pred = self.net_D(src_out.detach())
            self.trg_domain_pred = self.net_D(trg_out.detach())
            gp_loss = self.gp(src_out.detach(), trg_out.detach()) if self.dc_mode=='wss' else 0
        elif self.dc_input == 'noise': #src_out
            self.src_domain_pred = self.net_D(src_out.detach()-src)
            self.trg_domain_pred = self.net_D(trg_out.detach()-trg)
            gp_loss = self.gp(
                src_out.detach()-src,
                trg_out.detach()-trg
            ) if self.dc_mode=='wss' else 0
        elif self.dc_input == 'feature':
            self.src_domain_pred = self.net_D(src_feature.detach())
            self.trg_domain_pred = self.net_D(trg_feature.detach())
            gp_loss = self.gp(src_feature.detach(), trg_feature.detach()) if self.dc_mode=='wss' else 0
        elif self.dc_input == 'c_img': #concat2
            # d_src = self.net_D(torch.cat((src_out.detach(), src_lbl), 1))
            # d_trg = self.net_D(torch.cat((trg_out.detach(), trg_out.detach()), 1))
            # gp_loss = self.gp(torch.cat((src_out.detach(), src_lbl), 1), torch.cat((trg_out.detach(), trg_out.detach()),1)) if self.dc_mode=='wss' else 0

            self.src_domain_pred = self.net_D(torch.cat((src_out.detach(), self.src_x), 1))
            self.trg_domain_pred = self.net_D(torch.cat((trg_out.detach(), self.trg_x), 1))
            gp_loss = self.gp(
                torch.cat((src_out.detach(), self.src_x), 1),
                torch.cat((trg_out.detach(), self.trg_x), 1)
            ) if self.dc_mode=='wss' else 0

        elif self.dc_input == 'c_noise': #concat
            self.src_domain_pred = self.net_D(torch.cat((src_out.detach()-self.src_x, src_lbl-self.src_x), 1))
            self.trg_domain_pred = self.net_D(torch.cat((trg_out.detach()-self.trg_x, trg_out.detach()-self.trg_x), 1))
            gp_loss = self.gp(
                torch.cat((src_out.detach()-self.src_x, src_lbl-self.src_x), 1),
                torch.cat((trg_out.detach()-self.trg_x, trg_out.detach()-self.trg_x), 1)
            ) if self.dc_mode=='wss' else 0
        elif self.dc_input == 'c_feature': 
            raise NotImplementedError('you have to implement concat_feature')
        else:
            raise ValueError("Need to specify domain classifier input")
        
        if self.dc_mode in ['mse', 'bce']:
            trg_class = self.get_target_tensor(self.trg_domain_pred, True)
            src_class = self.get_target_tensor(self.src_domain_pred, False)
            self.d_loss = (self.dc_criterion(self.trg_domain_pred, trg_class) + self.dc_criterion(self.src_domain_pred, src_class)) * 0.5
        elif self.dc_mode == 'wss':
            self.loss = -torch.mean(self.trg_domain_pred) + torch.mean(self.src_domain_pred) + gp_loss

        return self.d_loss

    def gp(self, y, fake, lambda_=10):
        assert y.size() == fake.size()
        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1))).to(self.device)
        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.net_D(interp)
        fake_ = torch.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty
        

    def g_loss(self, trg_noise=None, rev=True):
        # lbl = self.src_target
        # out = self.src_out

        #perceptual loss
        if self.perceptual_loss:
            self.content_loss, self.style_loss = self.perceptual_loss_criterion(self.src_out, self.src_target)
            self.p_loss = self.content_loss + self.style_loss
        else:
            self.p_loss = self.l_criterion(self.src_out, self.src_target)
        
        #domain classifier loss
        if rev and (self.dc_input == 'img' or self.dc_input == 'origin'):
            self.trg_domain_pred = self.net_D(self.trg_out)
        elif rev and self.dc_input == 'noise':
            self.trg_domain_pred = self.net_D(self.trg_out-self.trg_x)
        elif rev and self.dc_input == 'feature':
            self.trg_domain_pred = self.net_D(self.trg_feature)
        elif rev and self.dc_input == 'c_img':
            # d_trg = self.net_D(torch.cat((self.src_out, src_lbl), 1))
            self.trg_domain_pred = self.net_D(torch.cat((self.trg_out, self.trg_x), 1))
        elif rev and self.dc_input == 'c_noise':
            self.trg_domain_pred = self.net_D(torch.cat((self.trg_out-trg, self.trg_out-trg), 1))
        else:
            pass

        if rev and self.dc_mode in ['mse', 'bce'] : 
            trg_class = self.get_target_tensor(self.trg_domain_pred, False)
            self.rev_loss = self.rev_weight * self.dc_criterion(self.trg_domain_pred, trg_class)
        elif rev : #wss
            self.rev_loss = -self.rev_weight * torch.mean(self.trg_domain_pred) #not sure,, changed src->trg
        else : 
            # no rev
            self.rev_loss = torch.from_numpy(np.array(0.0)).to(self.device)

        #weighted sum  
        self.loss_g = self.p_loss + self.rev_loss

        return self.loss_g

    def align_size(self, x, y):
        if x.size(0) == y.size(0) : 
            pass
        elif x.size(0) > y.size(0):
            x = x[0:y.size(0), :, :, :]
        elif x.size(0) < y.size(0) : 
            y = y[0:x.size(0), :, :, :]
        return x, y

    def content_randomization(self, src, trg, idx_swap=None):
        eps = 1e-5
        x = torch.cat((src, trg), 0)
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + eps).sqrt()
        if idx_swap == None: #if there's no designated idx_swap, give new idx_swap
            idx_swap = torch.randperm(n)
        
        x = x[idx_swap].detach() # ????? detach?????

        x = x * (var + eps).sqrt() + mean 
        x = x.view(n, c, h, w)

        return x[:int(n/2)], x[int(n/2):], idx_swap

    def get_target_tensor(self, prediction, real):
        if real:
            target_tensor = torch.ones(prediction.size())
        else: 
            target_tensor = torch.zeros(prediction.size())

        target_tensor = target_tensor.to(self.device)

        return target_tensor

    def backward_DC(self):
        self.loss_d = self.dc_loss()
        self.loss_d.backward()

    def backward_G(self):
        self.loss_g = self.g_loss()
        self.loss_g.backward()

    def optimize_parameters(self):
        self.set_requires_grad([self.net_D], True)
        for _ in range(self.n_d_train):
            self.optimizer_D.zero_grad()
            # self.net_D.zero_grad()
            self.forward_src()
            self.forward_trg()
            self.backward_DC()
            self.optimizer_D.step()

        # self.loss_d = torch.zeros([1])

        self.set_requires_grad([self.net_D], False)
        self.optimizer_G.zero_grad()
        # self.net_G.zero_grad()
        self.forward_src()
        self.forward_trg()
        self.backward_G()
        self.optimizer_G.step()
    
    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        mse_loss = self.mse_loss_criterion(self.trg_out, self.trg_target)
        self.psnr = 10 * torch.log10(1 / mse_loss)

        _, src_pred = torch.max(self.src_domain_pred, 1)
        src_correct = torch.sum(src_pred==self.src_domain_label)
        src_accr = src_correct / self.src_x.size(0)

        _, trg_pred = torch.max(self.trg_domain_pred, 1)
        print('trg_pred:', trg_pred)
        trg_correct = torch.sum(trg_pred==self.trg_domain_label)
        trg_accr = trg_correct / self.trg_x.size(0)

        src_out = self.src_out
        trg_out = self.trg_out
        smse_loss = self.mse_loss_criterion(self.src_out, self.src_target)
        spsnr = 10 * torch.log10(1 / smse_loss)
        
        nmse_loss = self.mse_loss_criterion(self.src_x, self.src_target)
        nspsnr = 10 * torch.log10(1 / nmse_loss)
        
        tmse_loss = self.mse_loss_criterion(self.trg_out, self.trg_target)
        tpsnr = 10 * torch.log10(1 / tmse_loss)
        
        tnmse_loss = self.mse_loss_criterion(self.trg_x, self.trg_target)
        ntpsnr = 10 * torch.log10(1 / tnmse_loss)

        self.loss = tnmse_loss
        self.psnr = tpsnr
        #update status
        # status = [(), p_loss.item(), rev_loss.item(), dc_loss.item(), spsnr, nspsnr, tpsnr, ntpsnr]
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss_P: {:.6f} Loss_G: {:.6f}, Loss_DC: {:.6f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter, self.p_loss.item(), self.loss_g.item(), self.loss_d.item()
        ))
        print("Source (Noise loss: {:.6f}, Noise PNSR: {:.6f}, MSE_loss: {:.6f}, PSNR: {:.6f})".format(
            nmse_loss.item(), nspsnr.item(), smse_loss.item(), spsnr.item()
        ))
        print("Target (Noise loss: {:.6f}, Noise PNSR: {:.6f}, MSE_loss: {:.6f}, PSNR: {:.6f})".format(
            tnmse_loss.item(), ntpsnr.item(), tmse_loss.item(), tpsnr.item()
        ))
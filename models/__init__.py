from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_model(opt):
    if opt.model == 'dncnn':
        module_name = 'models.dncnn'
    elif opt.model == 'unet':
        module_name = 'models.unet'
    elif opt.model == 'edsr':
        module_name = 'models.edsr'
    elif opt.model == 'unet_c':
        module_name = 'models.unet_c'  
    elif opt.model == 'auto':
        module_name = 'models.autoencoder'  
    else:
        raise ValueError("Need to specify model (redcnn, dncnn)")
    
    module = import_module(module_name)
    model = module.make_model(opt)

    if opt.use_cuda:
        model = model.to(opt.device)

    return model


def set_model_D(opt):
    if opt.model_d == 'discriminator':
        module_name = 'models.discriminator'
    else:
        module_name = 'models.wgan_discriminator'
        
    module = import_module(module_name)
    model_D = module.make_model(opt)

    if opt.use_cuda :
        model_D = model_D.to(opt.device)

    return model_D

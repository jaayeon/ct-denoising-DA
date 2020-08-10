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
    else:
        raise ValueError("Need to specify model (redcnn, dncnn)")
    
    module = import_module(module_name)
    model = module.make_model(opt)

    if opt.use_cuda:
        model = model.to(opt.device)

    return model


def set_model_D(opt):
    module_name = 'models.discriminator'
    module = import_module(module_name)
    model_D = module.make_model(opt)

    if opt.use_cuda :
        model_D = model_D.to(opt.device)

    return model_D

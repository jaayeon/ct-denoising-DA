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
    else:
        raise ValueError("Need to specify model (redcnn, dncnn)")
    
    module = import_module(module_name)
    model = module.make_model(opt)

    return model

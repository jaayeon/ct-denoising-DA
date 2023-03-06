import argparse
from utils.loader import load_config
# from pca import pca
import numpy as np
import imageio
import glob, os, math, random
import torch
from options import args
from models import networks_rev, networks_base
from utils.loader import load_config
import matplotlib.pyplot as plt

# mayo supervised
# 'mayo-20210429-0531-base-edsr'
# phantom supervised
# 'ge-20210429-0840-base-edsr-chest-pelvis'
# ours 
# 'ge-20210729-0602-rev-edsr-p-chest-pelvis'

if __name__=="__main__":
    opt=args

    #dataset 
    ge_imgs = '../../data/denoising/train/phantom/ge/chest/level5_*_crop320/*.tiff'
    mayo_imgs = '../../data/denoising/train/mayo/quarter_1mm/*/*.tiff'

    ge_img_list = glob.glob(ge_imgs)
    mayo_img_list = glob.glob(mayo_imgs)
    ge_img_list.sort()
    mayo_img_list.sort()
    random.seed(1)
    random.shuffle(ge_img_list)
    random.shuffle(mayo_img_list) 

    batch = 50
    gtensor = torch.zeros([batch,1,320,320])
    mtensor = torch.zeros([batch,1,512,512])
    gtensor_clean = torch.zeros([batch,1,320,320])
    mtensor_clean = torch.zeros([batch,1,512,512])


    for i in range(batch):
        ge = imageio.imread(ge_img_list[i])
        mayo = imageio.imread(mayo_img_list[i])
        ge_clean = imageio.imread(ge_img_list[i].replace('level5_???', 'level3_???'))
        mayo_clean = imageio.imread(mayo_img_list[i].replace('quarter', 'full'))
        gtensor[i,0,:,:] = torch.Tensor(ge)
        mtensor[i,0,:,:] = torch.Tensor(mayo)
        gtensor_clean[i,0,:,:] = torch.Tensor(ge_clean)
        mtensor_clean[i,0,:,:] = torch.Tensor(mayo_clean)

    checkpoints = ['mayo-20210429-0531-base-edsr', 'ge-20210429-0840-base-edsr-chest-pelvis', 'ge-20210729-0602-rev-edsr-p-chest-pelvis']
    dc_checkpoint = 'ge-20210622-2033-rev-edsr-chest-pelvis'
    checkpoint_dir = opt.checkpoint_dir
    dc_checkpoint_list = glob.glob(os.path.join(checkpoint_dir, dc_checkpoint, '*.pth'))
    dc_checkpoint_list.sort()
    dc_checkpoint = dc_checkpoint_list[-1]
    dc_checkpoint = torch.load(dc_checkpoint)
    dc = networks_rev.Networks_rev(opt)
    dc.load_state_dict(dc_checkpoint['model'], strict=False)
    dc.domain_discriminator.eval()
    
    for ch in range(len(checkpoints)):
        opt.checkpoint_dir = os.path.join(checkpoint_dir, checkpoints[ch])
        checkpoint_list = glob.glob(os.path.join(opt.checkpoint_dir, '*.pth'))
        checkpoint_list.sort()
        checkpoint = checkpoint_list[-1]
        print('loaded checkpoint path : {}'.format(checkpoint))

        if ch==0:
            model = networks_base.Networks(opt)
            checkpoint = torch.load(checkpoint)
            model.load_state_dict(checkpoint['model'], strict=False)

            with torch.no_grad():
                out_ge = model.denoiser(gtensor)
                out_mayo = model.denoiser(mtensor)
        else : 
            model = networks_rev.Networks_rev(opt)
            checkpoint = torch.load(checkpoint)
            model.load_state_dict(checkpoint['model'], strict=False)

            with torch.no_grad():
                out_ge, feature_ge = model.denoiser(gtensor)
                out_mayo, feature_mayo = model.denoiser(mtensor)
        
        out_ge_domain = torch.mean(torch.mean(dc.domain_discriminator(out_ge), dim=(2,3)))
        print(out_ge_domain.shape)
        out_ge_domain_avg = torch.mean(out_ge_domain)
        print(out_ge_domain_avg.shape)
        out_mayo_domain = torch.mean(torch.mean(dc.domain_discriminator(out_mayo), dim=(2,3)))
        print(out_mayo_domain.shape)
        out_mayo_domain_avg = torch.mean(out_mayo_domain)
        print(out_mayo_domain_avg.shape)
        print('domain accuracy_{}: ge_domain_avg:{}, mayo_domain_avg:{}'.format(ch, out_ge_domain_avg, out_mayo_domain_avg))




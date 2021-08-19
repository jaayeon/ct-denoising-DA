import numpy as np
import imageio
import glob, os, random
import torch
from options import args
from models import networks_rev
from utils.loader import load_config
import matplotlib.pyplot as plt

#model setup
#Ds
#'ge-20210622-2033-rev-edsr-chest-pelvis'
#0.000001
#
#0.00001
#
#0.0001
#'ge-20210819-0418-rev-edsr-p-chest-pelvis'
#0.001
#'ge-20210804-1936-rev-edsr-p-chest-pelvis'
#0.01
#'ge-20210819-0413-rev-edsr-p-chest-pelvis'
#0.1
#'ge-20210819-0425-rev-edsr-p-chest-pelvis'
#1
#'ge-20210819-0414-rev-edsr-p-chest-pelvis'
#10
#'ge-20210804-1938-rev-edsr-p-chest-pelvis'

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

    #load data
    #patch
    batch = 100
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

    # checkpoints = ['ge-20210622-2033-rev-edsr-chest-pelvis', 'ge-20210804-1936-rev-edsr-p-chest-pelvis', 'ge-20210811-0437-rev-edsr-p-chest-pelvis', 'ge-20210721-0127-rev-edsr-p-chest-pelvis', 'ge-20210721-1736-rev-edsr-p-chest-pelvis',
    #                 'ge-20210705-1432-rev-edsr-p-chest-pelvis','ge-20210721-0136-rev-edsr-p-chest-pelvis', 'ge-20210721-0132-rev-edsr-p-chest-pelvis', 'ge-20210804-1938-rev-edsr-p-chest-pelvis']
    checkpoints = ['ge-20210622-2033-rev-edsr-chest-pelvis', 'ge-20210819-0418-rev-edsr-p-chest-pelvis', 'ge-20210804-1936-rev-edsr-p-chest-pelvis',
                    'ge-20210819-0413-rev-edsr-p-chest-pelvis', 'ge-20210819-0425-rev-edsr-p-chest-pelvis', 'ge-20210819-0414-rev-edsr-p-chest-pelvis','ge-20210804-1938-rev-edsr-p-chest-pelvis']
    # checkpoints = ['ge-20210804-1938-rev-edsr-p-chest-pelvis']
    checkpoint_dir = opt.checkpoint_dir
    for ch in range(len(checkpoints)):
        #find last checkpoint
        opt.checkpoint_dir = os.path.join(checkpoint_dir, checkpoints[ch])
        checkpoint_list = glob.glob(os.path.join(opt.checkpoint_dir, '*.pth'))
        checkpoint_list.sort()
        checkpoint = checkpoint_list[-1]
        print('loaded checkpoint path : {}'.format(checkpoint))

        #checkpoint load 
        model = networks_rev.Networks_rev(opt)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()
        
        
        with torch.no_grad():
            out_ge, feature_ge = model.denoiser(gtensor)
            out_mayo, feature_mayo = model.denoiser(mtensor)
        
        '''
        #just test... all channels
        ge_mean = torch.mean(feature_ge, dim=(1,2,3))
        ge_std = torch.std(feature_ge, dim=(1,2,3))
        mayo_mean = torch.mean(feature_mayo, dim=(1,2,3))
        mayo_std = torch.std(feature_mayo, dim=(1,2,3))
        distance = torch.norm(torch.mean(mayo_mean)-torch.mean(ge_mean), torch.mean(mayo_std)-torch.mean(ge_std))
        print('feature distance_{} : {}'.format(ch, distance))
        plt.clf()
        plt.plot(ge_mean, ge_std, '.r')
        plt.plot(mayo_mean, mayo_std, '.b')
        plt.xlabel('mean')
        plt.ylabel('std')
        plt.legend(['ge-feature', 'mayo-feature'])
        plt.savefig('domain_gap_feature_{}.png'.format(ch))
        '''

        '''
        #same channels
        ge_mean = torch.mean(feature_ge, dim=(2,3)) #[b,c]
        ge_std = torch.std(feature_ge, dim=(2,3)) #[b,c]
        mayo_mean = torch.mean(feature_mayo, dim=(2,3))
        mayo_std = torch.std(feature_mayo, dim=(2,3))

        distance = 0
        for c in range(ge_mean.size(1)):
            distance += torch.norm(torch.mean(mayo_mean[:,c])-torch.mean(ge_mean[:,c]), torch.mean(mayo_std[:,c])-torch.mean(ge_std[:,c]))
        print('feature distance_{} : {}'.format(ch, distance))
        '''


        
        #feature's mean, std
        ge_mean = torch.mean(feature_ge, dim=(2,3)) #[b,c]
        ge_std = torch.std(feature_ge, dim=(2,3)) #[b,c]
        ge_mean_max, idx = torch.max(ge_mean, dim=1) 
        # print('ge_idx : {}'.format(idx))
        ge_std_max = torch.zeros([batch])
        for j in range(batch):
            ge_std_max[j] = ge_std[j, idx[j]]


        mayo_mean = torch.mean(feature_mayo, dim=(2,3))
        mayo_std = torch.std(feature_mayo, dim=(2,3))
        mayo_mean_max, idx = torch.max(mayo_mean, dim=1) 
        # print('mayo_idx : {}'.format(idx))
        mayo_std_max = torch.zeros([batch])
        for j in range(batch):
            mayo_std_max[j] = mayo_std[j, idx[j]]
        distance = torch.norm(torch.mean(mayo_mean_max)-torch.mean(ge_mean_max), torch.mean(mayo_std_max)-torch.mean(ge_std_max))
        # mean_distance = 

        # print('ge_mean_max : {}, ge_std_max : {}'.format(ge_mean_max, ge_std_max))
        # print('mayo_mean_max : {}, mayo_std_max : {}'.format(mayo_mean_max, mayo_std_max))
        print('feature distance_{} : {}'.format(ch, distance))
        plt.clf()
        plt.xlim(0,3)
        plt.ylim(0, 1.5)
        plt.plot(ge_mean_max, ge_std_max, '.r')
        plt.plot(mayo_mean_max, mayo_std_max, '.b')
        plt.xlabel('mean')
        plt.ylabel('std')
        plt.legend(['ge-feature', 'mayo-feature'])
        plt.savefig('domain_gap_feature_{}.png'.format(ch))
        

        
        #image's mean, std
        ge_input_mean=torch.mean(gtensor, dim=(2,3))
        ge_input_std = torch.std(gtensor, dim=(2,3))

        ge_mean = torch.mean(out_ge, dim=(2,3)) #[b,1]
        ge_std = torch.std(out_ge, dim=(2,3)) #[b,1]

        mayo_input_mean = torch.mean(mtensor, dim=(2,3))
        mayo_input_std = torch.std(mtensor, dim=(2,3))

        mayo_mean = torch.mean(out_mayo, dim=(2,3))
        mayo_std = torch.std(out_mayo, dim=(2,3))

        movement_ge = torch.norm(torch.mean(ge_mean)-torch.mean(mayo_input_mean), torch.mean(ge_std)-torch.mean(mayo_input_std))
        movement_mayo = torch.norm(torch.mean(mayo_mean)-torch.mean(ge_input_mean), torch.mean(mayo_std)-torch.mean(ge_input_std))

        print('feature distance_{} : \nge movement : {} mayo movement : {}'.format(ch, movement_ge, movement_mayo))
        plt.clf()
        plt.xlim(-0.2,0.5)
        plt.ylim(0, 0.5)
        plt.plot(ge_input_mean, ge_input_std, '.r')
        plt.plot(ge_mean, ge_std, '*r')
        plt.plot(mayo_input_mean, mayo_input_std, '.b')
        plt.plot(mayo_mean, mayo_std, '*b')
        plt.xlabel('mean')
        plt.ylabel('std')
        plt.legend(['ge-noisy', 'ge-output', 'mayo-noisy', 'mayo-output'])
        plt.savefig('domain_gap_image_{}.png'.format(ch))
        
        





    
from models import set_model
from utils.loader import load_config, load_model
from utils.helper import set_gpu
from options import args

import torch
import torch.nn as nn

import glob
import random
import imageio

import numpy as np

from skimage.external.tifffile import imsave


def make_img_tensor(dir_path, batch_size):
    print(dir_path)
    img_lists = glob.glob(dir_path)
    random.shuffle(img_lists)
    img_tensor_batch = []
    for i in range(batch_size):
        img = imageio.imread(img_lists[i])
        img_tensor = torch.Tensor(img)
        h, w = img_tensor.shape
        sh = random.randint(0,h-80)
        sw = random.randint(0,w-80)
        img_tensor = img_tensor[sh:sh+80,sw:sw+80]
        img_tensor = img_tensor.view([1,1,80,80])
        img_tensor_batch.append(img_tensor)
    img_tensor_batch = torch.cat(img_tensor_batch, dim=0)

    return img_tensor_batch

def domain_classifier(img, net_g, net_dc, domain):
    img = img.to(opt.device)

    #concat
    # img = torch.cat((img,img),1)

    domain_out = net_dc(img)
    print(domain_out)
    print('average: ',sum(domain_out)/len(domain_out))

    '''
    domain_out_list = domain_out.tolist()

    for i in range(len(domain_out)):
        img_i = img[i]
        img_i = torch.squeeze(img_i)
        img_list = img_i.cpu().numpy()
        # print(img_list)

        imsave('../../../eunji/dc_result/src_star/'+str(domain_out_list[i])+'.tiff',img_list.astype('float32'))
    '''

    #target
    if domain == True:
        correct = torch.sum(torch.round(domain_out)==1)
    else:
        correct = torch.sum(torch.round(domain_out)==0)

    #source
    if domain == False:
        correct = torch.sum(torch.round(domain_out)==0)
    else:
        correct = torch.sum(torch.round(domain_out)==1)


    return correct



if __name__ == "__main__":

    #load options
    opt = args
    opt = load_config(opt)
    print(opt)

    #load models
    net = set_model(opt)
    _, net, _ = load_model(opt, net)
    net_g = net.denoiser
    net_dc = net.domain_discriminator

    #setting gpu
    opt = set_gpu(opt)
    opt.device = 'cuda'

    if opt.use_cuda:
        net_g = net_g.to(opt.device)
        net_dc = net_dc.to(opt.device)

    if opt.multi_gpu:
        net_g = nn.DataParallel(net_g)
        net_dc = nn.DataParallel(net_dc)

    #noisy target (trg)
    trg_1mm_quarter = '../../data/denoising/test/mayo/quarter_1mm/L067/*.tiff'
    trg_3mm_quarter = '../../data/denoising/test/mayo/quarter_3mm/L067/*.tiff'

    #mayo supervised (trg') - 146
    trg_mayo_supervised_1mm = '../../data/denoising/test_result_DA/mayo-20210429-0531-base-edsr-testset-mayo-1mm/*.tiff'
    trg_mayo_supervised_3mm = '../../data/denoising/test_result_DA/mayo-20210429-0531-base-edsr-testset-mayo-3mm/*.tiff'

    #phantom supervised (trg') - 147
    trg_ph_supervised_1mm = '../../data/denoising/test_result_DA/ge-20210429-0840-base-edsr-chest-pelvis-testset-mayo-1mm/*.tiff'
    trg_ph_supervised_3mm = '../../data/denoising/test_result_DA/ge-20210429-0840-base-edsr-chest-pelvis-testset-mayo-3mm/*.tiff'

    #clean target (trg*)
    trg_1mm_full = '../../data/denoising/test/mayo/full_1mm/L067/*.tiff'
    trg_3mm_full = '../../data/denoising/test/mayo/full_3mm/L067/*.tiff'
    
    #quarter+poisson noise
    trg_quarter_noise_1mm = '../../data/denoising/train/mayo/quarter_1mm_noise/L096_poisson_400_1/*.tiff'
    trg_quarter_noise_3mm = '../../data/denoising/train/mayo/quarter_3mm_noise/L096_poisson_1400_1/*.tiff'

    #quarter+poisson noise + mayo supervised
    trg_quarter_noise_mayo_sup_1mm = '../../data/denoising/test_result_DA/mayo-20210429-0531-base-edsr-testset-mayo-1mm/*.tiff'
    trg_quarter_noise_mayo_sup_3mm = '../../data/denoising/test_result_DA/mayo-20210429-0531-base-edsr-testset-mayo-3mm/*.tiff'

    #quarter+poisson noise + phantom supervised
    trg_quarter_noise_ph_sup_1mm = '../../data/denoising/test_result_DA/ge-20210429-0840-base-edsr-chest-pelvis-testset-mayo-1mm/*.tiff'
    trg_quarter_noise_ph_sup_3mm = '../../data/denoising/test_result_DA/ge-20210429-0840-base-edsr-chest-pelvis-testset-mayo-3mm/*.tiff'

    #quarter+poisson noise + mayo-20210420-1841-base-edsr-p
    trg_quarter_noise_1841_1mm = '../../data/denoising/test_result_DA/mayo-20210420-1841-base-edsr-p-testset-mayo-1mm/*.tiff'
    trg_quarter_noise_1841_3mm = '../../data/denoising/test_result_DA/mayo-20210420-1841-base-edsr-p-testset-mayo-3mm/*.tiff'

    #quarter+poisson noise + mayo-syn-20210429-1236-base-edsr-p
    trg_quarter_noise_1236_1mm = '../../data/denoising/test_result_DA/mayo-syn-20210429-1236-base-edsr-p-testset-mayo-1mm/*.tiff'
    trg_quarter_noise_1236_3mm = '../../data/denoising/test_result_DA/mayo-syn-20210429-1236-base-edsr-p-testset-mayo-3mm/*.tiff'

    #quarter+poisson noise + mayo-syn-20210503-1603-base-edsr-p
    trg_quarter_noise_1603_1mm = '../../data/denoising/test_result_DA/mayo-syn-20210503-1603-base-edsr-p-testset-mayo-1mm/*.tiff'
    trg_quarter_noise_1603_3mm = '../../data/denoising/test_result_DA/mayo-syn-20210503-1603-base-edsr-p-testset-mayo-3mm/*.tiff'


    #noisy src (src)
    src = '../../data/denoising/train/phantom/ge/chest/level5_005/*.tiff'

    #mayo supervised (src') - 146
    src_mayo_supervised = '../../data/denoising/test_result_DA/mayo-20210429-0531-base-edsr-testset-ge-chest/*.tiff'

    #phantom supervised (src') - 147
    src_ph_supervised = "../../data/denoising/test_result_DA/ge-20210429-0840-base-edsr-chest-pelvis-testset-ge-chest/*.tiff"

    #clean src (src*)
    src_star = '../../data/denoising/train/phantom/ge/chest/level3_025/*.tiff'

    domain = True # target
    # domain = False # source

    for dir_path in [trg_quarter_noise_1841_1mm,trg_quarter_noise_1841_3mm,trg_quarter_noise_1236_1mm,trg_quarter_noise_1236_3mm,trg_quarter_noise_1603_1mm,trg_quarter_noise_1603_3mm]:
        img_tensor_batch = make_img_tensor(dir_path, 80)
        score = domain_classifier(img_tensor_batch, net_g, net_dc, domain)
        print('{} : {} --> score : {:3f}[{}/{}]'.format(dir_path, domain, float(score/80), score, 80))
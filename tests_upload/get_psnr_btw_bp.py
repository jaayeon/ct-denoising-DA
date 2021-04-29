import numpy as np
import imageio
from skimage.external.tifffile import imsave 
from skimage.external.tifffile import imread
import glob, os, random
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get PSNR between back projection data')
    parser.add_argument('--delete_percentage', type=float, default=0.3, help='delete percent for each direction')
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--p_val', type=int, default=150000)
    opt = parser.parse_args()

    bp3mm =  'D:/data/denoising/train/mayo/back_projection/{}mm/quarter_{}mm_L096_bp.tif'.format(opt.thickness, opt.thickness)
    nbp3mm = 'D:/data/denoising/train/mayo/back_projection/{}mm_noise/quarter_{}mm_L096_nbp_{}.tif'.format(opt.thickness, opt.thickness, opt.p_val)

    if opt.thickness == 3:
        real_std = 0.015521074319258333 #for all train set
        # real_std = 0.015963714195614948
    else : 
        real_std = 0.028929880168288947 #for all train set
        # real_std = 0.02814156667196325
    bp3mm = imread(bp3mm)
    nbp3mm = imread(nbp3mm)

    # norm_bp3mm = (bp3mm-np.min(bp3mm))/(np.max(bp3mm)-np.min(bp3mm))
    # norm_nbp3mm = (nbp3mm-np.min(nbp3mm))/(np.max(nbp3mm)-np.min(nbp3mm))

    imgnum = bp3mm.shape[0]
    print(imgnum)
    start_num = int(imgnum*opt.delete_percentage)
    nums = imgnum-2*start_num
    print(nums)

    avg_std = 0.0
    for i in range(start_num, start_num+nums):
        bp3mm_2d = bp3mm[i,:,:]
        nbp3mm_2d = nbp3mm[i,:,:]
        norm_bp3mm_2d = (bp3mm_2d-np.min(bp3mm_2d))/(np.max(bp3mm_2d)-np.min(bp3mm_2d))
        norm_nbp3mm_2d = (nbp3mm_2d-np.min(nbp3mm_2d))/(np.max(nbp3mm_2d)-np.min(nbp3mm_2d))
        noise = norm_nbp3mm_2d - norm_bp3mm_2d
        avg_std += np.std(noise)
    avg_std = avg_std/nums
    print('avg_std : {}\n scale : {}'.format(avg_std, avg_std/real_std))

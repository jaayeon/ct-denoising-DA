import glob, os
import argparse
import torch
import random
import numpy as np
from skimage.external.tifffile import imsave 
from skimage.external.tifffile import imread


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='normalize')
    parser.add_argument('--data', type=str, default='source', choices=['source', 'target'])
    parser.add_argument('--mm', type=int, default=1, choices=[1,3], help='you can specify the mm (1,3 mm)')
    opt = parser.parse_args()

    phantom_low = glob.glob('../../data/denoising/train/phantom/ge/chest/level5_005_crop/*.tiff')

    mayo_3q = glob.glob('../../data/denoising/train/mayo/quarter_3mm/L096/*.tiff')
    mayo_1q = glob.glob('../../data/denoising/train/mayo/quarter_1mm/L096/*.tiff')

    data = opt.data
    mayo_mm = opt.mm

    if data=='source':
        src_img_lists = phantom_low

        for idx, imgpath in enumerate(src_img_lists):
            print(imgpath)
            basename = os.path.basename(imgpath)

            src = imread(src_img_lists[idx])

            src_norm = (src-0.250) / 0.327

            save_path = os.path.join('../../data/denoising/train/phantom/ge/chest/level5_005_crop_normalize')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            save_path = os.path.join(save_path,'L096')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            imsave(os.path.join(save_path,basename),src_norm)


    if data=='target':
        if mayo_mm == 1:
            trg_img_lists = mayo_1q
        else:
            trg_img_lists = mayo_3q

        for i,imgpath in enumerate(trg_img_lists):
            print(imgpath)
            basename = os.path.basename(imgpath)

            trg = imread(trg_img_lists[i])

            trg_norm = (trg-0.377) / 0.338

            save_path = os.path.join('../../data/denoising/train/mayo','quarter_{}mm_normalize'.format(mayo_mm))
            if not os.path.exists(save_path):
                    os.mkdir(save_path)

            imsave(os.path.join(save_path,basename),trg_norm)
                

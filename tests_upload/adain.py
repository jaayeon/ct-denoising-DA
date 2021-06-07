import glob, os
import argparse
import torch
import random
import numpy as np
from skimage.external.tifffile import imsave 
from skimage.external.tifffile import imread


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AdaIN')
    parser.add_argument('--style', type=str, default='source', choices=['source', 'target'])
    parser.add_argument('--mm', type=int, default=0, choices=[0,1,3], help='in case of using target image as style image, you can specify the mm (0 for both 1,3 mm)')
    opt = parser.parse_args()

    phantom_low = glob.glob('../../data/denoising/train/phantom/ge/chest/level5_005_crop/*.tiff')

    mayo_3q = glob.glob('../../data/denoising/train/mayo/quarter_3mm/L096/*.tiff')
    mayo_1q = glob.glob('../../data/denoising/train/mayo/quarter_1mm/L096/*.tiff')
    mayo_1_3 = [mayo_1q,mayo_3q]

    style = opt.style
    mayo_mm = opt.mm

    # src(phantom) style to trg(mayo)
    if style=='source':
        src_img_lists = phantom_low

        for idx,mayo in enumerate(mayo_1_3):
            mayo_q = mayo_1_3[idx]
            mm =1 if idx==0 else 3

            for i,imgpath in enumerate(mayo,1):
                basename = os.path.basename(imgpath)

                random.shuffle(src_img_lists)

                trg = imread(imgpath)
                src = imread(src_img_lists[0])

                src_style_trg = np.std(src)*((trg-np.mean(trg))/np.std(trg)) + np.mean(src)

                save_path = os.path.join('../../data/denoising/train/mayo/src_style_trg_{}mm'.format(mm))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                save_path = os.path.join('../../data/denoising/train/mayo/src_style_trg_{}mm'.format(mm),'L096')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                imsave(os.path.join(save_path,basename),src_style_trg)


    if style=='target':
        if mayo_mm == 0:
            trg_img_lists = mayo_1q+mayo_3q
            print(len(trg_img_lists))
        elif mayo_mm == 1:
            trg_img_lists = mayo_1q
            print(len(trg_img_lists))
        else:
            trg_img_lists = mayo_3q
            print(len(trg_img_lists))
    
        for i,imgpath in enumerate(phantom_low):
            basename = os.path.basename(imgpath)

            random.shuffle(trg_img_lists)

            trg = imread(trg_img_lists[0])
            src = imread(imgpath)

            trg_style_src = np.std(trg)*((src-np.mean(src))/np.std(src)) + np.mean(trg)

            save_path = os.path.join('../../data/denoising/train/phantom/ge','chest_mayo_{}mm_style'.format(mayo_mm))
            if not os.path.exists(save_path):
                    os.mkdir(save_path)

            imsave(os.path.join(save_path,basename),trg_style_src)
                

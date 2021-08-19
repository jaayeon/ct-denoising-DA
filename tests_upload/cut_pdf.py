# from posix import POSIX_FADV_NOREUSE
import numpy as np
import imageio
import glob, os
import torch
import argparse

if __name__ == "__main__":

    pre_min = -1000.0
    pre_max = 400.0

    new_min = -160.0
    new_max = 240.0

    test_dirs = ['ge-20210819-0418-rev-edsr-p-chest-pelvis-testset-mayo-1mm',
                'ge-20210819-0413-rev-edsr-p-chest-pelvis-testset-mayo-1mm',
                'ge-20210819-0414-rev-edsr-p-chest-pelvis-testset-mayo-1mm',
                'ge-20210819-0425-rev-edsr-p-chest-pelvis-testset-mayo-1mm']
    # test_dirs = ['1mm_nlm_eunji',
    #             '3mm_nlm_eunji']
    test_path = '../../data/denoising/test_result_DA'
    save_path = '../../data/denoising/test_pdf'

    for test_dir in test_dirs:
        test_dir_path = os.path.join(test_path, test_dir, '*.tiff')
        test_imgs = glob.glob(test_dir_path)
        test_imgs.sort()

        for img_pth in test_imgs:
            print("img_name: ", os.path.basename(img_pth))
            img = imageio.imread(img_pth)
            dc_img = img * (pre_max-pre_min) + pre_min

            dc_img[dc_img<-160] = -160.0
            dc_img[dc_img>240] = 240.0

            # normalize
            dc_img = (dc_img-new_min)/(new_max-new_min)
            
            save_dir_path = os.path.join(save_path, test_dir)
            os.makedirs(save_dir_path, exist_ok=True)
            imageio.imwrite(os.path.join(save_dir_path, os.path.basename(img_pth)), dc_img)

            
        #     0-1 normalization

        # 저장
        # psnr구해주고
        # psnr
        # [1,1,h,w] tensor 지정하고~ 뽑아줘~

        # 노이즈 psnr..
        # nlm psnr : 


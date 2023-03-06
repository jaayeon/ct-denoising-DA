import sys, os, glob
import time

import numpy as np

import torch
from options import args
import torch.nn as nn
from utils.metric import calc_metrics, forward_ensemble
from options import args
from models import set_model
from utils.helper import set_gpu, set_test_dir
import imageio

from skimage import img_as_float, color, io
from skimage.restoration import denoise_nl_means, estimate_sigma


if __name__ == '__main__':

    img_dir = r'../../data/denoising/test/piglet/Oten' # quarter: img
    gt_img_dir = r'../../data/denoising/test/piglet/full' # full: gt_img
    device='cpu' #'cuda' for gpu

    img_list = glob.glob(os.path.join(img_dir, '**', '*.tiff'))
    gt_img_list = glob.glob(os.path.join(gt_img_dir, '**', '*.tiff'))
    test_result_dir = '../../data/denoising/test_result_DA/piglet_nlm'

    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    test_concat_dir = os.path.join(test_result_dir, 'concat_img')
    if not os.path.exists(test_concat_dir):
        os.makedirs(test_concat_dir)

    img_list.sort()
    gt_img_list.sort()
    print(len(img_list))
    print(len(gt_img_list))
    num_total_img = len(img_list)

    num = 0
    avg_loss = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0
    noise_avg_loss = 0.0
    noise_avg_psnr = 0.0
    noise_avg_ssim = 0.0
    for img_idx, path in enumerate(zip(img_list,gt_img_list),1):
        img_path = path[0]
        print('img_path : ', img_path)
        gt_img_path = path[1]
        print('gt_img_path ; ', gt_img_path)
        start_time = time.time()
        img_path = os.path.abspath(img_path)
        img_name = 'out_'+os.path.basename(img_path)
        concat_name = 'concat_'+os.path.basename(img_path)
        num_test_img = int(num_total_img)

        print("[{}/{}] processing {}".format(num+1, num_test_img, os.path.abspath(img_path)))

        input_img = imageio.imread(img_path)
        gt_img = imageio.imread(gt_img_path)

        # estimate the noise standard deviation from the noisy image
        input_sigma_est = np.mean(estimate_sigma(input_img, multichannel=False))

        patch_kw = dict(patch_size=5, patch_distance=13, multichannel=False)  # 5x5 patches, 13x13 search area 

        # fast nlm
        output_img = denoise_nl_means(input_img, h=0.8*(10^15)*input_sigma_est, fast_mode=True, **patch_kw)

        #numpy2tensor (h,w)->(b,c,h,w)
        input_img_tensor = torch.from_numpy(input_img.reshape(1,1,input_img.shape[0],input_img.shape[1])).to(device)
        output_img_tensor = torch.from_numpy(output_img.reshape(1,1,output_img.shape[0],output_img.shape[1])).to(device)
        gt_img_tensor = torch.from_numpy(gt_img.reshape(1,1,gt_img.shape[0],gt_img.shape[1])).to(device)

        out_img_path = os.path.join(test_result_dir, img_name)
        concat_img_path = os.path.join(test_concat_dir, concat_name)

        noise_loss, noise_psnr, noise_ssim, batch_loss, batch_psnr, batch_ssim = calc_metrics(input_img_tensor.float(), output_img_tensor.float(), gt_img_tensor.float())
        # print('nl : {:.8f}, np : {:.8f}, ol : {:.8f}, op : {:.8f}'.format(noise_loss, noise_psnr, batch_loss, batch_psnr))
        
        concat_img = np.concatenate((input_img, output_img, gt_img), axis=1)

        num += 1
        avg_loss += batch_loss
        avg_psnr += batch_psnr
        avg_ssim += batch_ssim
        noise_avg_loss += noise_loss
        noise_avg_psnr += noise_psnr
        noise_avg_ssim += noise_ssim
        

        print("** Test {:.3f}s => Image({}/{}): Noise Loss: {:.8f}, Noise PSNR: {:.8f}, Noise SSIM: {:.8f}, Loss: {:.8f}, PSNR: {:.8f}, SSIM: {:.8f}".format(
            time.time() - start_time, img_idx, num_total_img, noise_loss.item(), noise_psnr.item(), noise_ssim.item(), batch_loss.item(), batch_psnr.item(), batch_ssim.item()
        ))
        print()

        imageio.imwrite(concat_img_path, concat_img)
        imageio.imwrite(out_img_path, output_img)


    print(" #{:d} Test Average Noise Loss: {:.8f}, Average Noise PSNR: {:.8f}, Average Noise SSIM: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}, Average SSIM: {:.8f}".format(
        num, noise_avg_loss / num, noise_avg_psnr / num, noise_avg_ssim / num, avg_loss / num, avg_psnr / num, avg_ssim / num
    ))

    print("---Time: %.4fs\n" % (time.time() - start_time))
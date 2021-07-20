# from posix import POSIX_FADV_NOREUSE
import argparse
import numpy as np
import imageio
import glob, os
import torch
from utils.metric import calc_metrics

def select_test_dir():
    img_dir = '../../data/denoising/test_pdf'
    dirs = os.listdir(img_dir)
    dirs = sorted(dirs)

    for i, d in enumerate(dirs, 0):
        print("(%d) %s" % (i, d))
    d_idx = input("Select directory that you want to load: ")

    path_opt = dirs[int(d_idx)]

    img_dir = os.path.abspath(os.path.join(img_dir, dirs[int(d_idx)]))
    print("img_dir is: {}".format(img_dir))

    return img_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate psnr/ssim between two images')
    parser.add_argument('--thickness', type=str, default=None, choices=['3', '1']) # mayo dataset thickness option
    # parser.add_argument()
    args = parser.parse_args()

    test_dir = select_test_dir() # Select img dir to calculate psnr
    test_dir_path = os.path.join(test_dir, '*.tiff')
    test_imgs = glob.glob(test_dir_path)
    test_imgs.sort()

    gt_dir = '../../data/denoising/test_pdf/full_{}mm'.format(args.thickness)
    noisy_dir = '../../data/denoising/test_pdf/quarter_{}mm'.format(args.thickness)

    gt_imgs = glob.glob(os.path.join(gt_dir, '*.tiff'))
    noisy_imgs = glob.glob(os.path.join(noisy_dir, '*.tiff'))

    gt_imgs.sort()
    noisy_imgs.sort()

    num_imgs = len(gt_imgs)
    avg_loss = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0
    noise_avg_loss = 0.0
    noise_avg_psnr = 0.0
    noise_avg_ssim = 0.0

    for ind, path in enumerate(zip(noisy_imgs, test_imgs, gt_imgs), 1):
        noisy_path = path[0]
        test_path = path[1]
        gt_path = path[2]
        print("test_img_path: ", test_path)
        print("gt_img_path: ", gt_path)

        print("[{}/{}] processing {}".format(ind, num_imgs, os.path.abspath(test_path)))

        noisy_img = imageio.imread(noisy_path)
        test_img = imageio.imread(test_path)
        gt_img = imageio.imread(gt_path)

        test_img_shape = test_img.shape

        #only for gray scale img
        noisy_img_tensor = torch.from_numpy(noisy_img.reshape(1,1,noisy_img.shape[0], noisy_img.shape[1]))
        test_img_tensor = torch.from_numpy(test_img.reshape(1,1,test_img.shape[0], test_img.shape[1]))
        gt_img_tensor = torch.from_numpy(gt_img.reshape(1,1,gt_img.shape[0], gt_img.shape[1]))

        noise_loss, noise_psnr, noise_ssim, batch_loss, batch_psnr, batch_ssim = calc_metrics(noisy_img_tensor, test_img_tensor, gt_img_tensor)
        
        avg_loss += batch_loss
        avg_psnr += batch_psnr
        avg_ssim += batch_ssim
        noise_avg_loss += noise_loss
        noise_avg_psnr += noise_psnr
        noise_avg_ssim += noise_ssim

        print("** Image({}/{}): Noise Loss: {:.8f}, Noise PSNR: {:.8f}, Noise SSIM: {:.8f}, Loss: {:.8f}, PSNR: {:.8f}, SSIM: {:.8f}".format(
                ind, num_imgs, noise_loss.item(), noise_psnr.item(), noise_ssim.item(), batch_loss.item(), batch_psnr.item(), batch_ssim.item()
            ))
    
    print(" #{:d} Test Average Noise Loss: {:.8f}, Average Noise PSNR: {:.8f}, Average Noise SSIM: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}, Average SSIM: {:.8f}".format(
        num_imgs, noise_avg_loss / num_imgs, noise_avg_psnr / num_imgs, noise_avg_ssim / num_imgs, avg_loss / num_imgs, avg_psnr / num_imgs, avg_ssim / num_imgs
    ))
        #     0-1 normalization

        # 저장
        # psnr구해주고
        # psnr
        # [1,1,h,w] tensor 지정하고~ 뽑아줘~

        # 노이즈 psnr..
        # nlm psnr : 


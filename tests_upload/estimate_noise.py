#https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter
import numpy as np 
import imageio
import glob, os, math, random
import argparse
import torch.nn as nn
import torch
from skimage.restoration import denoise_nl_means, estimate_sigma

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='estimate Noise')
    parser.add_argument('--num', default=100, type=int, help='image num for each patient')

    opt = parser.parse_args()

    mayo_3q = glob.glob('../../data/denoising/train/mayo/quarter_3mm/L096/*')
    mayo_3f = glob.glob('../../data/denoising/train/mayo/full_3mm/L096/*')
    mayo_1q = glob.glob('../../data/denoising/train/mayo/quarter_1mm/L096/*')
    mayo_1f = glob.glob('../../data/denoising/train/mayo/full_1mm/L096/*')

    print('#mayo_3mm:{}, mayo_1mm:{}'.format(len(mayo_3q), len(mayo_1q)))
    rate = [0.0, 0.0]
    std = [0.0, 0.0]
    num = [0, 0]
    #1mm,3mm
    for p, mayo_q, mayo_f in zip([0,1], [mayo_1q, mayo_3q], [mayo_1f, mayo_3f]):
        imgs = len(mayo_q)
        idx = np.arange(0, len(mayo_q))
        random.shuffle(idx)
        num[p] = min(opt.num, len(mayo_q))
        for i in range(num[p]):
            basename = os.path.basename(mayo_q[idx[i]])
            qimg = imageio.imread(mayo_q[idx[i]])
            fimg = imageio.imread(mayo_f[idx[i]])

            qimg = np.array(qimg, dtype='f')
            fimg = np.array(fimg, dtype='f')

            noise = fimg-qimg
            noise_std = np.std(noise)
            noise_mean = np.mean(noise)

            sigma_est = np.mean(estimate_sigma(qimg, multichannel=False))
            print('sigma_est : {:.6f}, real_std : {:.6f} ..{:.2f}'.format(sigma_est, noise_std, noise_std/sigma_est))
            rate[p] += noise_std/sigma_est
            std[p] += noise_std

    print('[*Average*] \n1mm - rate:{}, std:{} \n3mm - rate:{}, std:{}'.format(rate[0]/num[0], std[0]/num[0], rate[1]/num[1], std[1]/num[1]))


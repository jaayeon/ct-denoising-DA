#https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter
import numpy as np 
import imageio
import glob, os, math, random
import argparse
import torch.nn as nn
import torch
import cv2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add Noise')
    parser.add_argument('--noise', type=str, default='poisson', choices=['poisson', 'gaussian', 'poisson-gaussian', 'bilateral'])
    parser.add_argument('--p_lam', type=float, nargs='+', default=[400, 1400], help='poisson parameter, each for 1,3mm')
    parser.add_argument('--g_std', type=float, nargs='+', default=[0.032, 0.016], help='gaussian parameter, each for 1,3mm')
    parser.add_argument('--b_dcs', type=float, nargs='+', default=[10, 3, 3]
                        , help='bilateral filter parameters.The diameter of each pixel neighborhood, Filter sigma in color space, Filter sigma in the coordinate space')
    parser.add_argument('--scale', type=float, default=1, help='scaling noise for gaussian, poisson')
    parser.add_argument('--num', type=int, default=20)
    opt = parser.parse_args()

    mayo_3q = glob.glob('../../data/denoising/train/mayo/quarter_3mm/L096/*')
    mayo_3f = glob.glob('../../data/denoising/train/mayo/full_3mm/L096/*')
    mayo_1q = glob.glob('../../data/denoising/train/mayo/quarter_1mm/L096/*')
    mayo_1f = glob.glob('../../data/denoising/train/mayo/full_1mm/L096/*')

    noise_mode = opt.noise
    if noise_mode=='bilateral':
        params = opt.b_dcs
        n_mayo_1q = '../../data/denoising/train/mayo/quarter_1mm_noise/L096_{}_{}_{}_{}'.format(noise_mode, params[0], params[1], params[2])
        n_mayo_3q = '../../data/denoising/train/mayo/quarter_3mm_noise/L096_{}_{}_{}_{}'.format(noise_mode, params[0], params[1], params[2])
    else : 
        params = opt.p_lam if opt.noise=='poisson' else opt.g_std
        n_mayo_1q = '../../data/denoising/train/mayo/quarter_1mm_noise/L096_{}_{}_{}'.format(noise_mode, params[0], opt.scale)
        n_mayo_3q = '../../data/denoising/train/mayo/quarter_3mm_noise/L096_{}_{}_{}'.format(noise_mode, params[1], opt.scale)
    if not os.path.exists(n_mayo_1q):
        os.mkdir(n_mayo_1q)
    if not os.path.exists(n_mayo_3q):
        os.mkdir(n_mayo_3q)
    mse = nn.MSELoss()
    #1mm,3mm
    for p, mayo_q, mayo_f, save_dir in zip([0,1], [mayo_1q, mayo_3q], [mayo_1f, mayo_3f], [n_mayo_1q, n_mayo_3q]):
        qf_avg = 0.0
        nf_avg = 0.0
        qn_avg = 0.0
        for i in range(opt.num):
            basename = os.path.basename(mayo_q[i*10])
            qimg = imageio.imread(mayo_q[i*10])
            fimg = imageio.imread(mayo_f[i*10])

            qimg = np.array(qimg, dtype='f')
            fimg = np.array(fimg, dtype='f')

            if noise_mode == 'poisson':
                imgname = '{}_{}_{}'.format(noise_mode, params[p], basename)
                # Generating noise for each unique value in image.
                # vals = len(np.unique(qimg))
                # vals = 2**np.ceil(np.log2(vals))
                # nimg = np.random.poisson(qimg*vals)/float(vals)
                nimg = np.random.poisson(qimg*opt.p_lam[p])/float(opt.p_lam[p])
                nimg = qimg + opt.scale*(nimg-qimg)

            elif noise_mode == 'gaussian':
                imgname = '{}_{}_{}'.format(noise_mode, params[p], basename)
                noise = np.random.normal(loc=0, scale=opt.g_std[p], size=qimg.shape).astype(float)
                nimg = qimg + opt.scale*noise
            elif noise_mode == 'poisson-gaussian':
                imgname = '{}_{}_{}'.format(noise_mode, params[p], basename)
                noise = 0.5*np.random.poisson(lam=opt.p_lam[p], size=qimg.shape).astype(float) + 0.5*np.random.normal(loc=0, scale=opt.g_std[p], size=qimg.shape).astype(float)
                nimg = qimg + noise
            elif noise_mode == 'bilateral':
                imgname = '{}_{}_{}_{}_{}'.format(noise_mode, params[0], params[1], params[2], basename)
                clean = cv2.bilateralFilter(qimg, int(params[0]), params[1], params[2])
                # print('clean ;; max : {}, min : {}'.format(np.max(clean), np.min(clean)))
                noise = qimg-clean
                # print('type : qimg : {}, clean : {}, noise : {}'.format(qimg.dtype, clean.dtype, noise.dtype))
                print('noise ;; max : {}, min : {}'.format(np.max(noise), np.min(noise)))
                nimg = qimg + noise*10
                
            nimg[nimg>1.0] = 1.0
            nimg[nimg<0.0] = 0.0

            qimg = torch.Tensor(qimg)
            fimg = torch.Tensor(fimg)
            nimg = torch.Tensor(nimg)

            qf_psnr = 10*math.log10(1/mse(qimg,fimg))
            nf_psnr = 10*math.log10(1/mse(nimg,fimg))
            qn_psnr = 10*math.log10(1/mse(qimg,nimg))
            qf_avg += qf_psnr
            nf_avg += nf_psnr
            qn_avg += qn_psnr
            
            print('qf_psnr : {:.4f}, nf_psnr : {:.4f}, qn_psnr : {:.4f}'.format(qf_psnr, nf_psnr, qn_psnr))
            
            imageio.imwrite(os.path.join(save_dir, imgname), nimg.numpy())
        qf_avg = qf_avg/opt.num
        nf_avg = nf_avg/opt.num
        qn_avg = qn_avg/opt.num
        print('[*Average*] qf_avg : {:.4f}, nf_avg : {:.4f}, qn_avg : {:.4f}\n'.format(qf_avg, nf_avg, qn_avg))

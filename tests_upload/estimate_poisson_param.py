import numpy as np 
import imageio
import glob, os, math, random
import argparse
import torch.nn as nn
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='estimate poisson param')
    parser.add_argument('--lam', type=float, default=100, help='poisson parameter')
    parser.add_argument('--num', type=int, default=1000)

    opt = parser.parse_args()

    dataset_n = glob.glob('../../data/denoising/train/piglet/Oten/*/*')
    dataset_c = glob.glob('../../data/denoising/train/piglet/full/*/*')

    dataset_n.sort()
    dataset_c.sort()

    write_pth = '../../data/denoising/train/piglet/Oten_noisy/'
    if not os.path.exists(write_pth):
        os.mkdir(write_pth)

    mse = nn.MSELoss()

    std = 0
    pstd = 0
    ratio = 0
    idxs = np.arange(0, len(dataset_n))
    random.shuffle(idxs)
    for i in idxs[:opt.num]:
        basename = os.path.basename(dataset_n[i])
        nimg = imageio.imread(dataset_n[i])
        cimg = imageio.imread(dataset_c[i])

        nimg = np.array(nimg, dtype='f')
        cimg = np.array(cimg, dtype='f')
        noise = nimg-cimg
        
        nnimg = np.random.poisson(opt.lam*nimg)/float(opt.lam)

        pnoise = nnimg-nimg

        noise_std = np.std(noise)
        pnoise_std = np.std(pnoise)

        scale = pnoise_std/noise_std

        std += noise_std
        pstd += pnoise_std
        ratio += scale
        print('noise std : {}, pnoise std : {}, ratio : {}'.format(noise_std, pnoise_std, scale))

    print('**[Average] p-lam : {} // noise std : {}, poisson noise std : {}, ratio : {}'.format(opt.lam, std/opt.num, pstd/opt.num, ratio/opt.num))



'''
[piglet]
p-lam : 100 // noise std : 0.018842634013853967, poisson noise std : 0.046875848440192526, ratio : 2.5273604591671717
p-lam : 160.0 // noise std : 0.018889585371594877, poisson noise std : 0.03710957360265927, ratio : 2.0002758277659725
p-lam : 285.0 // noise std : 0.018453767627943308, poisson noise std : 0.027327006043186285, ratio : 1.510957890582079
p-lam : 620.0 // noise std : 0.018800066684372722, poisson noise std : 0.018660798875140053, ratio : 1.0084789689986988
p-lam : 2500.0 // noise std : 0.018735420752316715, poisson noise std : 0.009338896369747591, ratio : 0.5081447620950298
'''
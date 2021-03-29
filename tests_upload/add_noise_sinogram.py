#https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter
import numpy as np 
import imageio
from skimage.external.tifffile import imsave 
from skimage.external.tifffile import imread
import glob, os, random
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add Noise in Sinogram')
    parser.add_argument('--noise', type=str, default='poisson', choices=['poisson', 'gaussian'])
    parser.add_argument('--p_lam', type=float, nargs='+', default=[400, 1400], help='poisson parameter, each for 1,3mm')
    parser.add_argument('--g_std', type=float, nargs='+', default=[0.032, 0.016], help='gaussian parameter, each for 1,3mm')
    opt = parser.parse_args()

    mayo_3q = glob.glob('../../data/denoising/train/mayo/forward_projection/3mm/quarter_3mm_*_fp.tif')
    # mayo_3f = glob.glob('../../data/denoising/train/mayo/forward_projection/3mm/full_3mm_*_fp.tif')
    mayo_1q = glob.glob('../../data/denoising/train/mayo/forward_projection/1mm/quarter_1mm_*_fp.tif')
    # mayo_1f = glob.glob('../../data/denoising/train/mayo/forward_projection/1mm/full_1mm_*_fp.tif')

    n_mayo_3q = '../../data/denoising/train/mayo/forward_projection/3mm_noise'
    n_mayo_1q = '../../data/denoising/train/mayo/forward_projection/1mm_noise'

    if not os.path.exists(n_mayo_3q):
        os.mkdir(n_mayo_3q)
    if not os.path.exists(n_mayo_1q):
        os.mkdir(n_mayo_1q)

    #1mm
    for i,imgpath in enumerate(mayo_1q,1):
        print('[{}/{}] start {}'.format(i,len(mayo_1q),imgpath))
        basename = os.path.basename(imgpath)
        nbasename = '_'.join(basename.split('_')[:-1]) + '_nfp'
        nimgpath = os.path.join(n_mayo_1q,nbasename)
        if not os.path.exists(nimgpath):
            os.mkdir(nimgpath)
        
        img = imread(imgpath)
        for j in range(720):
            arr = img[j,:,:]
            vals = len(np.unique(arr))
            vals = 2**np.ceil(np.log2(vals))
            narr = np.random.poisson(arr*vals)/float(vals)
            imsave(os.path.join(nimgpath, '{}_{}.tif'.format(nbasename, str(j).zfill(3))), narr.astype('float32'))
            if j%100 == 0:
                print('...{}/{} progressed'.format(j,720))
        print('[{}/{}] save {}'.format(i,len(mayo_1q),nimgpath))

    #3mm
    for i,imgpath in enumerate(mayo_3q,1):
        print('[{}/{}] start {}'.format(i,len(mayo_3q),imgpath))
        basename = os.path.basename(imgpath)
        nbasename = '_'.join(basename.split('_')[:-1]) + '_nfp'
        nimgpath = os.path.join(n_mayo_3q,nbasename)
        if not os.path.exists(nimgpath):
            os.mkdir(nimgpath)
        
        img = imread(imgpath)
        for j in range(720):
            arr = img[j,:,:]
            vals = len(np.unique(arr))
            vals = 2**np.ceil(np.log2(vals))
            narr = np.random.poisson(arr*vals)/float(vals)
            imsave(os.path.join(nimgpath, '{}_{}.tif'.format(nbasename, str(j).zfill(3))), narr.astype('float32'))
            if j%100 == 0:
                print('...{}/{} progressed'.format(j,720))
        print('[{}/{}] save {}'.format(i,len(mayo_3q),nimgpath))

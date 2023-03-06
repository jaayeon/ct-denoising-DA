#https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter
import numpy as np 
import imageio
from skimage.external.tifffile import imsave 
from skimage.external.tifffile import imread
import glob, os, random
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add Poisson Noise in Sinogram')
    parser.add_argument('--p_val', type=int, nargs='+', default=[45000,150000], help='each for 1, 3mm')
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'test'])
    opt = parser.parse_args()

    mayo_3q = glob.glob('../../data/denoising/{}/mayo/forward_projection/3mm/quarter_3mm_*_fp.tif'.format(opt.dataset))
    # mayo_3f = glob.glob('../../data/denoising/train/mayo/forward_projection/3mm/full_3mm_*_fp.tif')
    mayo_1q = glob.glob('../../data/denoising/{}/mayo/forward_projection/1mm/quarter_1mm_*_fp.tif'.format(opt.dataset))
    # mayo_1f = glob.glob('../../data/denoising/train/mayo/forward_projection/1mm/full_1mm_*_fp.tif')

    n_mayo_3q = '../../data/denoising/{}/mayo/forward_projection/3mm_noise'.format(opt.dataset)
    n_mayo_1q = '../../data/denoising/{}/mayo/forward_projection/1mm_noise'.format(opt.dataset)

    if not os.path.exists(n_mayo_3q):
        os.mkdir(n_mayo_3q)
    if not os.path.exists(n_mayo_1q):
        os.mkdir(n_mayo_1q)

    mayo_1_3 = [mayo_1q, mayo_3q]
    n_mayo_1_3 = [n_mayo_1q, n_mayo_3q]

    # mayo_1_3 = [mayo_3q, mayo_3q]
    # n_mayo_1_3 = [n_mayo_3q, n_mayo_3q]

    #1mm, 3mm
    for idx, mayo in enumerate(mayo_1_3):
        vals = opt.p_val[idx]
        n_mayo = n_mayo_1_3[idx]

        for i,imgpath in enumerate(mayo,1):
            print('[{}/{}] start {}'.format(i,len(mayo),imgpath))
            basename = os.path.basename(imgpath)
            nbasename = '_'.join(basename.split('_')[:-1]) + '_nfp_' + str(vals) 
            nimgpath = os.path.join(n_mayo,nbasename)
            if not os.path.exists(nimgpath):
                os.mkdir(nimgpath)
            
            img = imread(imgpath)
            for j in range(720):
                arr = img[j,:,:]
                #L096:(-0.828, 21.838)
                #
                max_val = np.max(arr)
                min_val = np.min(arr)
                arr_norm = (arr-min_val)/(max_val-min_val)
                # vals = len(np.unique(arr_norm))
                # vals = 2**np.ceil(np.log2(vals))
                # print(vals)
                narr = np.random.poisson(arr_norm*vals)/float(vals) #poisson
                narr = narr*(max_val-min_val)+min_val

                imsave(os.path.join(nimgpath, '{}_{}.tif'.format(nbasename, str(j).zfill(3))), narr.astype('float32'))
                if j%100 == 0:
                    print('...{}/{} progressed'.format(j,720))

            print('[{}/{}] save {}'.format(i,len(mayo_1q),nimgpath))
            # exit()
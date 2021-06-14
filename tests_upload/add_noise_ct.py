import numpy as np 
import imageio
from skimage.external.tifffile import imsave 
from skimage.external.tifffile import imread
import glob, os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='substract noise in bp and add it to mayo')
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--p_val', type=int, nargs='+', default=[100000,400000], help='each for 1,3 thickness')
    parser.add_argument('--delete', type=float, default=0.3, help='delete percentage for each end direction')
    opt = parser.parse_args()
    bp_mayo_3q = glob.glob('../../data/denoising/{}/mayo/back_projection/3mm/quarter_3mm_*_bp.tif'.format(opt.dataset))
    bp_mayo_1q = glob.glob('../../data/denoising/{}/mayo/back_projection/1mm/quarter_1mm_*_bp.tif'.format(opt.dataset))
    
    n_bp_mayo_3q = '../../data/denoising/{}/mayo/back_projection/3mm_noise'.format(opt.dataset)
    n_bp_mayo_1q = '../../data/denoising/{}/mayo/back_projection/1mm_noise'.format(opt.dataset)
    
    mayo_3q = '../../data/denoising/{}/mayo/quarter_3mm'.format(opt.dataset)
    mayo_1q = '../../data/denoising/{}/mayo/quarter_1mm'.format(opt.dataset)

    bp_mayo_1_3 = [bp_mayo_1q, bp_mayo_3q]
    n_bp_mayo_1_3 = [n_bp_mayo_1q, n_bp_mayo_3q]
    mayo_1_3 = [mayo_1q, mayo_3q]

    # bp_mayo_1_3 = [bp_mayo_3q]
    # n_bp_mayo_1_3 = [n_bp_mayo_3q]
    # mayo_1_3 = [mayo_3q]

    for idx, mayo in enumerate(bp_mayo_1_3):
        n_bp_mayo = n_bp_mayo_1_3[idx]
        q_mayo = mayo_1_3[idx]
        p_val = opt.p_val[idx]

        for i,imgpath in enumerate(mayo,1):
            print('[{}/{}] start {}'.format(i,len(mayo),imgpath))
            basename = os.path.basename(imgpath)
            nbasename = '_'.join(basename.split('_')[:-1])+'_nbp_'+str(p_val)+'.tif'
            nimgpath = os.path.join(n_bp_mayo,nbasename)
            
            img = imread(imgpath)
            print(imgpath)
            nimg = imread(nimgpath)
            print(nimgpath)

            start_num = int(img.shape[0]*opt.delete)
            end_num = img.shape[0]-int(img.shape[0]*opt.delete)
            for j in range(start_num, end_num):
                name = (basename.split('_')[:-1])
                qimgpath = os.path.join(q_mayo,name[-1],'quarter_'+name[-2]+'-'+name[-1]+'-{0:03d}.tiff'.format(j))
                qimg = imread(qimgpath)
                print(qimgpath)

                #calculate noise
                arr = img[j,:,:]
                n_arr = nimg[j,:,:]

                # arr_norm = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
                # n_arr_norm = (n_arr-np.min(arr))/(np.max(arr)-np.min(arr)) #min, max which are based on n_arr make distribution scale differences btw arr,n_arr

                arr_norm = arr/23.0
                n_arr_norm = n_arr/23.0 #min, max which are based on n_arr make distribution scale differences btw arr,n_arr

                noise = n_arr_norm-arr_norm

                #add noise
                noise_add = noise + qimg
                noise_add[np.where(noise_add>1.0)] = 1.0
                noise_add[np.where(noise_add<0.0)] = 0.0
                

                save_path = os.path.join('../../data/denoising/{}/mayo/'.format(opt.dataset),'qquarter_'+name[-2])
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = os.path.join(save_path,name[-1])
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                    
                imsave(os.path.join(save_path,'qquarter_'+name[-2]+'-'+name[-1]+'-{0:03d}.tiff'.format(j)),noise_add.astype('float32'))


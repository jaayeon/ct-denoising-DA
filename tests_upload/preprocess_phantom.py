import os,glob
import numpy as np
import imageio

def change_os_slash(dir_name):
    if os.name == 'nt':  #window
        dir_name = dir_name.split('\\')
        out_dir_name = '/'.join(dir_name)
    elif os.name == 'posix':  #linux
        out_dir_name = dir_name

    return out_dir_name

ge_pth = '../../data/denoising/train/phantom/ge/pelvis'

#before run this code,
#you have to delete phantom images below..
#phantom-ge-chest level3, level5 : delete 001~041 & 301~all | remain only 043~300
for thck in [3,5]:
    ge_dir = os.path.join(ge_pth, 'level{}_*'.format(thck))
    ge_dir = glob.glob(ge_dir)
    ge_crop_dir = ge_dir[0] + '_crop'
    thck_paths = glob.glob(os.path.join(ge_dir[0], '*.tiff'))
    if not os.path.exists(ge_crop_dir):
        os.makedirs(ge_crop_dir)
    for i, img_pth in enumerate(thck_paths):
        img = imageio.imread(img_pth)
        img_pth = change_os_slash(img_pth)
        img_name = img_pth.split('/')[-1] # change '/' to '//' if your os is windows
        img_npy = np.array(img)
        img_crop = img_npy[60:460,60:460]
        ge_crop_path = os.path.join(ge_crop_dir, img_name)
        print('[{}]/[{}] : img crop {}'.format(i,len(thck_paths),img_name))
        imageio.imwrite(ge_crop_path, img_crop)

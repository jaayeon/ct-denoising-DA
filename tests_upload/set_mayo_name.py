import os, glob
import imageio
import numpy as np


mayo_test_glob = '../../data/denoising/test/mayo/*/*'
mayo_train_glob = '../../data/denoising/train/mayo/*/*'


mayo_dir_list = glob.glob(mayo_test_glob)
mayo_dir_list.extend(glob.glob(mayo_train_glob))

for mayo_dir in mayo_dir_list:
    thck, pid = mayo_dir.split('/')[-2], mayo_dir.split('/')[-1]

    for img_pth in glob.iglob(os.path.join(mayo_dir, '*')):
        img_name = os.path.basename(img_pth)
        
        new_img_name = '-'.join([thck,pid,img_name])
        new_img_pth = os.path.join(mayo_dir, new_img_name)
        os.rename(img_pth, new_img_pth)
        print('thck-{},pid-{}.. change : {} --> {}'.format(thck,pid,img_name, new_img_name))

#full_3mm-L067-full_3mm-L067-full_3mm-L067-004.tiff
#quarter_3mm L506 103.tiff

""" sapjil
mayo_test_glob = '../../data/denoising/test/mayo/*/*'


mayo_dir_list = glob.glob(mayo_test_glob)

for mayo_dir in mayo_dir_list:
    thck, pid = mayo_dir.split('/')[-2], mayo_dir.split('/')[-1]

    for img_pth in glob.iglob(os.path.join(mayo_dir, '*')):
        img_name = os.path.basename(img_pth)
        new1, new2, new3 = img_name[:-12], img_name[-12:-8], img_name[-8:]
        new_img_name = '-'.join([new1, new2, new3])
        new_img_pth = os.path.join(mayo_dir, new_img_name)
        os.rename(img_pth, new_img_pth)
        print('thck-{},pid-{}.. change : {} --> {}'.format(thck,pid,img_name, new_img_name))
    
 """

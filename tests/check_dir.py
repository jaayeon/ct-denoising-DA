import os
import glob

lp_gt_img_dir = r'../../data/denoising/test/lp-mayo/full'
lp_img_dir = r'../../data/denoising/test/lp-mayo/low'

m_img_dir = r'../../data/denoising/test/mayo/quarter_{}mm'.format(3)
m_gt_img_dir = r'../../data/denoising/test/mayo/full_{}mm'.format(3)

lp_img_list = glob.glob(os.path.join(lp_img_dir, '*', '*'))
print(lp_img_list)

m_img_list = glob.glob(os.path.join(m_img_dir, '*', '*'))
print(m_img_list)
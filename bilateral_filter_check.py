import sys, os
import glob
import torch
import cv2
from models.losses import ssim_loss
from utils.metric import calc_metrics, forward_ensemble
import numpy as np
from options import args
from utils.helper import set_gpu, set_test_dir
from skimage.external.tifffile import imsave, imread

opt = args

filtered_result_dir = r'../../data/denoising/bilateral_filtered_results/'
concat_result_dir = r'../../data/denoising/concate_filtered_results/'


m3_img_dir = r'../../data/denoising/test/mayo/quarter_3mm'
m1_img_dir = r'../../data/denoising/test/mayo/quarter_1mm'

lbl3_img_dir = r'../../data/denoising/test/mayo/full_3mm'
lbl1_img_dir = r'../../data/denoising/test/mayo/full_1mm'


m3_img_list = glob.glob(os.path.join(m3_img_dir, '*', '*'))
m1_img_list = glob.glob(os.path.join(m1_img_dir, '*', '*'))


lbl3_img_list = glob.glob(os.path.join(lbl3_img_dir, '*', '*'))
lbl1_img_list = glob.glob(os.path.join(lbl1_img_dir, '*', '*'))

if not os.path.exists(filtered_result_dir):
        os.makedirs(filtered_result_dir)
        
#concat_result_dir = os.path.join(filtered_result_dir, 'concat_img')

if not os.path.exists(concat_result_dir):
    os.makedirs(concat_result_dir)

m3_img_list.sort()
m1_img_list.sort()
lbl3_img_list.sort()
lbl1_img_list.sort()

if opt.filter_dataset == '1mm':
    num_total_img = len(m1_img_list)
    m_img_list = m1_img_list 
    lbl_img_list = lbl1_img_list
elif opt.filter_dataset == '3mm':
    num_total_img = len(m3_img_list)
    m_img_list = m3_img_list 
    lbl_img_list = lbl3_img_list
else:
    num_total_img = len(m1_img_list) + len(m3_img_list)
    m_img_list = m1_img_list + m3_img_list 
    lbl_img_list = lbl1_img_list + lbl3_img_list
    
#print(num_total_img)

num = 0

avg_psnr = 0.0
avg_ssim = 0.0
avg_loss = 0.0
noise_avg_loss = 0.0
noise_avg_psnr = 0.0
noise_avg_ssim = 0.0

for img_idx, path in enumerate(zip(m_img_list,lbl_img_list),1):
    img_path = path[0]
    print('img_path : ', img_path)
    lbl_img_path = path[1]
    print('lbl_img_path ; ', lbl_img_path)

    img_path = os.path.abspath(img_path)
    img_name = 'out_'+os.path.basename(img_path)
    concat_name = 'concat_'+os.path.basename(img_path)
    
    print("[{}/{}] processing {}".format(img_idx, num_total_img, os.path.abspath(img_path)))

    input_img = imread(img_path)
    lbl_img = imread(lbl_img_path)
    input_img_shape = input_img.shape

    #only for gray scale img
    input_img_tensor = torch.from_numpy(input_img.reshape(1,1,input_img.shape[0], input_img.shape[1])).to('cpu')
    lbl_img_tensor = torch.from_numpy(lbl_img.reshape(1,1,lbl_img.shape[0], lbl_img.shape[1])).to('cpu')


    input_tensor = input_img_tensor

    input_tensor_shape = input_img_tensor.shape
    input_tensor = input_img_tensor.type(torch.FloatTensor)

    
    #print(input_tensor_shape)

    self_img = torch.squeeze(input_tensor,0)
    self_img = self_img[0,:,:]
    self_img = self_img.numpy()
    self_filter_img = cv2.bilateralFilter(self_img,opt.d, opt.window_size, opt.window_size)


    out_img_tensor = torch.tensor(self_filter_img)
    out_img_tensor = torch.unsqueeze(out_img_tensor,0)
    out_img_tensor = torch.unsqueeze(out_img_tensor,0)
    #print(out_img_tensor.shape)
    noise_loss, noise_psnr, noise_ssim, batch_loss, batch_psnr, batch_ssim = calc_metrics(input_img_tensor, out_img_tensor, lbl_img_tensor)
    out_img = out_img_tensor[0,0,:,:].detach().numpy()

   
    concat_img = np.concatenate((input_img, out_img, lbl_img), axis=1)

    num += 1
    avg_loss += batch_loss
    avg_psnr += batch_psnr
    avg_ssim += batch_ssim
    noise_avg_loss += noise_loss
    noise_avg_psnr += noise_psnr
    noise_avg_ssim += noise_ssim

    out_img_path = os.path.join(filtered_result_dir, img_name)
    concat_img_path = os.path.join(concat_result_dir, concat_name)

    print("** Test => Image({}/{}): Noise Loss: {:.8f}, Noise PSNR: {:.8f}, Noise SSIM: {:.8f}, Loss: {:.8f}, PSNR: {:.8f}, SSIM: {:.8f}".format(
            img_idx, num_total_img, noise_loss.item(), noise_psnr.item(), noise_ssim.item(), batch_loss.item(), batch_psnr.item(), batch_ssim.item()
        ))

        # print("out_img.shape:", out_img.shape)
        # print(os.path.abspath(out_img_path))
    imsave(concat_img_path, concat_img)
    imsave(out_img_path, out_img)

print(" #{:d} Test Average Noise Loss: {:.8f}, Average Noise PSNR: {:.8f}, Average Noise SSIM: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}, Average SSIM: {:.8f}".format(
        num, noise_avg_loss / num, noise_avg_psnr / num, noise_avg_ssim / num, avg_loss / num, avg_psnr / num, avg_ssim / num
    ))

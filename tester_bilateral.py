import sys, os, glob
import time

from skimage.external.tifffile import imsave, imread
import numpy as np
import cv2
import torch
import torch.nn as nn

import data.make_patches as mp
from data.make_patches import pad_tensor, unpad_tensor
from utils.saver import select_checkpoint_dir, load_model
from utils.metric import calc_metrics, forward_ensemble
from options import args

from models import set_model
from utils.helper import set_gpu, set_test_dir

def run_test(opt, img_list, gt_img_list):
    print('- Apply bilateral Filter -')

    opt = set_gpu(opt)
    opt.device = 'cpu'
    #set_test_dir(opt)
    opt.test_result_dir = opt.test_result_dir + '/Bilateral filter_' + str(opt.thickness) + 'mm'
    #set_test_dir(opt)
    if not os.path.exists(opt.test_result_dir):
        os.makedirs(opt.test_result_dir)

    test_concat_dir = os.path.join(opt.test_result_dir, 'concat_img')
    if not os.path.exists(test_concat_dir):
        os.makedirs(test_concat_dir)

    img_list.sort()
    gt_img_list.sort()

    num_total_img = len(img_list)

    num = 0
    avg_loss = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0
    noise_avg_loss = 0.0
    noise_avg_psnr = 0.0
    noise_avg_ssim = 0.0

    for img_idx, path in enumerate(zip(img_list,gt_img_list),1):

        if img_idx % 4 :
            pass
        else:
            img_path = path[0]
            print('img_path : ', img_path)
            gt_img_path = path[1]
            print('gt_img_path ; ', gt_img_path)
            start_time = time.time()
            img_path = os.path.abspath(img_path)
            img_name = 'out_'+os.path.basename(img_path)
            concat_name = 'concat_'+os.path.basename(img_path)
            num_test_img = int(num_total_img/4)

            print("[{}/{}] processing {}".format(num+1, num_test_img, os.path.abspath(img_path)))

            input_img = imread(img_path)
            gt_img = imread(gt_img_path)
            input_img_shape = input_img.shape

            #only for gray scale img
            input_img_tensor = torch.from_numpy(input_img.reshape(1,1,input_img.shape[0], input_img.shape[1])).to(opt.device)
            gt_img_tensor = torch.from_numpy(gt_img.reshape(1,1,gt_img.shape[0], gt_img.shape[1])).to(opt.device)

            if opt.test_patches:
                """  
                have to fix img2tensor
                """
                pad_ts = pad_tensor(input_img_tensor, opt.patch_size, opt.patch_offset)
                input_tensor = mp.make_tensor_arr_patches(pad_ts, opt.patch_size, opt.patch_offset)
            else : 
                input_tensor = input_img_tensor

            input_tensor_shape = input_tensor.shape
            input_tensor = input_tensor.type(torch.FloatTensor)
            out = torch.ones(input_tensor_shape)

            if opt.use_cuda:
                input_tensor = input_tensor.to(opt.device)

            self_img = torch.squeeze(input_tensor,0)
            self_img = self_img[0,:,:]
            self_img = self_img.numpy()
            self_filter_img = cv2.bilateralFilter(self_img, opt.d, opt.window, opt.window)

            out_img_tensor = torch.tensor(self_filter_img)
            out_img_tensor = torch.unsqueeze(out_img_tensor,0)
            out_img_tensor = torch.unsqueeze(out_img_tensor,0)
            #print(out_img_tensor.shape)

            out_img_path = os.path.join(opt.test_result_dir, img_name)
            concat_img_path = os.path.join(test_concat_dir, concat_name)

            noise_loss, noise_psnr, noise_ssim, batch_loss, batch_psnr, batch_ssim = calc_metrics(input_img_tensor, out_img_tensor, gt_img_tensor)
            # print('nl : {:.8f}, np : {:.8f}, ol : {:.8f}, op : {:.8f}'.format(noise_loss, noise_psnr, batch_loss, batch_psnr))
            
            #only for gray scale img
            if opt.use_cuda:
                out_img = out_img_tensor[0,0,:,:].to('cpu').detach().numpy()
            else : 
                out_img = out_img_tensor[0,0,:,:].detach().numpy()

            concat_img = np.concatenate((input_img, out_img, gt_img), axis=1)

            num += 1
            avg_loss += batch_loss
            avg_psnr += batch_psnr
            avg_ssim += batch_ssim
            noise_avg_loss += noise_loss
            noise_avg_psnr += noise_psnr
            noise_avg_ssim += noise_ssim
            
 
            print("** Test {:.3f}s => Image({}/{}): Noise Loss: {:.8f}, Noise PSNR: {:.8f}, Noise SSIM: {:.8f}, Loss: {:.8f}, PSNR: {:.8f}, SSIM: {:.8f}".format(
                time.time() - start_time, img_idx, num_test_img, noise_loss.item(), noise_psnr.item(), noise_ssim.item(), batch_loss.item(), batch_psnr.item(), batch_ssim.item()
            ))

            # print("out_img.shape:", out_img.shape)
            # print(os.path.abspath(out_img_path))
            imsave(concat_img_path, concat_img)
            imsave(out_img_path, out_img)

    print(" #{:d} Test Average Noise Loss: {:.8f}, Average Noise PSNR: {:.8f}, Average Noise SSIM: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}, Average SSIM: {:.8f}".format(
        num, noise_avg_loss / num, noise_avg_psnr / num, noise_avg_ssim / num, avg_loss / num, avg_psnr / num, avg_ssim / num))
    

    print("---Time: %.4fs\n" % (time.time() - start_time))
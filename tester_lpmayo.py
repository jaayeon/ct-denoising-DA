import sys, os
import glob
import time

from skimage.external.tifffile import imsave, imread
import numpy as np

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
    print('- Initialize networks for training')
    net = set_model(opt)
    _, net, _ = load_model(opt, net)
    print(net)

    opt = set_gpu(opt)

    if opt.use_cuda:
        net = net.to(opt.device)

    if opt.multi_gpu:
        net = nn.DataParallel(net)

    # opt.test_result_dir = opt.test_result_dir + "-" + opt.model + '-patch' + str(opt.patch_size)
    set_test_dir(opt)
    if not os.path.exists(opt.test_result_dir):
        os.makedirs(opt.test_result_dir)

    img_list.sort()
    gt_img_list.sort()

    num_total_img = len(img_list)
    C_num = 0
    C_avg_loss = 0.0
    C_avg_psnr = 0.0
    C_noise_avg_loss = 0.0
    C_noise_avg_psnr = 0.0

    L_num = 0
    L_avg_loss = 0.0
    L_avg_psnr = 0.0
    L_noise_avg_loss = 0.0
    L_noise_avg_psnr = 0.0

    N_num = 0
    N_avg_loss = 0.0
    N_avg_psnr = 0.0
    N_noise_avg_loss = 0.0
    N_noise_avg_psnr = 0.0
    for img_idx, path in enumerate(zip(img_list,gt_img_list),1):

        img_path = path[0]
        # print('img_path : ', img_path)
        gt_img_path = path[1]
        # print('gt_img_path ; ', gt_img_path)
        start_time = time.time()
        img_path = os.path.abspath(img_path)
        img_name = 'out_'+os.path.basename(img_path)

        print("[{}/{}] processing {}".format(img_idx, num_total_img, os.path.abspath(img_path)))

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


        # else:
        #     input_tensor = input_img
        #     if len(img.shape) ==2:
        #         img = img.reshape(1, 1, img.shape[0], img.shape[1])

        input_tensor_shape = input_tensor.shape
        input_tensor = input_tensor.type(torch.FloatTensor)

        if opt.use_cuda:
            input_tensor = input_tensor.to(opt.device)

        with torch.no_grad(): #(b,c,h,w) or (1,1,h,w)
            if opt.ensemble:
                """  
                ensemble code... i'll add it later
                """
                out_tensor = torch.zeros(input_tensor.shape)
                for i in range(input_tensor.size(0)):
                    print("input_tensor[{}:{}].shape: {}".format(i, i+1, input_tensor[i:i+1].shape))
                    out_tensor[i] = forward_ensemble(input_tensor[i:i+1], net=net, device=opt.device)
            else:
                out_tensor = net(input_tensor)
            # print("out_tensor:", out_tensor.size())

            out_tensor = out_tensor.to(opt.device)

            # out = out.reshape(img_dims)
            # print("out.shape:", out.shape)

        dst_img_path = os.path.join(opt.test_result_dir, img_name)

        if opt.test_patches:
            out_tensor = mp.recon_tensor_arr_patches(out_tensor, input_img.shape[1], input_img.shape[0], opt.patch_size, opt.patch_offset)
            print("out_img.shape:", out_tensor.shape)
            out_img_tensor = unpad_tensor(out_tensor, opt.patch_offset, input_tensor_shape)
            print("unpad out_img.shape:", out_img_tensor.shape)
        else:
            out_img_tensor = out_tensor

        noise_loss, noise_psnr, batch_loss, batch_psnr = calc_metrics(input_img_tensor, out_img_tensor, gt_img_tensor)
        # print('nl : {:.8f}, np : {:.8f}, ol : {:.8f}, op : {:.8f}'.format(noise_loss, noise_psnr, batch_loss, batch_psnr))
        
        #only for gray scale img
        if opt.use_cuda:
            out_img = out_img_tensor[0,0,:,:].to('cpu').detach().numpy()
        else : 
            out_img = out_img_tensor[0,0,:,:].detach().numpy()

        concat_img = np.concatenate((input_img, out_img, gt_img), axis=1)

        if 'C' in img_path:
            C_num += 1
            C_avg_loss += batch_loss
            C_avg_psnr += batch_psnr
            C_noise_avg_loss += noise_loss
            C_noise_avg_psnr += noise_psnr
        elif 'L' in img_path : 
            L_num += 1
            L_avg_loss += batch_loss
            L_avg_psnr += batch_psnr
            L_noise_avg_loss += noise_loss
            L_noise_avg_psnr += noise_psnr
        elif 'N' in img_path : 
            N_num += 1
            N_avg_loss += batch_loss
            N_avg_psnr += batch_psnr
            N_noise_avg_loss += noise_loss
            N_noise_avg_psnr += noise_psnr
        else :
            print('check img_path : {}'.format(img_path))
            raise KeyboardInterrupt

        print("** Test {:.3f}s => Image({}/{}): Noise Loss: {:.8f}, Noise PSNR: {:.8f}, Loss: {:.8f}, PSNR: {:.8f}".format(
            time.time() - start_time, img_idx, num_total_img, noise_loss.item(), noise_psnr.item(), batch_loss.item(), batch_psnr.item()
        ))

        # print("out_img.shape:", out_img.shape)
        # print(os.path.abspath(dst_img_path))
        imsave(dst_img_path, concat_img)

    print("[CHEST] #{:d} Test Average Noise Loss: {:.8f}, Average Noise PSNR: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}".format(
        C_num, C_noise_avg_loss / C_num, C_noise_avg_psnr / C_num, C_avg_loss / C_num, C_avg_psnr / C_num
    ))
    print("[ABDOMEN] #{:d} Test Average Noise Loss: {:.8f}, Average Noise PSNR: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}".format(
        L_num, L_noise_avg_loss / L_num, L_noise_avg_psnr / L_num, L_avg_loss / L_num, L_avg_psnr / L_num
    ))
    print("[HEAD] #{:d} Test Average Noise Loss: {:.8f}, Average Noise PSNR: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}".format(
        N_num, N_noise_avg_loss / N_num, N_noise_avg_psnr / N_num, N_avg_loss / N_num, N_avg_psnr / N_num
    ))

    print("---Time: %.4fs\n" % (time.time() - start_time))
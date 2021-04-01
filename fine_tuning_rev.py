import os, time, math, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from utils.helper import set_checkpoint_dir, set_gpu, set_test_dir
from utils.metric import calc_metrics
from utils.loader import load_model
from utils.saver import Record
from models import set_model

from skimage.external.tifffile import imsave, imread
from skimage.restoration import denoise_nl_means, estimate_sigma
import cv2
import copy

class Single_Image_Data(Dataset):
    def __init__(self, opt, iternum, patchsize, imgpath, refpath, noise=False, crop='center', batch_size=1):
        self.patchsize = patchsize
        self.iternum = iternum
        self.img = imread(imgpath)
        self.ref = imread(refpath)
        self.imgsize = self.img.shape[0]
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.add_noise = noise
        self.crop = crop
        self.batch_size = batch_size

        self.scale_min = opt.scale_min
        self.scale_max = opt.scale_max
        self.opt = opt

        self.random_rh = [random.randint(0,self.imgsize-self.patchsize) for _ in range(batch_size)]
        self.random_rw = [random.randint(0,self.imgsize-self.patchsize) for _ in range(batch_size)]

    def __getitem__(self, idx):
        idx = int(idx%self.batch_size)
        size = self.imgsize
        #random crop
        if self.crop == 'random':
            rh = self.random_rh[idx]
            rw = self.random_rw[idx]
        #center crop
        elif self.crop == 'center':
            rh = int((size-self.patchsize)/2)
            rw = int((size-self.patchsize)/2)

        img = self.img[rh:rh+self.patchsize, rw:rw+self.patchsize]
        ref = self.ref[rh:rh+self.patchsize, rw:rw+self.patchsize]

        if self.add_noise:
            nimg = self.make_noise(img)
            nimg = Image.fromarray(nimg)
            nimgtensor = self.transforms(nimg)
            nimgtensor = nimgtensor.type(torch.FloatTensor)

        #img2tensor
        # img = Image.fromarray(img)
        # ref = Image.fromarray(ref)
        imgtensor = self.transforms(img)
        reftensor = self.transforms(ref)

        imgtensor = imgtensor.type(torch.FloatTensor)
        reftensor = reftensor.type(torch.FloatTensor)

        if self.add_noise:
            return imgtensor, reftensor, nimgtensor
        else:
            return imgtensor, reftensor

    def __len__(self):
        return self.iternum*self.batch_size

    def get_tensor(self):
        h,w = self.img.shape
        imgtensor = self.transforms(self.img)
        reftensor = self.transforms(self.ref)
        imgtensor = imgtensor.reshape(1,1,h,w)
        reftensor = reftensor.reshape(1,1,h,w)
        return imgtensor, reftensor

    def make_noise(self, img):

        num_noise_modes = len(self.opt.noise)
        noise = self.opt.noise[random.randint(0,num_noise_modes-1)]
        pidx = 1 if self.opt.thickness==3 else 0

        scale = random.randint(self.scale_min*2,self.scale_max*2)/2
        sigma_est = np.mean(estimate_sigma(img, multichannel=False))

        if noise=='p':
            params = self.opt.p_lam
            nimg = np.random.poisson(params[pidx]*img)/float(params[pidx])
            nimg = img + scale*(nimg-img)
        elif noise=='g':
            params = self.opt.g_std
            # noise = np.random.normal(loc=0, scale=params[pidx], size=img.shape).astype(float)
            noise = np.random.normal(loc=0, scale=scale*sigma_est*self.opt.ratio_std, size=img.shape).astype(float)
            nimg = img + noise
        elif noise=='bf':
            params = self.opt.b_dcs
            clean = cv2.bilateralFilter(img, int(params[0]), scale*sigma_est*self.opt.ratio_std, params[2])
            noise = img-clean
            if params[1]<0.1:
                nimg = img + noise/params[1]/10 #amplify noise.. 0.1-> 1, 0.05->2, 0.01->10
            else : 
                nimg = img + noise
        elif noise=='nlm':
            clean = denoise_nl_means(img, h=sigma_est*self.opt.ratio_std, fast_mode=True, 
                                    patch_size=5, patch_distance=13, multichannel=False)
            noise = img-clean
            nimg = img + scale*noise
        return nimg


def run_train(opt, source, target):
    opt= set_gpu(opt)
    print('Initialize networks for fine tuning')
    net = set_model(opt)
    print(net)

    if opt.pretrained : 
        net, checkpoint = load_model(opt, net)
        # set_checkpoint_dir(opt)
        # net_init = load_model(opt, net)
        # net.load_state_dict(copy.deepcopy(net_init.state_dict()))
    else : 
        raise KeyboardInterrupt('--mode fine_tuning has to be used with --pretrained option')

    set_test_dir(opt)
    if not os.path.exists(opt.test_result_dir):
        os.makedirs(opt.test_result_dir)

    test_concat_dir = os.path.join(opt.test_result_dir, 'concat_img')
    if not os.path.exists(test_concat_dir):
        os.makedirs(test_concat_dir)

    if opt.multi_gpu:
        net.denoiser = nn.DataParallel(net.denoiser)
        net.domain_discriminator = nn.DataParallel(net.domain_discriminator)

    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    if opt.use_cuda:
        mse_criterion = mse_criterion.to(opt.device)
        mae_criterion = mae_criterion.to(opt.device)

    src_input_list_1 = glob.glob(os.path.join(opt.train_dir, 'phantom', opt.source, 'chest', '{}*_crop'.format(opt.mA_low), '*.tiff'))
    src_ref_list_1 = glob.glob(os.path.join(opt.train_dir, 'phantom', opt.source, 'chest', '{}*_crop'.format(opt.mA_full), '*.tiff'))
    src_input_list_2 = glob.glob(os.path.join(opt.train_dir, 'phantom', opt.source, 'pelvis', '{}*_crop'.format(opt.mA_low), '*.tiff'))
    src_ref_list_2 = glob.glob(os.path.join(opt.train_dir, 'phantom', opt.source, 'pelvis', '{}*_crop'.format(opt.mA_full), '*.tiff'))
    src_input_list = src_input_list_1 + src_input_list_2
    src_ref_list = src_ref_list_1 + src_ref_list_2

    print(opt.test_dir)
    trg_input_list = glob.glob(os.path.join(opt.test_dir, opt.target, 'quarter_{}mm'.format(opt.thickness), '*', '*.tiff'))
    trg_ref_list = glob.glob(os.path.join(opt.test_dir, opt.target, 'full_{}mm'.format(opt.thickness), '*', '*.tiff'))

    src_input_list.sort()
    src_ref_list.sort()
    trg_input_list.sort()
    trg_ref_list.sort()

    num_src_img = len(src_input_list)
    num_trg_img = len(trg_input_list)
    print(num_src_img, num_trg_img)

    num = 0
    avg_loss = 0.0
    avg_psnr = 0.0
    avg_ssim = 0.0
    noise_avg_loss = 0.0
    noise_avg_psnr = 0.0
    noise_avg_ssim = 0.0
    for img_idx, path in enumerate(zip(trg_input_list, trg_ref_list),1):
        if img_idx % 4 :
            pass
        else:
            trg_input_path = path[0]
            print('trg_input_path : ', trg_input_path)
            trg_ref_path = path[1]
            print('trg_ref_path ; ', trg_ref_path)
            start_time = time.time()
            trg_input_path = os.path.abspath(trg_input_path)
            img_name = 'out_'+os.path.basename(trg_input_path)
            concat_name = 'concat_'+os.path.basename(trg_input_path)
            num_test_img = int(num_trg_img/4)

            print("[{}/{}] processing {}".format(num+1, num_test_img, os.path.abspath(trg_input_path)))

            ridx = np.random.randint(num_src_img)
            
            trg_dataset = Single_Image_Data(opt, opt.fine_tuning_num, opt.patch_size, trg_input_path, trg_ref_path, noise=True, crop=opt.crop, batch_size=opt.batch_size)
            src_dataset = Single_Image_Data(opt, opt.fine_tuning_num, opt.patch_size, src_input_list[ridx], src_ref_list[ridx], crop=opt.crop, batch_size=opt.batch_size)
            
            trg_loader = DataLoader(dataset=trg_dataset, batch_size=opt.batch_size)
            src_loader = DataLoader(dataset=src_dataset, batch_size=opt.batch_size)

            #initialize net, optimizer
            # net.load_state_dict(copy.deepcopy(net_init.state_dict()))
            net.load_state_dict(checkpoint['model'])
            optimizer = optim.Adam(net.denoiser.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=opt.weight_decay)
            optimizer_dc = optim.Adam(net.domain_discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=opt.weight_decay_dc)
            net.denoiser.train()
            net.domain_discriminator.eval()

            #before fine_tuning
            with torch.no_grad():
                trg_img_tensor, trg_ref_tensor = trg_dataset.get_tensor()
                trg_img_tensor = trg_img_tensor.to(opt.device)
                trg_ref_tensor = trg_ref_tensor.to(opt.device)
                trg_out_tensor,_ = net.denoiser(trg_img_tensor)
                _, n_psnr, n_ssim,  _, psnr, ssim = calc_metrics(trg_img_tensor, trg_out_tensor, trg_ref_tensor) 

            best_loss = 10000
            fine_tuned = False
            #fine_tuning for patch imgs
            for idx, data in enumerate(zip(src_loader, trg_loader)):
                src_input_tensor, src_ref_tensor = data[0][0].to(opt.device), data[0][1].to(opt.device)
                trg_input_tensor, _ , trg_noise_tensor = data[1][0].to(opt.device), data[1][1].to(opt.device), data[1][2].to(opt.device)

                #domain classifier
                # optimizer_dc.zero_grad()
                # net.domain_discriminator.zero_grad()
                # dc_loss = net.dc_loss(src_input_tensor, src_ref_tensor, trg_input_tensor)
                # dc_loss.backward()
                # optimizer_dc.step()
                    
                #denoiser by src%trg_noise
                # optimizer.zero_grad()
                # net.denoiser.zero_grad()
                # loss, l_loss, p_loss, rev_loss = net.g_loss(src_input_tensor, trg_input_tensor, src_ref_tensor, perceptual=True, trg_noise=trg_noise_tensor, return_losses=True)
                # loss.backward()
                # optimizer.step()
                # print('loss : {:.6f}, l_loss : {:.6f}, p_loss : {:.6f}, rev_loss : {:.6f}'.format(loss.item(), l_loss.item(), p_loss.item(), rev_loss.item()))

                #denoiser by only trg_noise
                optimizer.zero_grad()
                net.denoiser.zero_grad()
                trg_patch_tensor, _ = net.denoiser(trg_noise_tensor)
                loss = mse_criterion(trg_patch_tensor, trg_input_tensor)
                if loss < best_loss:
                    best_loss = loss
                    #test original img
                    with torch.no_grad():
                        trg_out_tensor,_ = net.denoiser(trg_img_tensor)
                        _, n_psnr, n_ssim,  _, psnr, ssim = calc_metrics(trg_img_tensor, trg_out_tensor, trg_ref_tensor) 
                    print('--->#{} updated! loss : {:.6f}'.format(idx, loss.item()))

                loss.backward()
                optimizer.step()

            #test original img
            # with torch.no_grad():
            #     trg_input_tensor, trg_ref_tensor = trg_dataset.get_tensor()
            #     trg_input_tensor = trg_input_tensor.to(opt.device)
            #     trg_ref_tensor = trg_ref_tensor.to(opt.device)
            #     trg_out_tensor,_ = net.denoiser(trg_input_tensor)
            #     _, n_psnr, n_ssim,  _, psnr, ssim = calc_metrics(trg_input_tensor, trg_out_tensor, trg_ref_tensor) 


            out_img_path = os.path.join(opt.test_result_dir, img_name)
            concat_img_path = os.path.join(test_concat_dir, concat_name)

            #only for gray scale img
            if opt.use_cuda:
                trg_input_img = trg_img_tensor[0,0,:,:].to('cpu').detach().numpy()
                trg_ref_img = trg_ref_tensor[0,0,:,:].to('cpu').detach().numpy()
                trg_out_img = trg_out_tensor[0,0,:,:].to('cpu').detach().numpy()
            else : 
                trg_input_img = trg_img_tensor[0,0,:,:].detach().numpy()
                trg_ref_img = trg_ref_tensor[0,0,:,:].detach().numpy()
                trg_out_img = trg_out_tensor[0,0,:,:].detach().numpy()

            concat_img = np.concatenate((trg_input_img, trg_out_img, trg_ref_img), axis=1)

            num += 1
            avg_psnr += psnr
            avg_ssim += ssim
            noise_avg_psnr += n_psnr
            noise_avg_ssim += n_ssim

            imsave(concat_img_path, concat_img)
            imsave(out_img_path, trg_out_img)

            print("** Test {:.3f}s => Image({}/{}): Noise PSNR: {:.8f}, Noise SSIM: {:.8f}, PSNR: {:.8f}, SSIM: {:.8f}".format(
                time.time() - start_time, img_idx, num_test_img, n_psnr.item(), n_ssim.item(), psnr.item(), ssim.item()
            ))

    print(" #{:d} Test Average Noise PSNR: {:.8f}, Average Noise SSIM: {:.8f}, Average PSNR: {:.8f}, Average SSIM: {:.8f}".format(
        num, noise_avg_psnr / num, noise_avg_ssim / num, avg_psnr / num, avg_ssim / num
    ))

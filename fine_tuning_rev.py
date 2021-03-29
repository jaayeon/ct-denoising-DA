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

class Single_Image_Data(Dataset):
    def __init__(self, iternum, imgsize, imgpath, refpath):
        self.imgsize = imgsize
        self.iternum = iternum
        self.img = imread(imgpath)
        self.ref = imread(refpath)
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem(self, idx):
        #random crop
        size = self.img.shape[0]
        rh = random.randint(size-self.imgsize)
        rw = random.randint(size-self.imgsize)
        img = self.img[rh:rh+self.imgsize, rw:rw+self.imgsize]
        ref = self.ref[rh:rh+self.imgsize, rw:rw+self.imgsize]
        #img2tensor
        img = Image.fromarray(img)
        ref = Image.fromarray(ref)
        imgtensor = self.transforms(img)
        reftensor = self.transforms(ref)

        imgtensor = imgtensor.type(torch.FloatTensor)
        reftensor = reftensor.type(torch.FloatTensor)
        return imgtensor, reftensor

    def __len__(self):
        return self.iternum

    def get_tensor(self):
        imgtensor = self.transforms(self.img)
        reftensor = self.transforms(self.ref)
        imgtensor = imgtensor.reshape(1, imgtensor.size())
        reftensor = reftensor.reshape(1, reftensor.size())
        return imgtensor, reftensor


def run_train(opt, source, target):
    opt= set_gpu(opt)
    print('Initialize networks for fine tuning')
    net = set_model(opt)
    print(net)
    
    print("Setting Optimizer")
    if opt.optimizer == 'adam':
        optimizer = optim.Adam(net.denoiser.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=opt.weight_decay)
        optimizer_dc = optim.Adam(net.domain_discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=opt.weight_decay_dc)
        print("===> Use Adam optimizer")
    elif opt.optimizer == 'rms':
        optimizer = optim.RMSprop(net.denoiser.parameters(), lr=opt.lr, eps=1e-8, weight_decay=opt.weight_decay)
        optimizer_dc = optim.RMSprop(net.domain_discriminator.parameters(), lr=opt.lr, eps=1e-8, weight_decay=opt.weight_decay)
        print("===> Use RMSprop optimizer")

    if opt.pretrained : 
        net = load_model(opt, net)
        set_checkpoint_dir(opt)
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

    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')
    scheduler_dc = ReduceLROnPlateau(optimizer_dc, factor=0.5, patience=5, mode='min')

    mse_criterion = nn.MSELoss()
    if opt.use_cuda:
        mse_criterion = mse_criterion.to(opt.device)

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

            # trg_input_img = imread(trg_input_path)
            # trg_ref_img = imread(trg_ref_path)

            ridx = np.random.randint(num_src_img)
            # src_input_img = imread(src_input_list[ridx])
            # src_ref_img = imread(src_ref_list[ridx])
            
            trg_dataset = Single_Image_Data(opt.fine_tuning_num, opt.patch_size, trg_input_path, trg_ref_path)
            src_dataset = Single_Image_Data(opt.fine_tuning_num, opt.path_size, src_input_list[ridx], src_ref_list[ridx])
            
            trg_loader = DataLoader(dataset=trg_dataset, batch_size=1)
            src_loader = DataLoader(dataset=src_dataset, batch_size=1)

            # trg_out_tensor, net, optimizer, optimizer_dc, n_psnr, n_ssim,  psnr, ssim = fine_tuning(opt, trg_loader, src_loader, net, optimizer, optimizer_dc)
            net, optimizer, optimizer_dc = fine_tuning(opt, trg_loader, src_loader, net, optimizer, optimizer_dc)
            trg_input_tensor, trg_ref_tensor = trg_dataset.get_tensor()
            net.denoiser.eval()
            trg_out_tensor = net.denoiser(trg_input_tensor)
            _, n_psnr, n_ssim,  _, psnr, ssim = calc_metrics(trg_input_tensor, trg_out_tensor, trg_ref_tensor) 


            out_img_path = os.path.join(opt.test_result_dir, img_name)
            concat_img_path = os.path.join(test_concat_dir, concat_name)

            #only for gray scale img
            if opt.use_cuda:
                trg_input_img = trg_input_tensor[0,0,:,:].to('cpu').detach().numpy()
                trg_ref_img = trg_ref_tensor[0,0,:,:].to('cpu').detach().numpy()
                trg_out_img = trg_out_tensor[0,0,:,:].to('cpu').detach().numpy()
            else : 
                trg_input_img = trg_input_tensor[0,0,:,:].detach().numpy()
                trg_ref_img = trg_ref_tensor[0,0,:,:].detach().numpy()
                trg_out_img = trg_out_tensor[0,0,:,:].detach().numpy()

            concat_img = np.concatenate((trg_input_img, trg_out_img, trg_ref_img), axis=1)

            num += 1
            avg_psnr += psnr
            avg_ssim += ssim
            noise_avg_psnr += n_psnr
            noise_avg_ssim += n_ssim


            print("** Test {:.3f}s => Image({}/{}): Noise PSNR: {:.8f}, Noise SSIM: {:.8f}, PSNR: {:.8f}, SSIM: {:.8f}".format(
                time.time() - start_time, img_idx, num_test_img, n_psnr.item(), n_ssim.item(), psnr.item(), ssim.item()
            ))

            # print("out_img.shape:", out_img.shape)
            # print(os.path.abspath(out_img_path))
            imsave(concat_img_path, concat_img)
            imsave(out_img_path, trg_out_img)

    print(" #{:d} Test Average Noise PSNR: {:.8f}, Average Noise SSIM: {:.8f}, Average PSNR: {:.8f}, Average SSIM: {:.8f}".format(
        num, noise_avg_psnr / num, noise_avg_ssim / num, avg_psnr / num, avg_ssim / num
    ))


def fine_tuning(opt, trg_loader, src_loader, net, optimizer, optimizer_dc):

    # trg_input_tensor = torch.from_numpy(trg_input_img.reshape(1,1,trg_input_img.shape[0], trg_input_img.shape[1])).type(torch.FloatTensor).to(opt.device)
    # trg_ref_tensor = torch.from_numpy(trg_ref_img.reshape(1,1,trg_ref_img.shape[0], trg_ref_img.shape[1])).type(torch.FloatTensor).to(opt.device)
    # src_input_tensor = torch.from_numpy(src_input_img.reshape(1,1,src_input_img.shape[0], src_input_img.shape[1])).type(torch.FloatTensor).to(opt.device)
    # src_ref_tensor = torch.from_numpy(src_ref_img.reshape(1,1,src_ref_img.shape[0], src_ref_img.shape[1])).type(torch.FloatTensor).to(opt.device)

    # if src_input_tensor.size()[-1]<trg_input_tensor.size()[-1]:
    #     pad = trg_input_tensor.size()[-1]-src_input_tensor.size()[-1]
    #     padding = nn.ZeroPad2d(int(pad/2))
    #     src_input_tensor = padding(src_input_tensor)
    #     src_ref_tensor = padding(src_ref_tensor)
    # elif src_input_tensor.size()[-1]>trg_input_tensor.size()[-1]:
    #     raise NotImplementedError('crop the src_tensor')

    for srcs, trgs in zip(src_loader, trg_loader):

        src_input_tensor, src_ref_tensor = srcs[0].to(opt.device), srcs[1].to(opt.device)
        trg_input_tensor, trg_ref_tensor = trgs[0].to(opt.device), trgs[1].to(opt.device)

        #choose noise
        num_noise_modes = len(opt.noise)
        noise = opt.noise[random.randint(0,num_noise_modes-1)]

        #set parameter index (for mayo 1mm-0, 3mm-1, else-0)
        param_idx = 1 if opt.thickness==3 else 0 
        trg_noise = make_noise(opt, trg_input_img, noise=noise, pidx=param_idx, scale_max=opt.scale_max, scale_min=opt.scale_min)
        trg_noise_tensor = torch.from_numpy(trg_noise.reshape(1,1,trg_noise.shape[0], trg_noise.shape[1])).to(opt.device)
        trg_noise_tensor = trg_noise_tensor.type(torch.FloatTensor).to(opt.device)

        net.denoiser.train()
        net.domain_discriminator.train()

        #domain classifier
        optimizer_dc.zero_grad()
        net.domain_discriminator.zero_grad()
        dc_loss = net.dc_loss(src_input_tensor, src_ref_tensor, trg_input_tensor)
        dc_loss.backward()
        optimizer_dc.step()
            
        #denoiser
        optimizer.zero_grad()
        net.denoiser.zero_grad()
        loss, l_loss, p_loss, rev_loss = net.g_loss(src_input_tensor, trg_input_tensor, src_ref_tensor, perceptual=True, trg_noise=trg_noise_tensor, return_losses=True)
        loss.backward()
        optimizer.step()

    return net, optimizer, optimizer_dc
    # trg_out_tensor = net.trg_out
    # _, n_psnr, n_ssim,  _, psnr, ssim = calc_metrics(trg_input_tensor, trg_out_tensor, trg_ref_tensor)
    
    # return trg_out_tensor, net, optimizer, optimizer_dc, n_psnr, n_ssim,  psnr, ssim



def make_noise(opt,img, noise='p', pidx=0, scale_max=3, scale_min=0.5):
        scale = random.randint(scale_min*2,scale_max*2)/2
        sigma_est = np.mean(estimate_sigma(img, multichannel=False))
        if noise=='p':
            params = opt.p_lam
            nimg = np.random.poisson(params[pidx]*img)/float(params[pidx])
            nimg = img + scale*(nimg-img)
        elif noise=='g':
            params = opt.g_std
            # noise = np.random.normal(loc=0, scale=params[pidx], size=img.shape).astype(float)
            noise = np.random.normal(loc=0, scale=sigma_est*3*opt.std_scale, size=img.shape).astype(float)
            nimg = img + scale*noise
        elif noise=='bf':
            params = opt.b_dcs
            clean = cv2.bilateralFilter(img, int(params[0]), sigma_est*3*opt.std_scale, params[2])
            noise = img-clean
            if params[1]<0.1:
                nimg = img + noise/params[1]/10 #amplify noise.. 0.1-> 1, 0.05->2, 0.01->10
            else : 
                nimg = img + noise
        elif noise=='nlm':
            clean = denoise_nl_means(img, h=3*sigma_est*opt.std_scale, fast_mode=True, 
                                    patch_size=5, patch_distance=13, multichannel=False)
            noise = img-clean
            nimg = img + scale*noise
        return nimg
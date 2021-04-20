import os, cv2
import glob
import random
import pickle

from skimage.io import imsave, imread
from skimage.external.tifffile import imsave as t_imsave
from skimage.external.tifffile import imread as t_imread
from skimage.restoration import denoise_nl_means, estimate_sigma

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data

class PatchData(data.Dataset):
    def __init__(self, opt, name='', mode='train', add_noise=None, benchmark=False):
        self.opt = opt
        self.dataset = name
        self.in_mem = opt.in_mem

        self.n_channels = opt.n_channels

        self.mode = mode
        self.benchmark = benchmark

        self.add_noise = True if add_noise else False
        self.noise = opt.noise
        self.scale_max = opt.scale_max # max noise scale
        self.scale_min = opt.scale_min # min noise scale

        print("Set file system for dataset {}".format(self.dataset))
        self._set_filesystem(opt.data_dir)
        # print("apath:", os.path.abspath(self.apath))
        # print("dir_hr:", os.path.abspath(self.dir_hr))
        # print("dir_lr:", os.path.abspath(self.dir_lr))

        if opt.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        if opt.use_pt : 
            self.images_hr, self.images_lr = [], []
            self.dir_hr = os.path.join(path_bin, self.dir_hr.split('/')[-1])
            self.dir_lr = os.path.join(path_bin, self.dir_lr.split('/')[-1])
            self.ext = ('.pt', '.pt')

            self.images_hr, self.images_lr = self._scan()
            print("[*]RESET FILE SYSTEM to PT FOLDER")
            print("dir_hr:", os.path.abspath(self.dir_hr))
            print("dir_lr:", os.path.abspath(self.dir_lr))
            print('image length hr {} lr {}'.format(len(self.images_hr), len(self.images_lr)))

            if self.in_mem:
                self._load2mem()
        else : 
            list_hr, list_lr = self._scan()

            if opt.ext.find('img') >= 0 or benchmark:
                self.images_hr, self.images_lr = list_hr, list_lr
            elif opt.ext.find('sep') >= 0:
                self.images_hr, self.images_lr = [], []
                for h in list_hr:
                    b = h.replace(self.apath, path_bin)
                    os.makedirs(os.path.dirname(b), exist_ok=True)

                    b = b.replace(self.ext[0], '.pt')
                    self.images_hr.append(b)
                    self._check_and_load(opt.ext, h, b, verbose=True) 
                for l in list_lr:
                    b = l.replace(self.apath, path_bin)
                    os.makedirs(os.path.dirname(b), exist_ok=True)

                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr.append(b)
                    self._check_and_load(opt.ext, l, b, verbose=True)

                if self.in_mem:
                    self._load2mem()
            
        if mode == 'train':
            n_patches = opt.batch_size * opt.test_every #test_every : # of images per each epoch
            n_images = len(opt.train_datasets) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                # self.repeat = max(n_patches // n_images, 1)
                self.repeat = n_patches / n_images

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[1]))
        )

        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, self.mode, self.dataset)
        self.dir_hr = os.path.join(self.apath, 'hr')
        self.dir_lr = os.path.join(self.apath, 'lr')
        
    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                # if self.opt.n_channels == 1:
                #     pickle.dump(t_imread(img), f)
                # else : 
                #     pickle.dump(imread(img), _f)
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        if not self.in_mem:
            lr, hr, filename = self._load_file(idx)
        else:
            lr, hr, filename = self._load_mem(idx)
        pair = self.get_patch(lr, hr)
        # pair = common.set_channel(*pair, n_channels=self.opt.n_colors)
        if self.n_channels == 3:
            pair = [(p / 255.0) for p in pair]
        
        pair_t = common.np2Tensor(*pair, n_channels=self.n_channels)
        if self.add_noise :
            #choose noise
            num_noise_modes = len(self.noise)
            noise = self.noise[random.randint(0,num_noise_modes-1)]
            #set parameter index (for mayo 1mm-0, 3mm-1, else-0)
            # if noise == 'sp':
            #     nfilename = filename.replace('full', 'qquarter')
            #     nimg = imageio.imread(nfilename)
            param_idx = 1 if '3mm' in filename else 0 
            nimg = self.make_noise(pair[0], noise=noise, pidx=param_idx, scale_max=self.scale_max, scale_min=self.scale_min)
            ntensor = common.np2Tensor(nimg, n_channels=self.n_channels)
            return pair_t[0], pair_t[1], ntensor[0], filename
        else : 
            return pair_t[0], pair_t[1], filename
            
    def __len__(self):
        if self.mode == 'train':
            return int(len(self.images_hr) * self.repeat)
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.mode == 'train':
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.opt.ext == 'img' or self.benchmark:
            # if self.opt.n_channels == 1 :
            #     hr = t_imread(f_hr)
            #     lr = t_imread(f_lr)
            # else : 
            #     hr = imread(f_hr)
            #     lr = imread(f_lr)
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)

        elif self.opt.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)
                
        hr = np.asarray(hr)
        lr = np.asarray(lr)
            
        return lr, hr, filename

    def _load_mem(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[idx]
        hr = self.images_hr[idx]
        filename = self.filename_list[idx]

        return lr, hr, filename

    def _load2mem(self):
        images_hr_list = []
        images_lr_list = []
        self.filename_list = []
        for f_hr, f_lr in zip(self.images_hr, self.images_lr):
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)
            images_hr_list.append(hr)
            images_lr_list.append(lr)
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            self.filename_list.append(filename)

        self.images_hr = images_hr_list
        self.images_lr = images_lr_list

    def get_patch(self, lr, hr): # h,w,c
        scale = 1
        if self.mode == 'train':
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.opt.patch_size,
                n_channels=self.n_channels
            )
            if self.opt.augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        # if self.add_noise:
        #     lr = common.add_noise(lr, self.noise)

        return lr, hr

    def make_noise(self, img, noise='p', pidx=0, scale_max=3, scale_min=0.5):
        scale = random.randint(scale_min*2,scale_max*2)/2
        sigma_est = np.mean(estimate_sigma(img, multichannel=False))
        if noise=='p':
            if scale == 0.5:
                p_scale=4
            elif scale == 1:
                p_scale=1
            elif scale == 1.5:
                p_scale=0.5
            elif scale == 2:
                p_scale=0.28
            elif scale == 2.5:
                p_scale=0.18
            elif scale == 3:
                p_scale=0.12
            else:
                raise NotImplementedError('--ratio_std must be one of the [0.5, 1, 1.5, 2, 2.5, 3]')
            params = self.opt.p_lam
            nimg = np.random.poisson(params[pidx]*p_scale*img)/float(params[pidx]*p_scale)
        elif noise=='g':
            params = self.opt.g_std
            noise = np.random.normal(loc=0, scale=scale*params[pidx], size=img.shape).astype(float)
            # noise = np.random.normal(loc=0, scale=scale*sigma_est*self.opt.ratio_std, size=img.shape).astype(float)
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
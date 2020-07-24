import os
import glob
import h5py

import numpy as np
import torch
import torch.utils.data as data

from data.common import augment, is_image_file, load_img
from data.patchdata import PatchData
from data import common

class LPMAYO(PatchData):
    def __init__(self, args, name='lp-mayo', mode='train', benchmark=False):
        super(LPMAYO, self).__init__(
            args, name=name, mode=mode, benchmark=benchmark
        )
        # LPMAYO specific
        

    def _scan(self):

        names_hr = []
        names_lr = []
        for bp in self.body_part : 
            names_hr.extend(glob.glob(os.path.join(self.dir_hr, '{}*'.format(bp), '*' + self.ext[0])))
            names_lr.extend(glob.glob(os.path.join(self.dir_lr, '{}*'.format(bp), '*' + self.ext[1])))
        
        sorted(names_hr)
        sorted(names_lr)

        # print('names_hr : ',names_hr)
        # print('names_lr : ',names_lr)
        # names_hr = sorted(
        #     glob.glob(os.path.join(self.dir_hr, '*L*', '*' + self.ext[0]))
        # )
        # names_lr = sorted(
        #     glob.glob(os.path.join(self.dir_lr, '*L*', '*' + self.ext[1]))
        # )
        # print(type(names_hr))
        # print(type(names_lr))

        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        super(LPMAYO, self)._set_filesystem(data_dir)

        self.dir_hr = os.path.join(self.apath, 'full')
        self.dir_lr = os.path.join(self.apath, 'low')
        self.ext = ('.tiff', '.tiff')



class LPMayoDataset(data.Dataset):
    def __init__(self, opt):
        super(LPMayoDataset, self).__init__()

        self.use_npy = opt.use_npy

        print(opt.train_dir)
        base_dir = opt.train_dir
        low_path = os.path.join(base_dir, low_opt)
        high_path = os.path.join(base_dir, high_opt)
        
        if self.use_npy:
            low_path = low_path + ".npy"
            high_path = high_path + ".npy"

        self.dsets = {}
        if self.use_npy:
            self.dsets['low'] = np.load(low_path)
            self.dsets['high'] = np.load(high_path)
        else:
            self.dsets['low'] = [os.path.join(low_path, x) for x in os.listdir(low_path) if is_image_file(x)]
            self.dsets['high'] = [os.path.join(high_path, x) for x in os.listdir(high_path) if is_image_file(x)]

    def __getitem__(self, idx):
        if self.use_npy:
            input = self.dsets['low'][idx]
            target = self.dsets['high'][idx]
        else:
            input = load_img(self.dsets['low'][idx])
            target = load_img(self.dsets['high'][idx])

        if len(target.shape) == 2:
            input = input.reshape(1, input.shape[0], input.shape[1])
            target = target.reshape(1, target.shape[0], target.shape[1])
        else:
            input = np.transpose(input, (2, 0, 1))
            target = np.transpose(target, (2, 0, 1))

        input = torch.from_numpy(input).type(torch.FloatTensor)
        target = torch.from_numpy(target).type(torch.FloatTensor)

        return input, target

    def __len__(self):
        return len(self.dsets['high'])

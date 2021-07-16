import os
import glob
import h5py
import random

import numpy as np
import torch
import torch.utils.data as data

from data.patchdata import PatchData
from data import common

class MayoSyn(PatchData):
    def __init__(self, args, name='mayo', mode='train', add_noise=None, benchmark=False):
        self.thickness = args.thickness
        name='mayo'
        super(MayoSyn, self).__init__(
            args, name=name, mode=mode, add_noise=add_noise, benchmark=benchmark
        )
        # MayoSyn specific
        

    def _scan(self):
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '**', '*' + self.ext[0]))
        )
        names_hr = [path.replace('qquarter', 'quarter') for path in names_lr]
        return names_hr, names_lr #qquarter, quarter pair

    def _scan_sp(self):
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '**', '*' + self.ext[0]))
        )
        names_hr = [path.replace('qquarter', 'quarter') for path in names_lr]
        return names_hr, names_lr #qquarter, quarter pair

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

        # self.noise += ['sp']
        num_noise_modes = len(self.noise)
        noise = self.noise[random.randint(0,num_noise_modes-1)]
        if 'sp' in noise : 
            return pair_t[0], pair_t[1], filename #quarter + sinogram poisson
        else : #self.add_noise (not 'sp')
            param_idx = 1 if '3mm' in filename else 0 
            nimg = self.make_noise(pair[1], noise=noise, pidx=param_idx, scale_max=self.scale_max, scale_min=self.scale_min)
            ntensor = common.np2Tensor(nimg, n_channels=self.n_channels)
            return ntensor[0], pair_t[1], filename #quarter + noise


    def _set_filesystem(self, data_dir):
        super(MayoSyn, self)._set_filesystem(data_dir)
        self.ext = ('.tiff', '.tiff')

        if self.thickness == 0:
            quarter_dose = 'quarter_*mm'
            qquarter_dose = 'qquarter_*mm'
            self.dir_hr = os.path.join(self.apath, quarter_dose)
            self.dir_lr = os.path.join(self.apath, qquarter_dose)
        else:
            quarter_dose = 'quarter_{}mm'.format(self.thickness)
            qquarter_dose = 'qquarter_{}mm'.format(self.thickness)
            self.dir_hr = os.path.join(self.apath, quarter_dose)
            self.dir_lr = os.path.join(self.apath, qquarter_dose)
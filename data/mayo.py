import os
import glob
import h5py
import random
import imageio

import numpy as np
import torch
import torch.utils.data as data

from data.patchdata import PatchData
from data import common

class Mayo(PatchData):
    def __init__(self, args, name='mayo', mode='train', add_noise=None, benchmark=False):
        self.thickness = args.thickness
        super(Mayo, self).__init__(
            args, name=name, mode=mode, add_noise=add_noise, benchmark=benchmark
        )
        # Mayo specific
        
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '**', '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '**', '*' + self.ext[1]))
        )

        return names_hr, names_lr

    def _scan_sp(self):
        dir_nlr = self.dir_hr.replace('full', 'qquarter')

        names_nlr = sorted(
            glob.glob(os.path.join(dir_nlr, '**', '*' + self.ext[0]))
        )
        print('names_nlr num: {}'.format(len(names_nlr)))
        names_hr = [path.replace('qquarter', 'full') for path in names_nlr]
        names_lr = [path.replace('qquarter', 'quarter') for path in names_nlr]
        return names_hr, names_lr

    def __getitem__(self, idx):
        #sinogram poisson dataset
        if not self.in_mem:
            lr, hr, filename = self._load_file(idx)
        else:
            lr, hr, filename = self._load_mem(idx)
        if self.add_noise:
            num_noise_modes = len(self.noise)
            ridx=random.randint(0,num_noise_modes-1)
            noise = self.noise[ridx]
            sp_filename=filename.replace('full', 'qquarter')
            if noise == 'sp' and os.path.exists(sp_filename):
                nlr = imageio.imread(sp_filename)
                pair = self.get_patch(lr, hr, nlr)
                if self.n_channels == 3:
                    pair = [(p / 255.0) for p in pair]
                pair_t = common.np2Tensor(*pair, n_channels=self.n_channels)
                return pair_t[0], pair_t[1], pair_t[2], filename
            else:
                pair = self.get_patch(lr, hr)
                if self.n_channels == 3:
                    pair = [(p / 255.0) for p in pair]
                pair_t = common.np2Tensor(*pair, n_channels=self.n_channels)

                if noise == 'sp' : noise=self.noise[ridx-1] #sp file doesn't exist --> other noise
                param_idx = 1 if '3mm' in filename else 0 
                nimg = self.make_noise(pair[0], noise=noise, pidx=param_idx, scale_max=self.scale_max, scale_min=self.scale_min)
                ntensor = common.np2Tensor(nimg, n_channels=self.n_channels)
                return pair_t[0], pair_t[1], ntensor[0], filename
        else:
            pair = self.get_patch(lr, hr)
            if self.n_channels == 3:
                pair = [(p / 255.0) for p in pair]
            pair_t = common.np2Tensor(*pair, n_channels=self.n_channels)
            return pair_t[0], pair_t[1], filename

    def _set_filesystem(self, data_dir):
        super(Mayo, self)._set_filesystem(data_dir)
        # self.apath = self.apath.replace('train', 'test')
        self.ext = ('.tiff', '.tiff')

        if self.thickness == 0:
            full_dose = 'full_*mm'
            quarter_dose = 'quarter_*mm'
            self.dir_hr = os.path.join(self.apath, full_dose)
            self.dir_lr = os.path.join(self.apath, quarter_dose)
        else:
            full_dose = 'full_{}mm'.format(self.thickness)
            quarter_dose = 'quarter_{}mm'.format(self.thickness)
            self.dir_hr = os.path.join(self.apath, full_dose)
            self.dir_lr = os.path.join(self.apath, quarter_dose)
    
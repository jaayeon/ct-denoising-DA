import os
import glob
import h5py

import numpy as np
import torch
import torch.utils.data as data

from data.patchdata import PatchData
from data import common

class PigletSyn(PatchData):
    def __init__(self, args, name='piglet', mode='train', add_noise=None, benchmark=False):
        super(PigletSyn, self).__init__(
            args, name=name, mode=mode, add_noise=add_noise, benchmark=benchmark
        )
        # PIGLET specific
        

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '**', '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '**', '*' + self.ext[1]))
        )

        return names_hr, names_lr

    def __getitem__(self, idx):
        if not self.in_mem:
            lr, hr, filename = self._load_file(idx)
        else:
            lr, hr, filename = self._load_mem(idx)

        if self.n_channels == 3:
            pair = [(p/255.0) for p in pair]
        
        pair_t = common.np2Tensor(*pair, n_channels=self.n_channels)
        if self.add_noise:
            num_noise_modes=len(self.noise)
            noise = self.noise[random.randint(0, num_noise_modes-1)]
            param_idx=1

    def _set_filesystem(self, data_dir):
        super(PIGLET, self)._set_filesystem(data_dir)

        self.dir_hr = os.path.join(self.apath, 'Oten')
        self.dir_lr = os.path.join(self.apath, 'Oten')
        self.ext = ('.tiff', '.tiff')



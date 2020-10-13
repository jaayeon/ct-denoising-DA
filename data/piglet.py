import os
import glob
import h5py

import numpy as np
import torch
import torch.utils.data as data

from data.patchdata import PatchData
from data import common

class PIGLET(PatchData):
    def __init__(self, args, name='piglet', mode='train', domain_sync=None, benchmark=False):
        super(PIGLET, self).__init__(
            args, name=name, mode=mode, domain_sync=domain, benchmark=benchmark
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

    def _set_filesystem(self, data_dir):
        super(PIGLET, self)._set_filesystem(data_dir)

        self.dir_hr = os.path.join(self.apath, 'full')
        self.dir_lr = os.path.join(self.apath, 'Oten')
        self.ext = ('.tiff', '.tiff')



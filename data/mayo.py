import os
import glob
import h5py

import numpy as np
import torch
import torch.utils.data as data

from data.common import augment, is_image_file, load_img
from data.patchdata import PatchData
from data import common

class Mayo(PatchData):
    def __init__(self, args, name='mayo', mode='train', benchmark=False):
        self.thickness = args.thickness
        super(Mayo, self).__init__(
            args, name=name, mode=mode, benchmark=benchmark
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

    def _set_filesystem(self, data_dir):
        super(Mayo, self)._set_filesystem(data_dir)

        # full_dose = 'full_{}mm'.format(self.thickness)
        # quarter_dose = 'quarter_{}mm'.format(self.thickness)
        full_dose = 'full_*mm'
        quarter_dose = 'quarter*mm'

        self.dir_hr = os.path.join(self.apath, full_dose)
        self.dir_lr = os.path.join(self.apath, quarter_dose)
        self.ext = ('.tiff', '.tiff')

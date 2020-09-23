import os
import glob
import h5py

import numpy as np
import torch
import torch.utils.data as data

from data.patchdata import PatchData
from data import common

class LPMAYO(PatchData):
    def __init__(self, args, name='lp-mayo', mode='train', domain_sync=None, benchmark=False):
        self.body_part = args.body_part
        super(LPMAYO, self).__init__(
            args, name=name, mode=mode, domain_sync=domain_sync, benchmark=benchmark
        )
        # LPMAYO specific
<<<<<<< HEAD
        
        
=======
>>>>>>> jeeylee

    def _scan(self):

        names_hr = []
        names_lr = []
        print('body part : ',self.body_part)
        for bp in self.body_part : 
            names_hr.extend(glob.glob(os.path.join(self.dir_hr, '{}*'.format(bp), '*' + self.ext[0])))
            names_lr.extend(glob.glob(os.path.join(self.dir_lr, '{}*'.format(bp), '*' + self.ext[1])))
        
        names_hr = sorted(names_hr)
        names_lr = sorted(names_lr)

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

        if self.domain_sync == 'ref2trg':
            self.dir_hr = os.path.join(self.apath, 'fake_full')
            self.dir_lr = os.path.join(self.apath, 'fake_low')
        elif self.domain_sync == 'out2src':
            self.dir_hr = os.path.join(self.apath, 'full')
            self.dir_lr = os.path.join(self.apath, 'fake_low')
        else : #self.domain_sync = 'None'
            self.dir_hr = os.path.join(self.apath, 'full')
            self.dir_lr = os.path.join(self.apath, 'low')
        self.ext = ('.tiff', '.tiff')

        print('[**] Set File System : \ndir_hr {} \ndir_lr {}'.format(self.dir_hr, self.dir_lr))




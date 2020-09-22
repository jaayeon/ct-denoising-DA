import os
import glob
import h5py

import numpy as np
import torch
import torch.utils.data as data

from data.patchdata import PatchData
from data import common

class PHANTOM(PatchData):
    def __init__(self, args, name='siemens', mode='train', domain_sync=None, benchmark=False):
        self.anatomy = args.anatomy
        self.mA_full = args.mA_full
        self.mA_low = args.mA_low
        super(PHANTOM, self).__init__(
            args, name=name, mode=mode, domain_sync=domain_sync, benchmark=benchmark
        )
        # PHANTOM specific


        
    def _scan(self):

        names_hr = []
        names_lr = []
        print('anatomy : ',self.anatomy)
        names_hr.extend(glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])))
        names_lr.extend(glob.glob(os.path.join(self.dir_lr, '*' + self.ext[1])))
        
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
        super(PHANTOM, self)._set_filesystem(data_dir)
        self.apath = os.path.join(data_dir, self.mode, 'phantom', self.dataset, self.anatomy)

        if self.domain_sync == 'ref2trg':
            self.dir_hr = os.path.join(self.apath, 'fake_{}*'.format(self.mA_full))
            self.dir_lr = os.path.join(self.apath, 'fake_{}*'.format(self.mA_low))
        elif self.domain_sync == 'out2src':
            self.dir_hr = os.path.join(self.apath, '{}*'.format(self.mA_full))
            self.dir_lr = os.path.join(self.apath, 'fake_{}*'.format(self.mA_low))
        else : #self.domain_sync = 'None'
            self.dir_hr = os.path.join(self.apath, '{}*'.format(self.mA_full))
            self.dir_lr = os.path.join(self.apath, '{}*'.format(self.mA_low))    
        self.ext = ('.tiff', '.tiff')

        print('[**] Set File System : \ndir_hr {} \ndir_lr {}'.format(self.dir_hr, self.dir_lr))




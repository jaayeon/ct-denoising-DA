import os
import glob
import h5py

import numpy as np
import torch
import torch.utils.data as data

from data.patchdata import PatchData
from data import common

class PHANTOM_T(PatchData):
    def __init__(self, args, name='phantom_t', mode='train', domain=None, benchmark=False):
        self.anatomy = args.anatomy
        self.mA_full = args.mA_full
        self.mA_low = args.mA_low
        self.target_vendor = args.target_vendor
        super(PHANTOM_T, self).__init__(
            args, name=name, mode=mode, domain=domain, benchmark=benchmark
        )
        # PHANTOM specific
        

        
    def _scan(self):

        names_hr = []
        names_lr = []
        print('anatomy : ',self.anatomy)
        for ap in self.anatomy : 
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
        super(PHANTOM_T, self)._set_filesystem(data_dir)

        anatomy = self.anatomy
        target_vendor = self.target_vendor
        full = self.mA_full
        low = self.mA_low
        
        apath = self.apath
        apath = apath[0:-2]

        if self.domain == 'ref2trg':
            self.dir_hr = os.path.join(self.apath, 'fake_full')
            self.dir_lr = os.path.join(self.apath, 'fake_low')
        elif self.domain == 'out2src':
            self.dir_hr = os.path.join(self.apath, 'full')
            self.dir_lr = os.path.join(self.apath, 'fake_low')
        else : #self.domain = 'None'
            self.dir_hr = os.path.join(apath, target_vendor, anatomy, full)
            self.dir_lr = os.path.join(apath, target_vendor, anatomy, low)    
        self.ext = ('.tiff', '.tiff')

        print('[**] Set File System : \ndir_hr {} \ndir_lr {}'.format(self.dir_hr, self.dir_lr))



import os
import glob
import h5py

import numpy as np
import torch
import torch.utils.data as data

from data.patchdata import PatchData
from data import common

class PHANTOM(PatchData):
    def __init__(self, args, name='phantom', mode='train', domain=None, benchmark=False):
        super(PHANTOM, self).__init__(
            args, name=name, mode=mode, domain=domain, benchmark=benchmark
        )
        # PHANTOM specific

        
    # 여기 수정해야 할 것 같음 같같같 왜 안되지?
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
        super(PHANTOM, self)._set_filesystem(data_dir)

        anatomy = self.anatomy

        # phantom_siemens
        if anatomy == 'chest':
            mA = ['180', '120', '60', '30']
            full = '180'
            low = '60'
        elif anatomy == 'hn':
            mA = ['180', '120', '60', '40']
            full = '180'
            low = '60'
        elif anatomy == 'pelvis':
            mA = ['200', '150', '100', '50']
            full = '200'
            low = '100'
        
        if self.domain == 'ref2trg':
            self.dir_hr = os.path.join(self.apath, 'fake_full')
            self.dir_lr = os.path.join(self.apath, 'fake_low')
        elif self.domain == 'out2src':
            self.dir_hr = os.path.join(self.apath, 'full')
            self.dir_lr = os.path.join(self.apath, 'fake_low')
        else : #self.domain = 'None'
            self.dir_hr = os.path.join(self.apath, 'siemens', anatomy, full)
            self.dir_lr = os.path.join(self.apath, 'siemens', anatomy, low)
        self.ext = ('.tiff', '.tiff')

        print('[**] Set File System : \ndir_hr {} \ndir_lr {}'.format(self.dir_hr, self.dir_lr))




import os
import glob
import h5py

import numpy as np
import torch
import torch.utils.data as data

from data.patchdata import PatchData
from data import common

class PHANTOM(PatchData):
    def __init__(self, args, name='siemens', mode='train', add_noise=None, benchmark=False):
        self.anatomy = args.anatomy
        self.mA_full = args.mA_full
        self.mA_low = args.mA_low
        super(PHANTOM, self).__init__(
            args, name=name, mode=mode, add_noise=add_noise, benchmark=benchmark
        )
        # PHANTOM specific


        
    def _scan(self):

        names_hr = []
        names_lr = []
        print('anatomy : ',self.anatomy)
        
        for dh, dl in zip(self.dir_hr,self.dir_lr):
            names_hr.extend(glob.glob(os.path.join(dh, '*' + self.ext[0])))
            names_lr.extend(glob.glob(os.path.join(dl, '*' + self.ext[1])))
        """ 
        names_hr.extend(glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])))
        names_lr.extend(glob.glob(os.path.join(self.dir_lr, '*' + self.ext[1])))
        """
        names_hr = sorted(names_hr)
        names_lr = sorted(names_lr)

        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        super(PHANTOM, self)._set_filesystem(data_dir)
        # self.apath = os.path.join(data_dir, self.mode, 'phantom', self.dataset, self.anatomy)
        print('[**] Set File System :')
        
        self.dir_hr = []
        self.dir_lr = []
        for at in self.anatomy: 
            self.apath = os.path.join(data_dir, self.mode, 'phantom', self.dataset)
            self.dir_hr.append(os.path.join(self.apath, at, '{}*_crop320'.format(self.mA_full)))
            self.dir_lr.append(os.path.join(self.apath, at, '{}*_crop320'.format(self.mA_low)))    
        self.ext = ('.tiff', '.tiff')

        for i in range(len(self.dir_hr)):
            print('dir_hr {} \ndir_lr {}'.format(self.dir_hr[i], self.dir_lr[i]))
       

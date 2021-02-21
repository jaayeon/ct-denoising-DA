import os
import glob
import h5py
import copy
import numpy as np
import torch
import torch.utils.data as data
from data.patchdata import PatchData
from data import common

class Mayo(PatchData):
    def __init__(self, args, name='mayo', mode='train', domain_sync=None, benchmark=False):
        self.thickness = args.thickness
        super(Mayo, self).__init__(
            args, name=name, mode=mode, domain_sync=domain_sync, benchmark=benchmark)
            
    def get_noisy_noisy_image_with_noise(self,noise_np, img_np, no_clip= True):
        if not no_clip:
            img_noisy_np = np.clip(img_np + noise_np, 0, 1).astype(np.float32)
            #img_noisy_pil = np_to_pil(img_noisy_np)
            img_noisy_noisy_np = np.clip(img_np + noise_np*2, 0, 1).astype(np.float32)
            #img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)
        elif no_clip:
            img_noisy_np = (img_np + noise_np).astype(np.float32)
            #img_noisy_pil = np_to_pil(img_noisy_np)
            img_noisy_noisy_np = (img_noisy_np + noise_np).astype(np.float32)
            #img_noisy_noisy_pil = np_to_pil(img_noisy_noisy_np)

        return img_noisy_np, img_noisy_noisy_np

    def generate_mask(self, input):
        ratio = 0.85
        size_window = (7,7)
        size_data = (self.args.patch_size,self.args.patch_size)
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio)) 

        mask = np.ones(size_data)
        output = input

        for ich in range(1):
            idy_msk = np.random.randint(0, size_data[0], num_sample)
            idx_msk = np.random.randint(0, size_data[1], num_sample)    ##0 to 64 sample num 409

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample) ##-32 to 32 num 409
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample) ##-32 to 32 num 409

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

            id_msk = (idy_msk, idx_msk)
            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)

            #print(len(id_msk))
            #print(len(id_msk_neigh))

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0

        return output, mask  

    def __getitem__(self, idx):
        if not self.in_mem:
            lr, hr, idx = self._load_file(idx)         
            # print(len(lr))
            # print(len(hr))
        else:
            lr, hr = self._load_mem(idx)
        pair = self.get_patch(lr, hr)
        # pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        if self.n_channels == 3:
            pair = [(p / 255.0) for p in pair]

        label = pair[0] 
        #print(pair[0].shape)

        if self.args.way == 'n2v':
            input, mask = self.generate_mask(copy.deepcopy(label))
            pro_data = common.np2Tensor(input, label, mask, n_channels=self.n_channels)
            clean = common.np2Tensor(pair[1], n_channels=self.n_channels)
            input, label, mask = pro_data[0], pro_data[1], pro_data[2] 
            proc_data = {'label': label, 'input': input, 'mask': mask, 'clean': clean[0]}

            return proc_data


        elif self.args.way == 'n2c':
            #TRAIN_PLAN = [5/255., 10/255., 15/255., 20/255., 25/255.] 
            #num = int(int(idx) % 5)
            #sigma_now = TRAIN_PLAN[num]

            sigma_now = 5/255.
            noisy_np_norm = np.random.normal(0.0, 1.0, size= (80,80))
            noisy_np = noisy_np_norm * (sigma_now)
            #print(pair[0].shape)
            noisy, input = self.get_noisy_noisy_image_with_noise(noisy_np, label)
            pro_data = common.np2Tensor(noisy, input,  n_channels=self.n_channels)
            clean = common.np2Tensor(pair[1], n_channels=self.n_channels)
            noisy, input = pro_data[0], pro_data[1]
            proc_data = {'input': input, 'noisy': noisy, 'clean': clean[0]}

            return proc_data


        else:
            pair_t = common.np2Tensor(*pair, n_channels=self.n_channels)
            # print(pair_t[0].shape)
            # print(pair_t[1].shape)
            return pair_t[0], pair_t[1]


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

        if self.thickness == 0:
            full_dose = 'full_*mm'
            quarter_dose = 'quarter_*mm'
            self.dir_hr = os.path.join(self.apath, full_dose)
            self.dir_lr = os.path.join(self.apath, quarter_dose)
            self.ext = ('.tiff', '.tiff')
        else:
            full_dose = 'full_{}mm'.format(self.thickness)
            quarter_dose = 'quarter_{}mm'.format(self.thickness)
            self.dir_hr = os.path.join(self.apath, full_dose)
            self.dir_lr = os.path.join(self.apath, quarter_dose)
            self.ext = ('.tiff', '.tiff')
        # full_dose = 'full_*mm'
        # quarter_dose = 'quarter*mm'

        # self.dir_hr = os.path.join(self.apath, full_dose)
        # self.dir_lr = os.path.join(self.apath, quarter_dose)
        # self.ext = ('.tiff', '.tiff')

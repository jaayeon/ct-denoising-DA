""" 
from skimage.external.tifffile import imsave, imread

import numpy as np
import torch
import torch.utils.data as data

from data.make_patches import make_patches, pad_img, unpad_img

from data.common import augment

class ImageDataset(data.Dataset):
    def __init__(self, opt, img):
        super(ImageDataset, self).__init__()

        if opt.n_channels == 3:
            img = img / 255.0

        self.img_shape = img.shape

        self.opt = opt

        padded_img = pad_img(img, opt.patch_size, opt.patch_offset)
        self.pad_img_shape =padded_img.shape
        self.img_patches = make_patches(padded_img, opt.patch_size, opt.patch_offset)

        patches_dims = self.img_patches.shape

        if opt.n_channels == 1:
            self.img_patches = self.img_patches.reshape(patches_dims[0], opt.n_channels, patches_dims[1], patches_dims[2])
        else : 
            self.img_patches = self.img_patches.transpose((0,3,1,2))
        # print(self.img_patches.shape)

    def __getitem__(self, idx):
        patch = self.img_patches[idx]

        if self.opt.model == 'ffdnet':
            sigma_test = 25
            np.random.seed(seed = self.opt.seed)

            # patch += np.random.normal(0, sigma_test, patch.shape)
            # noise_level = torch.FloatTensor([sigma_test])
            patch += np.random.normal(0, sigma_test/(255.0*255.0), patch.shape)
            noise_level = torch.FloatTensor([sigma_test/(255.0*255.0)])
            # patch += np.random.normal(0, sigma_test/(255.0), patch.shape)
            # noise_level = torch.FloatTensor([sigma_test/(255.0)])

            noise_level = noise_level.unsqueeze(1).unsqueeze(1)
            patch = torch.from_numpy(patch).type(torch.FloatTensor)
            
            return patch, noise_level
        else : 
            patch = torch.from_numpy(patch).type(torch.FloatTensor)
            return patch


    def __len__(self):
        return len(self.img_patches)

    def get_img_shape(self):
        return self.img_shape

    def get_padded_img_shape(self):
        return self.pad_img_shape
 """
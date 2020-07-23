import random
import numpy as np
from skimage.external.tifffile import imsave, imread

import skimage.color as sc

import torch

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".PNG", ".png", ".tiff"])

def load_img(file_path):
    img = imread(file_path)
    # img = normalize(img)
    return img

def add_gaussian_noise(image, mean=0, std=0.001):
    img_dims = image.shape
    
    sigma = std
    gauss = np.random.normal(mean, sigma, img_dims)
    gauss = gauss.reshape(img_dims[0], img_dims[1], img_dims[2])
    # noisy = image + gauss
    noisy = image
    return noisy

# def add_noise(x, noise=None):
#     if noise is not None:
#         noise_type = noise[0]
#         noise_value = int(noise[1:])
#         if noise_type == 'G':
#             noises = np.random.normal(scale=noise_value, size=x.shape)
#             noises = noises.round()
#         elif noise_type == 'S':
#             noises = np.random.poisson(x * noise_value) / noise_value
#             noises = noises - noises.mean(axis=0).mean(axis=0)

#         x_noise = x.astype(np.int16) + noises.astype(np.int16)
#         x_noise = x_noise.clip(0, 255).astype(np.uint8)
#         return x_noise
#     else:
#         return x

def add_noise(x, noise=None):
    if noise == 0:
        noise_value = np.random.randint(25)
    else:
        noise_value = noise

    noises = np.random.normal(scale=noise_value, size=x.shape)
    noises = noises.round()
        
    x_noise = x.astype(np.int16) + noises.astype(np.int16)
    x_noise = x_noise.clip(0, 255).astype(np.uint8)
    return x_noise

def add_blind_noise(x, noise=None):
    noise_value = np.random.randint(25)
    
    noises = np.random.normal(scale=noise_value, size=x.shape)
    noises = noises.round()

    x_noise = x.astype(np.int16) + noises.astype(np.int16)
    x_noise = x_noise.clip(0, 255).astype(np.uint8)

    return x_noise

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if img.ndim == 2:
            if hflip: img = img[:, ::-1].copy()
            if vflip: img = img[::-1, :].copy()
            if rot90: img = img.transpose(1, 0).copy()
        elif img.ndim == 3:
            if hflip: img = img[:, ::-1, :].copy()
            if vflip: img = img[::-1, :, :].copy()
            if rot90: img = img.transpose(1, 0, 2).copy()
            
        return img

    return [_augment(a) for a in args]

def get_patch(*args, patch_size=96, n_channels=1, scale=1, multi=False, input_large=False):

    ih, iw = args[0].shape[:2]

    tp = patch_size
    ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    # n_channel is 7 when swt is enabled on one channel
    if n_channels == 1:
        ret = [
            args[0][iy:iy + ip, ix:ix + ip],
            *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]]
        ]
    else:
        ret = [
            args[0][iy:iy + ip, ix:ix + ip, :],
            *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
        ]

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, n_channels=1, swt=False):
    def _np2Tensor(img):
        if n_channels == 1 and not swt:
            np_transpose = np.expand_dims(img, axis=0)
        elif n_channels == 1 and swt:
            np_transpose = img
        elif n_channels == 3 and not swt:
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        elif n_channels == 3 and swt:
            np_transpose = img
        

        tensor = torch.from_numpy(np_transpose).float()

        # if pixel_range == 255:
        # I normalize images in patchdata.py
        # if n_channels == 3:
        #     tensor.mul_(1 / 255)

        return tensor


    return [_np2Tensor(a) for a in args]

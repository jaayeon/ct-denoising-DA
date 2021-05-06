import os, glob
import time
from shutil import copyfile, move

import numpy as np


def pad_tensor(tensor_img, patch_size, patch_offset):
    # print("tensor_img.shape:", tensor_img.shape)
    stride = patch_size - 2 * patch_offset
    bs, c, h, w = tensor_img.shape

    rw = (patch_offset + w) % stride
    rh = (patch_offset + h) % stride
    w_stride_pad_size = stride - rw
    h_stride_pad_size = stride - rh

    stride_pad_w = patch_offset + w + w_stride_pad_size
    stride_pad_h = patch_offset + h + h_stride_pad_size

    w_pad_size = w_stride_pad_size + patch_size
    h_pad_size = h_stride_pad_size + patch_size

    npad = (patch_offset, h_pad_size, patch_offset, w_pad_size)

    tensor_img = F.pad(tensor_img, npad, mode='reflect')
    # print("padded tensor_img.shape:", tensor_img.shape)

    return tensor_img
    


def unpad_tensor(tensor_img, patch_offset, tensor_img_shape):
    bs, c, h, w = tensor_img_shape
    tensor_ret = tensor_img[:, :, patch_offset:patch_offset+h, patch_offset:patch_offset+w]
    return tensor_ret



def make_tensor_arr_patches(tensor_img, patch_size, patch_offset):
    bs, c, h, w = tensor_img.shape

    assert bs == 1

    stride = patch_size - 2 * patch_offset
    mod_h = h - np.mod(h - patch_size, stride)
    mod_w = w - np.mod(w - patch_size, stride)
    
    num_patches = (mod_h // stride) * (mod_w // stride)

    patch_arr = torch.zeros((num_patches, c, patch_size, patch_size), dtype=tensor_img.dtype)

    ps = patch_size

    patch_idx = 0
    for y in range(0, mod_h - stride + 1, stride):
        for x in range(0, mod_w - stride + 1, stride):
            patch = tensor_img[:, :, y:y+ps, x:x+ps]
            patch_arr[patch_idx] = patch
            patch_idx += 1

    return patch_arr



def recon_tensor_arr_patches(patch_arr, width, height, patch_size, patch_offset):
    stride = patch_size - 2 * patch_offset

    _, c, _, _ = patch_arr.shape

    tesnor_img = torch.zeros((1, c, height, width), dtype=patch_arr.dtype)

    mod_h = height - np.mod(height - 2 * patch_offset, stride)
    mod_w = width - np.mod(width - 2 * patch_offset, stride)

    ps = patch_size
    po = patch_offset

    patch_idx = 0
    for y in range(0, mod_h - (patch_size - patch_offset) + 1, stride):
        for x in range(0, mod_w - (patch_size - patch_offset) + 1, stride):
            patch = patch_arr[patch_idx]
                
            tesnor_img[:, : ,y+po:y+ps-po, x+po:x+ps-po] = patch[:, po:-po, po:-po]
            patch_idx += 1

    return tesnor_img

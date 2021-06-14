from models import networks_rev
from options import args
import glob
import imageio
import torch
import numpy as np

def make_tensor(img):
    tensor = torch.Tensor(img)
    h,w = tensor.size()
    tensor = tensor.reshape([1,1,h,w])
    return tensor

def get_saliency_map(net, img, idx=0):
    _, _, h,w = img.size()
    imgs = make_tensor_arr_patches(img,80,15)
    imgs.requires_grad_()
    output = net(imgs)
    mse = torch.nn.MSELoss()
    loss = mse(output, idx*torch.ones(output.size()))
    loss.backward()
    # out_idx = output.argmax()
    # output_max=output[0, output_idx]
    # output_max.backward()
    saliency_tensors = imgs.grad.data.abs()
    saliency_tensor = recon_tensor_arr_patches(saliency_tensors, w, h, 80, 15)
    saliency = saliency_tensor[0,0,:,:].numpy()
    return saliency


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

opt = args

# checkpoint_path = '../../data/denoising/checkpoint_DA/ge-20210506-0609-rev-edsr-chest-pelvis/edsr_epoch_0171_psnr_35.40038188.pth'
checkpoint_path = '../../data/denoising/checkpoint_DA/ge-20210528-2317-rev-edsr-p-chest-pelvis/edsr_epoch_0190_psnr_35.20759986.pth'
src_img = imageio.imread('../../data/denoising/train/phantom/ge/chest/level5_005_crop/ge_chest_level5_005_192.tiff')
trg_img = imageio.imread('../../data/denoising/test/mayo/quarter_1mm/L067/quarter_1mm-L067-099.tiff')

src_write = './tests/ge_chest_level5_005_192_map.tiff'
trg_write = './tests/quarter_1mm-L067-099_map.tiff'

model = networks_rev.Networks_rev(opt)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'], strict=False)

tensor = make_tensor(src_img)
saliency_map = get_saliency_map(model.domain_discriminator, tensor, idx=1)
imageio.imwrite(src_write, saliency_map)

tensor = make_tensor(trg_img)
saliency_map = get_saliency_map(model.domain_discriminator, tensor, idx=0)
imageio.imwrite(trg_write, saliency_map)


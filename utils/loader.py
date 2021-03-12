import os
import glob
import json
import torch

def select_checkpoint_dir(opt):
    checkpoint_dir = opt.checkpoint_dir
    dirs = os.listdir(checkpoint_dir)
    dirs = sorted(dirs)

    for i, d in enumerate(dirs, 0):
        print("(%d) %s" % (i, d))
    d_idx = input("Select directory that you want to load: ")

    path_opt = dirs[int(d_idx)]
    opt.path_opt = path_opt

    checkpoint_dir = os.path.abspath(os.path.join(checkpoint_dir, dirs[int(d_idx)]))
    print("checkpoint_dir is: {}".format(checkpoint_dir))

    return checkpoint_dir


def load_config(opt):
    batch_size = opt.batch_size
    mode = opt.mode
    ensemble = opt.ensemble
    epoch_num = opt.epoch_num
    resume_best = opt.resume_best
    resume = opt.resume
    patch_size = opt.patch_size
    test_patches = opt.test_patches
    patch_offset = opt.patch_offset
    target = opt.target
    mA_full = opt.mA_full
    mA_low = opt.mA_low
    anatomy = opt.anatomy
    thickness = opt.thickness

    checkpoint_dir = select_checkpoint_dir(opt)
    
    config_file = os.path.join(checkpoint_dir, "config.txt")
    with open(config_file, 'r') as f:
        opt.__dict__ = json.load(f)
    
    opt.checkpoint_dir = checkpoint_dir
    opt.ensemble = ensemble
    opt.batch_size = batch_size
    opt.mode = mode
    opt.epoch_num = epoch_num
    opt.resume_best = resume_best
    opt.resume = resume
    opt.patch_size = patch_size
    opt.test_patches = test_patches
    opt.patch_offset = patch_offset
    opt.target = target
    opt.anatomy = anatomy
    opt.thickness = thickness

    if opt.target == 'lp-mayo':
        opt.gt_img_dir = r'../../data/denoising/test/lp-mayo/full'
        opt.img_dir = r'../../data/denoising/test/lp-mayo/low'
    elif opt.target == 'mayo':
        opt.img_dir = r'../../data/denoising/test/mayo/quarter_{}mm/*'.format(opt.thickness)
        opt.gt_img_dir = r'../../data/denoising/test/mayo/full_{}mm/*'.format(opt.thickness)
    elif opt.target == 'piglet':
        opt.gt_img_dir = r'../../data/denoising/test/piglet/full/*'
        opt.img_dir = r'../../data/denoising/test/piglet/Oten/*'
    else:
        opt.gt_img_dir = r'../../data/denoising/test/phantom/{}/{}/{}*'.format(opt.target, opt.anatomy[0], opt.mA_full)
        opt.img_dir = r'../../data/denoising/test/phantom/{}/{}/{}*'.format(opt.target, opt.anatomy[0], opt.mA_low)
    return opt


def load_model(opt, model, optimizer=None):
    if opt.mode == 'train':
        checkpoint_dir = select_checkpoint_dir(opt)
    elif opt.mode == 'test':
        checkpoint_dir = opt.checkpoint_dir
        print("Use {} director as checkpoint".format(os.path.abspath(opt.checkpoint_dir)))
    elif opt.mode == 'result_sidd' or opt.mode == 'result':
        checkpoint_dir = opt.checkpoint_dir
    else:
        checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.model)
        raise RuntimeError("Please check option to load model")

    # we will check from last, best, and specific epoch_num model
    # checkpoint_list = os.listdir(checkpoint_dir)
    checkpoint_list = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    checkpoint_list.sort()
    n_epoch = 0

    if opt.resume_best:
        loss_list = list(map(lambda x: float(os.path.basename(x).split('_')[4][:-4]), checkpoint_list))
        best_loss_idx = loss_list.index(min(loss_list))
        checkpoint_path = checkpoint_list[best_loss_idx]
    elif opt.epoch_num > 0:
        checkpoint_path = checkpoint_list[opt.epoch_num - 1]
    else:
        # default load the last checkpoint
        checkpoint_path = checkpoint_list[len(checkpoint_list) - 1]

    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        # print(checkpoint.keys())
        n_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        if optimizer is not None:
            for i in range(len(optimizer)):
                optimizer[i].load_state_dict(checkpoint['optimizer'][i])
            # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_path, n_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    print("Using epoch_num:", n_epoch)
    
    opt.checkpoint_dir = checkpoint_dir
    print(model)
    if opt.mode == 'train':
        return n_epoch+1, model, optimizer
    else : 
        if opt.way == 'wgan' or opt.way == 'wganrev':
            return n_epoch+1, model.generator, optimizer
        elif opt.way == 'rev' or opt.way == 'base':
            return n_epoch+1, model.denoiser, optimizer

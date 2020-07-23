import os
import glob
import json
import torch

def save_config(opt):
    config_file = os.path.join(opt.checkpoint_dir, "config.txt")
    with open(config_file, 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

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
    dataset = opt.dataset

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
    opt.dataset = dataset

    if opt.dataset == 'lp-mayo':
        opt.gt_img_dir = r'../../data/denoising/test/lp-mayo/full'
        opt.img_dir = r'../../data/denoising/test/lp-mayo/low'
    elif opt.dataset == 'mayo':
        opt.img_dir = r'../../data/denoising/test/mayo/quarter_{}mm/L506'.format(opt.thickness)
        opt.gt_img_dir = r'../../data/denoising/test/mayo/full_{}mm/L506'.format(opt.thickness)
    elif opt.dataset == 'piglet':
        opt.gt_img_dir = r'../../data/denoising/test/piglet/full'
        opt.img_dir = r'../../data/denoising/test/piglet/quarter'

    return opt


def select_checkpoint_dir(opt):
    checkpoint_dir = opt.checkpoint_dir
    dirs = os.listdir(checkpoint_dir)

    for i, d in enumerate(dirs, 0):
        print("(%d) %s" % (i, d))
    d_idx = input("Select directory that you want to load: ")

    path_opt = dirs[int(d_idx)]
    opt.path_opt = path_opt

    checkpoint_dir = os.path.abspath(os.path.join(checkpoint_dir, dirs[int(d_idx)]))
    print("checkpoint_dir is: {}".format(checkpoint_dir))

    return checkpoint_dir

def save_checkpoint(opt, model, optimizer, epoch, loss):
    # checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.model + '-patch' + str(opt.patch_size))
    checkpoint_dir = opt.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "models_epoch_%04d_loss_%.8f.pth" % (epoch, loss))
    checkpoint_path = os.path.abspath(checkpoint_path)
    
    if torch.cuda.device_count() > 1 and opt.multi_gpu:
        state = {"epoch": epoch, "model": model.module.state_dict(), "optimizer": optimizer.state_dict()}
    else:
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}

    torch.save(state, checkpoint_path)
    print("Checkpoint saved to {}".format(checkpoint_path))


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
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_path, n_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))

    print("Using epoch_num:", n_epoch)
    
    opt.checkpoint_dir = checkpoint_dir
    return n_epoch + 1, model, optimizer


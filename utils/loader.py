import os
from torch.utils import data as D
from torch.utils.data import DataLoader

from data.genoray import GenorayDataset
from data.mayo import MayoDataset
from utils.saver import select_checkpoint_dir

def select_train_dir(opt):
    train_dir = os.path.join(opt.train_dir, opt.dataset)
    dirs = os.listdir(train_dir)
    for i, d in enumerate(dirs, 0):
        print("(%d) %s" % (i, d))
    
    print("Directory{}".format(os.path.abspath(train_dir)))
    d_idx = input("Select directory that contains images you want to train: ")

    path_opt = dirs[int(d_idx)]
    opt.path_opt = path_opt

    train_dir = os.path.abspath(os.path.join(train_dir, dirs[int(d_idx)]))
    print("train_dir is: {}".format(os.path.abspath(train_dir)))
    opt.train_dir = train_dir

    # checkpoint_dir = os.path.join(os.path.abspath(opt.checkpoint_dir), opt.model, opt.path_opt)
    # opt.checkpoint_dir = checkpoint_dir

    return train_dir

def set_train_dir(opt):
    train_dir = os.path.join(opt.train_dir, opt.dataset)
    patch_opt = 'patch' + str(opt.patch_size)
    opt.train_dir = os.path.join(train_dir, patch_opt)
    print("Training directory is: {}".format(os.path.abspath(opt.train_dir)))

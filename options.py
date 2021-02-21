import argparse
import os
import torch

data_dir = r'../../data/denoising'

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
checkpoint_dir= os.path.join(data_dir, 'checkpoint_DA')
test_result_dir = os.path.join(data_dir, 'test_result_DA')

parser = argparse.ArgumentParser(description='CT Denoising Domain Adaptation Benchmark')

parser.add_argument('--mode', type=str, default= 'train', choices=['train', 'test'])
parser.add_argument('--model', type=str, default= 'unet', choices=['dncnn', 'unet', 'resnet'])
parser.add_argument('--way', type=str, default= 'bilateral', choices=['n2v', 'n2s', 'n2c', 'n2sim', 'n2i', 'bilateral'])

parser.add_argument('--multi_gpu', default=False, action='store_true',
                    help='Use multiple GPUs')
parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                    help='Use cuda')
parser.add_argument('--use_cpu', dest='use_cuda', action='store_true',
                    help='Use cpu, do not use cuda')
parser.set_defaults(use_cuda=True)
parser.add_argument('--device', type=str, default='cpu',
                    help='CPU or GPU')
parser.add_argument("--n_threads", type=int, default=2,
                    help="Number of threads for data loader to use, Default: 8")
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed')

parser.add_argument('--patch_size', type=int, default=80,
                    help='Size of patch')
parser.add_argument('--patch_offset', type=int, default=15,
                    help='Size of patch offset')
parser.add_argument('--n_channels', type=int, default=1, choices=[1, 3],
                    help='Number of image channels')
parser.add_argument("--train_ratio", type=float, default=0.95,
                    help="Ratio of train dataset (ex: train:validation = 0.95:0.05), Default: 0.95")                 

parser.add_argument('--ext', type=str, default='sep', choices=['sep', 'img'],
                    help='File extensions')
parser.add_argument('--in_mem', default=False, action='store_true',
                    help="Load whole data into memory, Default: False")
parser.add_argument('--use_pt', default=False, action='store_true',
                    help='use pt data, do not check img files')

parser.add_argument('--source', type=str, default='mayo', choices=['lp-mayo', 'piglet', 'mayo', 'siemens', 'toshiba', 'ge'], 
                    help='Specify dataset name for source dataset (not for base)')
parser.add_argument('--target', type=str, default='mayo', choices=['lp-mayo', 'piglet', 'mayo', 'siemens', 'toshiba', 'ge'],
                    help='Specify dataset name for target dataset (not for base)')
parser.add_argument('--domain_sync', type=str, default=None, choices=[None, 'ref2trg', 'out2src'])
parser.add_argument('--fake_dir', type=str, default=None, help='specify the fake source directory base name(domain_sync : ref2trg, out2src)')
parser.add_argument('--train_datasets', nargs='+', default=None,
                    choices=['mayo','lp-mayo','piglet', 'fake-lp-mayo', 'siemens', 'toshiba', 'ge'],
                    help='Specify dataset name for base, default=source')

parser.add_argument('--data_dir', type=str, default=data_dir,
                    help='Path of training directory contains both lr and hr images')
parser.add_argument('--train_dir', type=str, default=train_dir,
                    help='Path of training directory contains both lr and hr images')
parser.add_argument('--test_dir', type=str, default=test_dir,
                    help='Path of directory to be tested (no ground truth)')
parser.add_argument('--test_result_dir', type=str, default=test_result_dir,
                    help='test result dir')

parser.add_argument("--test_patches", dest='test_patches', action='store_true',
                    help="Divide image into patches")
parser.add_argument('--test_image', dest='test_patches', action='store_false',
                    help='Test whole image instead of dividing into patches')
parser.set_defaults(test_patches=False)
parser.add_argument("--img_dir", type=str, default=None,
                    help="path to image directory")
parser.add_argument('--gt_img_dir', type=str, default=None,
                    help='Path to ground truth image directory')
parser.add_argument('--augment', dest='augment', action='store_true',
                    help='Do random flip (vertical, horizontal, rotation)')
parser.add_argument('--no_augment', dest='augment', action='store_false',
                    help='Do not random flip (vertical, horizontal, rotation)')
parser.set_defaults(augment=True)


# Mayo dataset specifications
parser.add_argument('--thickness', type=int, default=0, choices=[0,1,3],
                    help='Specify thicknesses of mayo dataset (1 or 3 mm or 0(1+3))')

#lp-mayo dataset specifications
parser.add_argument('--body_part', '-bp',type=str, nargs='+', choices=['C', 'L', 'N'], default='L',
                    help='choose body part in ldct-projection-mayo')

#phantom dataset specifications
parser.add_argument('--anatomy', type=str, default=['chest', 'pelvis'], nargs='+',
                    help='Specify anatomy of phantom dataset (chest/hn/pelvis)')
parser.add_argument('--mA_full', '-f', type=str, default='level3', choices = ['level1','level2','level3','level4','level5','level6'],
                    help='Specify full mA level 1,2,3,4,5,6 of phantom dataset')
parser.add_argument('--mA_low', '-l',type=str, default='level5', choices = ['level1','level2','level3','level4','level5','level6'],
                    help='Specify low mA level 1,2,3,4,5,6 of phantom dataset')

parser.add_argument('--test_every', type=int, default=500,
                    help='do test per every N batches')
parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir,
                    help='Path to checkpoint directory')
parser.add_argument('--select_checkpoint', default=False, action='store_true',
                    help='Choose checkpoint directory')
parser.add_argument('--resume', default=False, action='store_true',
                    help='Resume last checkpoint')
parser.add_argument('--resume_best', default=False, action='store_true',
                    help='Resume best accuracy checkpoint')
parser.add_argument('--epoch_num', type=int, default=0,
                    help='epoch number to restart')
parser.add_argument('--ensemble', default=False, action='store_true',
                    help='self ensemble in test')

# Optimizer specification
parser.add_argument("--optimizer", type=str, default='adam',
                    help="Loss function (adam, sgd)")
parser.add_argument('--loss', type=str, default='l1', choices=['l1','l2'])
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9,
                    help='Adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999,
                    help='Adam: decay of second order momentum of gradient')
parser.add_argument("--start_epoch", type=int, default=1,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--batch_size', type=int, default=32,
                    help='Size of the batches')
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--weight_decay', type=float, default= 0.001)
parser.add_argument('--weight_decay_dc', type=float, default= 0.1)


#Unet
parser.add_argument('--bilinear', type=str, default='bilinear', 
                    help='up convolution type (bilinear or transposed2d)')

#Bilateral filter
parser.add_argument('--window', type=float, default=10.0,
                    help='Set window size for apply bilateral filter')
parser.add_argument('--d', type=int, default=3,
                    help='Set d for apply bilateral filter')                    


args = parser.parse_args()

if args.target == 'lp-mayo':
    args.gt_img_dir = r'../../data/denoising/test/lp-mayo/full'
    args.img_dir = r'../../data/denoising/test/lp-mayo/low'
elif args.target == 'mayo':
    if args.thickness == 0:
        args.img_dir = r'../../data/denoising/test/mayo/quarter_*mm'
        args.gt_img_dir = r'../../data/denoising/test/mayo/full_*mm'
    else:
        args.img_dir = r'../../data/denoising/test/mayo/quarter_{}mm'.format(args.thickness)
        args.gt_img_dir = r'../../data/denoising/test/mayo/full_{}mm'.format(args.thickness)
elif args.target == 'piglet':
    args.gt_img_dir = r'../../data/denoising/test/piglet/full'
    args.img_dir = r'../../data/denoising/test/piglet/Oten'
else:
    args.gt_img_dir = r'../../data/denoising/test/phantom/{}/{}/{}*'.format(args.target, args.anatomy, args.mA_full)
    args.img_dir = r'../../data/denoising/test/phantom/{}/{}/{}*'.format(args.target, args.anatomy, args.mA_low)

if args.train_datasets is None:
    args.train_datasets = [args.source]


torch.manual_seed(args.seed)
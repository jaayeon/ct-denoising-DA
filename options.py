import argparse
import os
import torch

data_dir = r'../../data/denoising'

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
checkpoint_dir= os.path.join(data_dir, 'checkpoint_DA')
test_result_dir = os.path.join(data_dir, 'test_result_DA')

parser = argparse.ArgumentParser(description='CT Denoising Domain Adaptation')

parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'fine_tuning'])
parser.add_argument('--model', type=str, default='unet', choices=['dncnn', 'unet', 'edsr'])
parser.add_argument('--way', type=str, default='rev', choices=['base', 'rev', 'wgan', 'wganrev'])
parser.add_argument('--no_rev', dest='rev', action='store_false', help='no domain adversarial loss for denoiser')
parser.set_defaults(rev=True)

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

parser.add_argument('--source', type=str, default='ge', choices=['lp-mayo', 'piglet', 'mayo', 'siemens', 'toshiba', 'ge', 'mayo-syn'], 
                    help='Specify dataset name for source dataset (not for base)')
parser.add_argument('--target', type=str, default='mayo', choices=['lp-mayo', 'piglet', 'mayo', 'siemens', 'toshiba', 'ge', 'mayo-syn'],
                    help='Specify dataset name for target dataset (not for base)')
parser.add_argument('--fine_tuning_num', type=int, default=10, help='back prop num for each image')
parser.add_argument('--fine_tuning_rev', action='store_true', help='add gradient reversal in fine tuning')
parser.add_argument('--train_datasets', nargs='+', default=None,
                    choices=['mayo','lp-mayo','piglet', 'fake-lp-mayo', 'siemens', 'toshiba', 'ge', 'mayo-syn'],
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
parser.add_argument('--crop', type=str, default='random', choices=['center', 'random'], help='just for fine_tuning data transform')
parser.add_argument('--noise', type=str, nargs='+', default=None, choices=['p', 'g', 'bf', 'nlm', 'sp'],
                    help='noise options for target image. p-poisson, g-gaussian, bf-bilateral filter, nlm-non local means, sp-sinogram poisson')
parser.add_argument('--p_lam', type=float, nargs='+', default=[400,1400], help='poisson parameter, default 400,1400 for 1,3mm')
parser.add_argument('--g_std', type=float, nargs='+', default=[0.032,0.016], help='gaussian parameter, default 0.032, 0.016 for 1,3mm')
parser.add_argument('--b_dcs', type=float, nargs='+', default=[10, 0.01, 1],
                    help='bilateral filter parameters.The diameter of each pixel neighborhood, Filter sigma in color space, Filter sigma in the coordinate space')
parser.add_argument('--scale_max', type=float, default=2.0, help='scaling noise in [scale_min, scale_max]')
parser.add_argument('--scale_min', type=float, default=0.5, help='scaling noise in [scale_min, scale_max]')
parser.add_argument('--ratio_std', type=float, default=3.0, help='noise std/real std')

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

#model common
parser.add_argument('--bn', default=False, action='store_true', help='batch normalization')

#edsr
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--n_resblocks', type=int, default=16, 
                    help='# resblocks for edsr')

#wganvgg
parser.add_argument('--lambda_gp', type=float, default= 10, 
                    help='lambda_gp for wgan discriminator loss')
parser.add_argument('--n_d_train', type=float, default=4,
                    help='num of discriminator training for each generator training')
parser.add_argument('--vgg_weight', type=float, default=1,
                    help='perceptual loss weight (wganvgg default was 0.5)') # change p_weight to vgg_weight
parser.add_argument('--l_weight', type=float, default=1,
                    help = 'l1 pixel wise loss in gloss')

#rev
parser.add_argument('--rev_weight', type=float, default=0.001,
                    help='domain classifier reversal loss')
parser.add_argument('--dc_mode', type=str, default='mse', choices=['mse', 'bce', 'wss'], 
                    help='domain classifier loss mode')
parser.add_argument('--dc_input', type=str, default='c_img', choices=['img', 'noise', 'feature', 'c_img', 'c_noise', 'c_feature', 'origin'],
                    help = 'domain classifier input')
parser.add_argument('--src_loss', action='store_true', help='add src domain pixel wise loss in 2nd training')
parser.add_argument('--style_stage', type=int, default=4, choices=[1,2,3,4,5,6],
                    help='stage for feature which is extracted from generator to domain classifier input')
parser.add_argument('--content_randomization', default=False, action='store_true')
parser.add_argument('--sagnet', default=False, action='store_true', 
                    help='only update batch normalization parameters in rev_loss')


parser.add_argument('--test_every', type=int, default=1000,
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
parser.add_argument('--pretrained', default=False, action='store_true',
                    help='initialize weight to pretrained model')

# Optimizer specification
parser.add_argument("--optimizer", type=str, default='adam',
                    help="Loss function (adam, sgd, rms)")
parser.add_argument('--loss', type=str, default='l1', choices=['l1','l2'])
parser.add_argument('--lr', type=float, default=0.0002,
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
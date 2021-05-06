import os
import glob

from data.patchdata import PatchData
from . import common

class MayoSynt2(PatchData):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.set_defaults(n_channels=1)
        # parser.set_defaults(rgb_range=1.0)
        parser.add_argument('--thickness', type=int, default=0, choices=[0, 1, 3],
            help='Specify thicknesses of mayo dataset (1 or 3 mm)')
        parser.add_argument('--add_noise', action='store_true',
            help='add noise')
        parser.set_defaults(add_noise=True)
        return parser

    def __init__(self, args, name='mayosynt2', is_train=True, dir='A'):
        # Mayo specific
        self.thickness = args.thickness
        self.add_noise = args.add_noise
        super(MayoSynt2, self).__init__(
            args, name=name, is_train=is_train
        )

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '**', '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '**', '*' + self.ext[1]))
        )

        domain_labels = [1] * len(names_hr)

        return names_hr, names_lr, domain_labels

    def _set_filesystem(self, data_dir):
        super(MayoSynt2, self)._set_filesystem(data_dir)
        self.apath = os.path.join(data_dir, self.mode, 'mayo')

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

    def __getitem__(self, idx):
        if not self.in_mem:
            lr, hr, filename, domain_label = self._load_file(idx)
        else:
            lr, hr, filename, domain_label = self._load_mem(idx)

        if self.is_train or self.test_random_patch:
            pair = self.get_patch(lr, hr)
        else:
            pair = [lr, hr]

        if self.add_noise:
            lr_noise = common.add_gaussian_noise(pair[0])
            pair = [pair[0], pair[1], lr_noise]

        pair = common.set_channel(*pair, n_channels=self.n_channels)
        # if self.n_channels == 3: pair = [(p.astype(np.float) / 255.0) for p in pair]
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        data_dict = {
            'lr': pair_t[0],
            'hr': pair_t[1],
            'filename': filename,
            'domain_label': domain_label
        }

        if self.add_noise:
            # print('lr_noise.shape:', pair_t[2].shape)
            data_dict['lr_noise'] = pair_t[2]
        return data_dict
import os
import glob

from data.patchdata import PatchData

class Phantom(PatchData):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(n_channels=1)
        parser.set_defaults(rgb_range=1.0)
        
        return parser

    def __init__(self, args, name='phantom', is_train=True, dir='A'):
        super(Phantom, self).__init__(
            args, name=name, is_train=is_train
        )
        

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '**', '*' + self.ext[0]), recursive=True)
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '**', '*' + self.ext[1]), recursive=True)
        )

        assert len(names_hr) == len(names_lr), "Numbers of samples in hr and lr are different"

        domain_labels = [0] * len(names_hr)

        return names_hr, names_lr, domain_labels

    def _set_filesystem(self, data_dir):
        super(Phantom, self)._set_filesystem(data_dir)

        self.dir_hr = os.path.join(self.apath, 'high')
        self.dir_lr = os.path.join(self.apath, 'low')
        self.ext = ('.tiff', '.tiff')
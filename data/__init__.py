import os, glob
from importlib import import_module
# from torch.utils.data import dataloader
from torch.utils import data as D
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.mode = datasets[0].mode

    def set_scale(self, idx_scale):
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)



def get_module_attr(dataset):
    if dataset == 'mayo':
        module_name = 'mayo'
        attr = 'Mayo'
    elif 'lp-mayo' in dataset:
        module_name = 'lp-mayo'
        attr = 'LPMAYO'
    elif 'piglet' in dataset:
        module_name = 'piglet'
        attr = 'PIGLET'
    elif 'siemens' in dataset:
        module_name = 'phantom'
        attr = 'PHANTOM'
    elif 'toshiba' in dataset:
        module_name = 'phantom'
        attr = 'PHANTOM'
    elif 'ge' in dataset:   
        module_name = 'phantom'
        attr = 'PHANTOM'  

    print("{} module_name: {}".format(__file__, module_name))
    print("attr:", attr)
    return module_name, attr



def get_train_valid_dataloader(args, train_datasets=None, domain_sync=None):
    datasets = []
    if train_datasets == None:
        train_datasets = args.train_datasets
    else : 
        train_datasets = [train_datasets]
    print("Train datasets: ", train_datasets)
    for d in train_datasets:
        # module_name = d
        module_name, attr = get_module_attr(d)
        m = import_module('data.' + module_name.lower())
        datasets.append(getattr(m, attr)(args, name=d, domain_sync= domain_sync))

    # module_name, attr = get_module_attr(args.dataset)
    # m = import_module('data.' + module_name.lower())

    # datasets.append(getattr(m, attr)(args, name=args.dataset))

    train_ds = MyConcatDataset(datasets)
    train_len = int(args.train_ratio * len(train_ds))
    valid_len = len(train_ds) - train_len
    train_dataset, valid_dataset = D.random_split(train_ds, lengths=[train_len, valid_len])
    
    print("Number of train dataset samples:", train_len)
    print("Number of valid dataset samples:", valid_len)
    print("Threading {}".format(args.n_threads))

    train_data_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=args.n_threads)
    valid_data_loader = DataLoader(dataset=valid_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=args.n_threads)

    return train_data_loader, valid_data_loader




def get_test_img_list(opt):
    if opt.way == 'bilateral':
        img_list = glob.glob(os.path.join(opt.img_dir, '*','*'))
        gt_img_list = glob.glob(os.path.join(opt.gt_img_dir, '*','*'))
    else:
        img_list = glob.glob(os.path.join(opt.img_dir, '**'))
        gt_img_list = glob.glob(os.path.join(opt.gt_img_dir, '**'))
    # print(img_list)
    print('test img low path : {}\ntest img high path : {}'.format(opt.img_dir, opt.gt_img_dir))
    
    return img_list, gt_img_list

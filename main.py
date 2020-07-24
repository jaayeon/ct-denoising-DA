from data import get_train_valid_dataloader, get_test_img_list, get_test_noisy_list, get_train_dataloader

import trainer as T
import tester as TST
import tester_lpmayo as TSTM

from utils.saver import load_config
from options import args

if __name__ == '__main__':
    opt = args

    if opt.mode == 'train':
        print(opt)

        train_data_loader, valid_data_loader = get_train_valid_dataloader(opt)
        # only_train_data_loader = get_train_dataloader(opt)
        T.run_train(opt, train_data_loader, valid_data_loader)

    elif opt.mode == 'test':
        # opt.resume_best = True
        opt = load_config(opt)
        print(opt)
        img_list, gt_img_list = get_test_img_list(opt)
        if opt.dataset == 'lp-mayo':
            TSTM.run_test(opt, img_list, gt_img_list)
        else :
            TST.run_test(opt, img_list, gt_img_list)

from data import get_train_valid_dataloader, get_test_img_list

import trainer as T
import trainer_BDL as BT
import trainer_BDLW as BTW
import trainer_SELF as BTS
import tester as TST
import tester_lpmayo as TSTM

from utils.saver import load_config
from options import args

if __name__ == '__main__':
    opt = args

    if opt.mode == 'train':
        print(opt)

        if opt.way == 'adv':
            train_source_loader, valid_source_loader = get_train_valid_dataloader(opt, train_datasets=opt.source, domain=opt.sync_domain)
            train_target_loader, valid_target_loader = get_train_valid_dataloader(opt, train_datasets=opt.target, domain=None)
            BT.run_train(opt, train_source_loader, valid_source_loader, train_target_loader, valid_target_loader)
         

        elif opt.way == 'base':
            train_data_loader, valid_data_loader = get_train_valid_dataloader(opt, domain=opt.sync_domain)
            # only_train_data_loader = get_train_dataloader(opt)
            T.run_train(opt, train_data_loader, valid_data_loader)


        elif opt.way == 'wadv':
            train_source_loader, valid_source_loader = get_train_valid_dataloader(opt, train_datasets=opt.source, domain=opt.sync_domain)
            train_target_loader, valid_target_loader = get_train_valid_dataloader(opt, train_datasets=opt.target, domain=None)
            BTW.run_train(opt, train_source_loader, valid_source_loader, train_target_loader, valid_target_loader)

        elif opt.way == 'self':
            train_self_loader, valid_self_loader = get_train_valid_dataloader(opt, train_datasets=opt.source)
            BTS.run_train(opt, train_self_loader, valid_self_loader)


    elif opt.mode == 'test':
        # opt.resume_best = True
        opt = load_config(opt)
        print(opt)
        img_list, gt_img_list = get_test_img_list(opt)
        if opt.target == 'lp-mayo':
            TSTM.run_test(opt, img_list, gt_img_list)
        else :
            TST.run_test(opt, img_list, gt_img_list)

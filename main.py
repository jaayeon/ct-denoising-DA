from data import get_train_valid_dataloader, get_test_img_list

import trainer as T
import trainer_rev as REVT
import trainer_wganvgg as WT
import trainer_wganvgg_rev as WREVT
import fine_tuning_rev as FREV
import tester as TST
import tester_lpmayo as TSTM

from utils.loader import load_config
from options import args

if __name__ == '__main__':
    opt = args

    if opt.mode == 'train':
        print(opt)

        if opt.way == 'base':
            train_data_loader, valid_data_loader = get_train_valid_dataloader(opt)
            T.run_train(opt, train_data_loader, valid_data_loader)

        elif opt.way == 'rev':
            train_source_loader, valid_source_loader = get_train_valid_dataloader(opt, train_datasets=opt.source)
            train_target_loader, valid_target_loader = get_train_valid_dataloader(opt, train_datasets=opt.target, add_noise=opt.noise)
            REVT.run_train(opt, train_source_loader, valid_source_loader, train_target_loader, valid_target_loader)
            
        elif opt.way == 'wgan':
            train_data_loader, valid_data_loader = get_train_valid_dataloader(opt, train_datasets=opt.source)
            WT.run_train(opt, train_data_loader, valid_data_loader)
                
        elif opt.way == 'wganrev':
            train_source_loader, valid_source_loader = get_train_valid_dataloader(opt, train_datasets=opt.source)
            train_target_loader, valid_target_loader = get_train_valid_dataloader(opt, train_datasets=opt.target, add_noise=opt.noise)
            WREVT.run_train(opt, train_source_loader, valid_source_loader, train_target_loader, valid_target_loader)
    
    if opt.mode == 'fine_tuning':
        print(opt)
        if opt.way == 'rev':
            FREV.run_train(opt, opt.source, opt.target)
        else : 
            raise NotImplementedError('not implemented')

    elif opt.mode == 'test':
        # opt.resume_best = True
        opt = load_config(opt)
        print(opt)
        img_list, gt_img_list = get_test_img_list(opt)
        if opt.target == 'lp-mayo':
            TSTM.run_test(opt, img_list, gt_img_list)
        else :
            TST.run_test(opt, img_list, gt_img_list)

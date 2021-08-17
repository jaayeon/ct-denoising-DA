from data import get_train_valid_dataloader, get_test_img_list

import trainer as T

import trainer_n2v as N2VT
import trainer_n2c as N2CT
import trainer_n2s as N2ST
import trainer_n2sim as N2SimT
import trainer_n2n as N2N
import tester as TST
import tester_lpmayo as TSTM
import tester_n2n as TSTN
import tester_bilateral as TBL

from utils.saver import load_config
from options import args

if __name__ == '__main__':
    opt = args

    if opt.mode == 'train':
        print(opt)

        if opt.way == 'n2v':
            train_n2v_loader, valid_n2v_loader = get_train_valid_dataloader(opt, train_datasets=opt.source)
            N2VT.run_train(opt, train_n2v_loader, valid_n2v_loader)
        
        elif opt.way == 'n2s':
            train_n2s_loader, valid_n2s_loader = get_train_valid_dataloader(opt, train_datasets=opt.source)
            N2ST.run_train(opt, train_n2s_loader, valid_n2s_loader)
            
        elif opt.way == 'n2c':
            train_n2c_loader, valid_n2c_loader = get_train_valid_dataloader(opt, train_datasets=opt.source)
            N2CT.run_train(opt, train_n2c_loader, valid_n2c_loader)
        
        elif opt.way == 'n2n':
            train_n2n_loader, valid_n2n_loader = get_train_valid_dataloader(opt, train_datasets=opt.source)
            N2N.run_train(opt, train_n2n_loader, valid_n2n_loader)

        elif opt.way == 'bilateral':
            img_list, gt_img_list = get_test_img_list(opt)
            TBL.run_test(opt, img_list, gt_img_list)


    elif opt.mode == 'test':
        opt.resume_best = True
        opt = load_config(opt)
        #print(opt)
        img_list, gt_img_list = get_test_img_list(opt)

        #print(img_list)
    
        if opt.target == 'lp-mayo':
            TSTM.run_test(opt, img_list, gt_img_list)
        elif opt.way == 'n2n':
            TSTN.run_test(opt, img_list, gt_img_list)
        else :
            TST.run_test(opt, img_list, gt_img_list)
           
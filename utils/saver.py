import os
import json
import torch
import time

class Record():
    def __init__(self, opt, train_length=None, valid_length=None, keys=None):
        self.opt = opt
        self.checkpoint_dir = opt.checkpoint_dir
        self.model_name = opt.model

        self.keys = keys
        self.key_length = len(self.keys)
        self.n_epochs = opt.n_epochs
        self.train_length = train_length
        self.valid_length = valid_length

        self.epoch = opt.start_epoch
        self.train = [0]*self.key_length
        self.valid = [0]*self.key_length
        self.buffer = [0]*self.key_length
        self.start_time = time.time()
        self.train_iter = 0
        self.valid_iter = 0
        self.best_psnr = 0
        self.best_loss = 99999999

        self.log_file = os.path.join(self.checkpoint_dir, opt.model + "_log.csv")
        self.opt_file = os.path.join(self.checkpoint_dir, "config.txt")

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if self.epoch == 1:
            with open(self.log_file, mode='w', newline='\n') as f:
                f.write('epoch, {}, {}\n'.format('-t, '.join(self.keys), '-v, '.join(self.keys)))

        with open(self.opt_file, 'w') as f:
            json.dump(opt.__dict__, f, indent=2)

    def update_status(self, buffer, mode='train'):
        #update buffer and add it to train/valid
        self.buffer = buffer
        if mode=='train':
            self.train_iter += 1
            for i in range(self.key_length):
                self.train[i] += self.buffer[i]
        else : 
            self.valid_iter += 1
            for i in range(self.key_length):
                self.valid[i] += self.buffer[i]

    def print_buffer(self, mode='train'):
        #print buffer
        now = time.time()-self.start_time
        if mode == 'train':
            mode='Training'
            print('{} {:.2f}s => Epoch[{}/{}]({}/{}) : {}'.format(mode, now, self.epoch, self.n_epochs, self.train_iter, self.train_length,
                    ', '.join(['{}:{:2.3f}'.format(self.keys[i], self.buffer[i]) for i in range(self.key_length)])))
        else : 
            mode='Validation'
            print('{} {:.2f}s => Epoch[{}/{}]({}/{}) : {}'.format(mode, now, self.epoch, self.n_epochs, self.valid_iter, self.valid_length,
                    ', '.join(['{}:{:2.3f}'.format(self.keys[i], self.buffer[i]) for i in range(self.key_length)])))

    def print_average(self, mode='train'):
        #update train/valid to average & print status
        if mode == 'train':
            for i in range(self.key_length):
                self.train[i] = self.train[i]/self.train_length
            print('[*] Training Epoch[{}/{}] : {}'.format(self.epoch, self.n_epochs,
                ', '.join(['{}:{:.3f}'.format(self.keys[i], self.train[i]) for i in range(self.key_length)])))
        else : 
            for i in range(self.key_length):
                self.valid[i] = self.valid[i]/self.valid_length
            print('[*] Validation Epoch[{}/{}] : {}'.format(self.epoch, self.n_epochs,
                ', '.join(['{}:{:.3f}'.format(self.keys[i], self.valid[i]) for i in range(self.key_length)])))

    def write_log(self):
        #write log
        with open(self.log_file, mode='a') as f:
            f.write('{}, {}, {}\n'.format(self.epoch, ', '.join([str(i) for i in self.train]), ', '.join([str(i) for i in self.valid])))
        #initialize for next epoch
        self.train = [0]*self.key_length
        self.valid = [0]*self.key_length
        self.train_iter = 0
        self.valid_iter = 0
        self.epoch +=1
        self.start_time = time.time()


    def save_checkpoint(self, model, optimizer, save_criterion=None):
        #it should precede write_log() 
        try : current = self.valid[self.keys.index(save_criterion)]
        except : raise KeyError("'save criterion' should be one of keys")

        if 'psnr' in save_criterion:
            criterion = 'psnr'
            if self.best_psnr < current:
                self.best_psnr = current
                save = True
            else : 
                save = False
        elif 'loss' in save_criterion:
            criterion = 'loss'
            if self.best_loss > current:
                self.best_loss = current
                save = True
            else : 
                save = False
        else : 
            raise KeyError('save_criterion should be one of loss or psnr')

        if save:
            checkpoint_path = os.path.join(self.checkpoint_dir, "{}_epoch_{:04d}_{}_{:.8f}.pth" .format(self.model_name, self.epoch, criterion, current))
            checkpoint_path = os.path.abspath(checkpoint_path)
            optimizers=[optimizer[i].state_dict() for i in range(len(optimizer))]
            if torch.cuda.device_count() > 1 and self.opt.multi_gpu:
                state = {"epoch": self.epoch, "model": model.module.state_dict(), "optimizer": optimizers}
            else:
                state = {"epoch": self.epoch, "model": model.state_dict(), "optimizer": optimizers}

            torch.save(state, checkpoint_path)
            print("Checkpoint saved to {}".format(checkpoint_path))
        else : 
            pass

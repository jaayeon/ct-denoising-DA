import time

class Record():
    def __init__(self, opt, log_file=None, train_length=None, valid_length=None):
        self.keys = ['gloss', 'advloss', 'lloss', 'ploss', 'revloss', 'dcloss', 'dloss', 'src_psnr', 'nsrc_psnr', 'trg_psnr', 'ntrg_psnr']
        self.key_length = len(self.keys)
        self.log_file = log_file
        self.n_epochs = opt.n_epochs
        self.train_length = train_length
        self.valid_length = valid_length

        self.epoch = 1
        self.train = [0]*self.key_length
        self.valid = [0]*self.key_length
        self.buffer = [0]*self.key_length
        self.start_time = time.time()
        self.train_iter = 0
        self.valid_iter = 0

        with open(self.log_file, mode='w') as f:
            f.write('epoch, {}, {}'.format('-t, '.join(self.keys), '-v, '.join(self.keys)))

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
                    ', '.join(['{}:{:.3f}'.format(self.keys[i], self.buffer[i]) for i in range(self.key_length)])))
        else : 
            mode='Validation'
            print('{} {:.2f}s => Epoch[{}/{}]({}/{}) : {}'.format(mode, now, self.epoch, self.n_epochs, self.valid_iter, self.valid_length,
                    ', '.join(['{}:{:.3f}'.format(self.keys[i], self.buffer[i]) for i in range(self.key_length)])))

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


    def write_log(self, epoch):
        #write log
        with open(self.log_file, mode='a') as f:
            f.write('{}, {}, {}\n'.format(self.epoch, ', '.join([str(i) for i in self.train]), ', '.join([str(i) for i in self.valid])))
        #initiate it for next epoch
        self.train = [0]*self.key_length
        self.valid = [0]*self.key_length
        self.train_iter = 0
        self.valid_iter = 0
        self.epoch = epoch+1 
        self.start_time = time.time()

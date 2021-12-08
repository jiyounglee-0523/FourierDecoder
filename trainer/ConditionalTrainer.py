import torch

import os
import wandb
import time
from datetime import datetime


from datasets.cond_dataset import get_dataloader
from models.latentmodel import ConditionalQueryFNP
from utils.model_utils import count_parameters, EarlyStopping
from utils.trainer_utils import log

class ConditionalBaseTrainer():
    def __init__(self, args):
        self.train_dataloader = get_dataloader(args, 'train')
        self.eval_dataloader = get_dataloader(args, 'eval')
        self.n_epochs = args.n_epochs

        self.debug = args.debug
        self.dataset_type = args.dataset_type
        self.n_harmonics = args.n_harmonics
        NP = 'NP' if args.NP else 'nonNP'

        filename = f'{datetime.now().date()}_{args.dataset_type}_{args.dataset_name}_{NP}_{args.lower_bound}_{args.upper_bound}_{args.encoder}_{args.encoder_blocks}_{args.encoder_hidden_dim}_decoder_{args.decoder}_{args.decoder_layers}'

        args.filename = filename

        self.path = args.path + filename
        self.file_path = self.path + '/' + filename
        print(f'Model will be saved at {self.path}')

        os.mkdir(self.path)
        self.logger = log(path=self.path + '/', file=filename + '.logs')



class ConditionalNPTrainer(ConditionalBaseTrainer):
    def __init__(self, args):
        super(ConditionalNPTrainer, self).__init__(args)

        self.model = ConditionalQueryFNP(args).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.alpha = 1
        self.max_num = 0

        if not self.debug:
            wandb.init(project='NIPS_workshop', config=args)
            self.logger.info(f'Number of parameters: {count_parameters(self.model)}')
            self.logger.info(f'Wandb Project Name: {args.dataset_type+args.dataset_name}')
        if args.rerun:
            file_list = os.listdir(self.path)
            num_list = [int(file.split('.')[-2].split('_')[-1]) for file in file_list if
                        file.split('.')[-2].split('_')[-1] != 'best' and file.split('.')[-2].split('_')[-1] != 'alpha1']
            max_num = max(num_list)

            ckpt = torch.load(self.file_path + f'_{max_num}.pt')
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.best_loss = ckpt['loss']
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('Successfully loaded!')
            self.max_num = max_num


        print(f'Number of parameters: {count_parameters(self.model)}')

    def train(self):
        best_mse = float('inf')
        for n_epoch in range(self.n_epochs):
            starttime = time.time()

            for it, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin = sample['sin'].cuda()    # B, S, 1
                label = sample['label'].squeeze(-1).cuda()     # B
                orig_ts = sample['orig_ts'].cuda() # B, S
                index = sample['index'].cuda()  # B, N

                mse_loss, kl_loss = self.model(orig_ts, samp_sin, label, index)
                loss = mse_loss + self.alpha * kl_loss
                # loss = mse_loss
                loss.backward()
                self.optimizer.step()

                if not self.debug:
                    wandb.log({'train_loss': loss,
                               'train_kl_loss': kl_loss,
                               'train_mse_loss': mse_loss,
                               'epoch': n_epoch,
                               'alpha': self.alpha})
                    self.logger.info(f'[Train Loss]: {loss:.4f}     [Train MSE]: {mse_loss:.4f}      [Train KL]: {kl_loss:.4f}')

                else:
                    print(f'[Train Loss]: {loss:.4f}      [Train MSE]: {mse_loss:.4f}    [Train KL]: {kl_loss:.4f}')

            endtime = time.time()
            if not self.debug:
                self.logger.info(f'[Time] : {endtime-starttime}')
            else:
                print(f'[Time] : {endtime-starttime}')

            eval_loss, eval_mse, eval_kl = self.evaluation()
            if not self.debug:
                wandb.log({'eval_loss': eval_loss,
                           'eval_mse': eval_mse,
                           'eval_kl': eval_kl,
                           'epoch': n_epoch,
                           'alpha': self.alpha})

                self.logger.info(f'[Eval Loss]: {eval_loss:.4f}      [Eval MSE]: {eval_mse:.4f}   [Eval KL]: {eval_kl:.4f}')
            else:
                print(f'[Eval Loss]: {eval_loss:.4f}      [Eval MSE]: {eval_mse:.4f}      [Eval KL]: {eval_kl:.4f}')

            if best_mse > eval_loss:
                best_mse = eval_loss
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': best_mse}, self.file_path+'_best.pt')
                    self.logger.info(f'Model parameter saved at {n_epoch}')

            if n_epoch % 50 == 0:    # 50 epoch 마다 모델 저장하기
                torch.save({'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': eval_loss}, self.file_path + f'_{n_epoch + self.max_num}.pt')

    def evaluation(self):
        self.model.eval()
        avg_eval_loss = 0.
        avg_eval_mse = 0.
        avg_kl = 0.

        with torch.no_grad():
            for it, sample in enumerate(self.eval_dataloader):
                samp_sin = sample['sin'].cuda()
                label = sample['label'].squeeze(-1).cuda()
                orig_ts = sample['orig_ts'].cuda()
                index = sample['index'].cuda()

                mse_loss, kl_loss = self.model(orig_ts, samp_sin, label, index)
                loss = mse_loss + self.alpha * kl_loss
                # loss = mse_loss
                avg_eval_loss += (loss.item() * samp_sin.size(0))
                avg_eval_mse += (mse_loss.item() * samp_sin.size(0))
                avg_kl += (kl_loss * samp_sin.size(0))

            avg_eval_loss /= self.eval_dataloader.dataset.__len__()
            avg_eval_mse /= self.eval_dataloader.dataset.__len__()
            avg_kl /= self.eval_dataloader.dataset.__len__()

        return avg_eval_loss, avg_eval_mse, avg_kl


    def test(self):
        self.model.eval()
        ckpt = torch.load(self.path)
        self.model.load_state_dict(ckpt['model_state_dict'])

        avg_test_loss = 0.
        avg_test_mse = 0.
        avg_kl = 0.
        with torch.no_grad():
            for it, sample in enumerate(self.test_dataloder):
                samp_sin = sample['sin'].cuda()
                label = sample['label'].squeeze(-1).cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss = self.model(orig_ts, samp_sin, label, sampling=False)
                loss = mse_loss + kl_loss
                avg_test_loss += (loss.item() / len(self.test_dataloder))
                avg_test_mse += (mse_loss.item() / len(self.test_dataloder))
                avg_kl += (kl_loss.item() / len(self.test_dataloder))

        if not self.debug:
            wandb.log({'test_loss': avg_test_loss,
                       'test_mse': avg_test_mse,
                       'test_kl': avg_kl})


class ConditionalNPContinualTrainer(ConditionalBaseTrainer):
    def __init__(self, args):
        super(ConditionalNPContinualTrainer, self).__init__(args)
        print('Continual Learning')
        if args.query:
            self.model = ConditionalQueryContinualFNP(args).cuda()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.early_stopping = EarlyStopping(patience=50, verbose=True)

        if not self.debug:
            wandb.init(project='conditionalODE', config=args)

    def train(self):
        best_mse = float('inf')
        coeff_num = 50

        for n_epoch in range(self.n_epochs):
            starttime = time.time()
            for it, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin = sample['sin'].cuda()  # (B, S, 1)
                label = sample['label'].squeeze(-1).cuda()   # B
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss = self.model(orig_ts, samp_sin, label, sampling=False, coeff_num=coeff_num)
                loss = mse_loss + kl_loss
                loss.backward()
                self.optimizer.step()

                if not self.debug:
                    wandb.log({'train_loss': loss,
                               'train_kl_loss': kl_loss,
                               'train_mse_loss': mse_loss,
                               'coeff_num': coeff_num,
                               'epoch': n_epoch})

            endtime = time.time()
            print(f'[Time]: {endtime-starttime}')

            eval_loss, eval_mse, eval_kl = self.evaluation(coeff_num)

            if coeff_num != self.n_harmonics:
                self.early_stopping(eval_loss)
                if self.early_stopping.early_stop:
                    coeff_num += 50
                    print(f'coeff num: {coeff_num} at epoch {n_epoch}')

                    # reset the early stopping
                    self.early_stopping.counter = 0
                    self.early_stopping.best_score = None
                    self.early_stopping.early_stop = False
                    self.early_stopping.val_loss_min = float('inf')

            if not self.debug:
                wandb.log({'eval_loss': eval_loss,
                           'eval_mse': eval_mse,
                           'eval_kl': eval_kl})

            print(f'[Eval Loss]: {eval_loss:.4f}      [Eval MSE]: {eval_mse:.4f}')

            if best_mse > eval_loss:
                best_mse = eval_loss
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_mse}, self.path)
                    print(f'Model parameter saved at {n_epoch}')


    def evaluation(self, coeff_num):
        self.model.eval()
        avg_eval_loss = 0.
        avg_eval_mse = 0.
        avg_kl = 0.
        with torch.no_grad():
            for it, sample in enumerate(self.eval_dataloader):
                samp_sin = sample['sin'].cuda()
                label = sample['label'].squeeze(-1).cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss = self.model(orig_ts, samp_sin, label, sampling=False, coeff_num=coeff_num)
                loss = mse_loss + kl_loss

                avg_eval_loss += (loss.item() * samp_sin.size(0))
                avg_eval_mse += (mse_loss.item() * samp_sin.size(0))
                avg_kl += (kl_loss.item() * samp_sin.size(0))
        return avg_eval_loss / 2000, avg_eval_mse/2000, avg_kl/2000







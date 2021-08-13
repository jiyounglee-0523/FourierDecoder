import torch
import torch.nn as nn

import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
import time

from datasets.cond_dataset import get_dataloader
from models.latentmodel import *
from models.AttentionFourier import AEAttnFNP, UnconditionalAttnFNP
from utils.model_utils import count_parameters
from utils.trainer_utils import update_learning_rate

class UnconditionalBaseTrainer():
    def __init__(self, args):
        self.train_dataloader = get_dataloader(args, 'train')
        self.eval_dataloader = get_dataloader(args, 'eval')
        self.test_dataloader = get_dataloader(args, 'test')
        self.n_epochs = args.n_epochs
        self.run_continue = args.run_continue

        self.debug = args.debug
        self.dataset_type = args.dataset_type
        self.n_harmonics = args.n_harmonics
        self.path = args.path + args.filename
        self.file_path = self.path + '/' + args.filename
        print(f'Model will be saved at {self.path}')

        if not self.debug:
            if not args.run_continue:
                os.mkdir(self.path)
                print('New experiment')
            # if os.path.exists(self.path):
            #     print(self.path)
            #     raise OSError('saving directory already exists')
            # else:
            #     os.mkdir(self.path)


#######################################################################################
#######################################################################################
class UnconditionalAETrainer(UnconditionalBaseTrainer):
    def __init__(self, args):
        super(UnconditionalAETrainer, self).__init__(args)

        if args.query and not args.attn:
            self.model = nn.DataParallel(AEQueryFNP(args).cuda())
        elif not args.query and args.attn:
            self.model = nn.DataParallel(AEAttnFNP(args).cuda())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        if args.run_continue:
            file_list = os.listdir(self.path)
            num_list = [int(file.split('.')[0].split('_')[-1]) for file in file_list if file.split('.')[0].split('_')[-1] != 'best']
            max_num = max(num_list)

            ckpt = torch.load(self.file_path + f'_{max_num}.pt')
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.best_loss = ckpt['loss']
            print('Successfully loaded model parameter')


        print(f'Number of parameters: {count_parameters(self.model)}')
        print(f'Description: {str(args.notes)}')

        if not self.debug:
            wandb.init(project=args.dataset_type+args.dataset_name, config=args, entity='fourierode')

    def train(self):
        if not self.run_continue:
            best_mse = float('inf')
        else:
            best_mse = self.best_loss

        print('Start Training!')
        for n_epoch in range(self.n_epochs):
            starttime = time.time()
            for it, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin = sample['sin'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, ortho_loss = self.model(orig_ts, samp_sin)   # add orthonormal loss if necessary
                mse_loss = mse_loss.mean() ; ortho_loss = ortho_loss.mean()

                loss = mse_loss + (0.01 * ortho_loss)
                loss.backward()
                self.optimizer.step()

                if not self.debug:
                    wandb.log({'train_loss': loss,
                               'train_mse_loss': mse_loss,
                               'train_ortho_loss': ortho_loss,
                               'epoch': n_epoch})

                print(f'[Train Loss]: {loss:.4f}     [Train MSE]: {mse_loss:.4f}      [Train Ortho]: {ortho_loss:.4f}')

            endtime = time.time()
            print(f'[Time]: {endtime-starttime}')

            eval_loss, eval_mse, eval_ortho_loss = self.evaluation()

            if not self.debug:
                wandb.log({'eval_loss': eval_loss,
                           'eval_mse': eval_mse,
                           'eval_ortho_loss': eval_ortho_loss,
                           'epoch': n_epoch})

            print(f'[Eval Loss]: {eval_loss:.4f}   [Eval MSE]: {eval_mse:.4f}  [Eval Ortho]: {eval_ortho_loss:.4f}')

            if best_mse > eval_loss:
                best_mse = eval_loss
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                               'loss': best_mse}, self.file_path + '_best.pt')
                    print(f'Best model parameter saved at {n_epoch}')

            if n_epoch % 50 == 0:
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': best_mse}, self.file_path + f'_{n_epoch}.pt')
                # if n_epoch != 0:
                #     update_learning_rate(self.optimizer, decay_rate=0.999, lowest=1e-5)


    def evaluation(self):
        self.model.eval()
        avg_eval_loss = 0.
        avg_eval_mse = 0.
        avg_ortho_loss = 0.

        with torch.no_grad():
            for it, sample in enumerate(self.eval_dataloader):
                samp_sin = sample['sin'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, ortho_loss = self.model(orig_ts, samp_sin)
                mse_loss = mse_loss.mean(); ortho_loss = ortho_loss.mean()
                loss = mse_loss + (0.01 * ortho_loss)
                avg_eval_loss += (loss.item() * samp_sin.size(0))
                avg_eval_mse += (mse_loss.item() * samp_sin.size(0))
                avg_ortho_loss += (ortho_loss * samp_sin.size(0))

            avg_eval_loss /= self.eval_dataloader.dataset.__len__()
            avg_eval_mse /= self.eval_dataloader.dataset.__len__()
            avg_ortho_loss /= self.eval_dataloader.dataset.__len__()

        return avg_eval_loss, avg_eval_mse, avg_ortho_loss





class UnconditionalNPTrainer(UnconditionalBaseTrainer):
    def __init__(self, args):
        super(UnconditionalNPTrainer, self).__init__(args)

        self.model = nn.DataParallel(NonConditionalQueryFNP(args).cuda())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = args.lr)

        print(f'Number of parameters: {count_parameters(self.model)}')
        print(f'Description: {str(args.notes)}')

        if not self.debug:
            wandb.init(project='conditionalODE', config=args)

    def train(self):
        best_mse = float('inf')
        print('Start Training!')
        for n_epoch in range(self.n_epochs):
            starttime = time.time()
            for it, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin = sample['sin'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss, ortho_loss = self.model(orig_ts, samp_sin)   # add orthonormal loss if necessary
                mse_loss = mse_loss.mean() ; kl_loss = kl_loss.mean() ; ortho_loss = ortho_loss.mean()

                loss = mse_loss + kl_loss + (0.01 * ortho_loss)
                loss.backward()
                self.optimizer.step()

                if not self.debug:
                    wandb.log({'train_loss': loss,
                               'train_kl_loss': kl_loss,
                               'train_mse_loss': mse_loss,
                               'train_ortho_loss': ortho_loss,
                               'epoch': n_epoch})

                print(f'[Train Loss]: {loss:.4f}     [Train MSE]: {mse_loss:.4f}      [Train Ortho]: {ortho_loss:.4f}')

            endtime = time.time()
            print(f'[Time]: {endtime-starttime}')

            eval_loss, eval_mse, eval_kl, eval_ortho_loss = self.evaluation()

            if not self.debug:
                wandb.log({'eval_loss': eval_loss,
                           'eval_mse': eval_mse,
                           'eval_kl': eval_kl,
                           'eval_ortho_loss': eval_ortho_loss,
                           'epoch': n_epoch})

            print(f'[Eval Loss]: {eval_loss:.4f}   [Eval MSE]: {eval_mse:.4f}  [Eval Ortho]: {eval_ortho_loss:.4f}')

            if best_mse > eval_loss:
                best_mse = eval_loss
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_mse}, self.file_path + '_best.pt')
                    print(f'Best model parameter saved at {n_epoch}')

            if n_epoch % 50 == 0:
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_mse}, self.file_path + f'_{n_epoch}.pt')

    def evaluation(self):
        self.model.eval()
        avg_eval_loss = 0.
        avg_eval_mse = 0.
        avg_kl = 0.
        avg_ortho_loss = 0.

        with torch.no_grad():
            for it, sample in enumerate(self.eval_dataloader):
                samp_sin = sample['sin'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss, ortho_loss = self.model(orig_ts, samp_sin)
                mse_loss = mse_loss.mean(); kl_loss = kl_loss.mean(); ortho_loss = ortho_loss.mean()
                loss = mse_loss + kl_loss + (0.01 * ortho_loss)
                avg_eval_loss += (loss.item() * samp_sin.size(0))
                avg_eval_mse += (mse_loss.item() * samp_sin.size(0))
                avg_kl += (kl_loss.item() * samp_sin.size(0))
                avg_ortho_loss += (ortho_loss * samp_sin.size(0))

            avg_eval_loss /= 20000
            avg_eval_mse /= 20000
            avg_kl /= 20000
            avg_ortho_loss /= 20000

        return avg_eval_loss, avg_eval_mse, avg_kl, avg_ortho_loss


class UnconditionalAttnTrainer(UnconditionalBaseTrainer):
    def __init__(self, args):
        super(UnconditionalAttnTrainer, self).__init__(args)

        self.model = nn.DataParallel(UnconditionalAttnFNP(args).cuda())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        print(f'Number of parameters: {count_parameters(self.model)}')
        print(f'Description: {args.notes}')

        if not self.debug:
            wandb.init(project='conditionalODE', config=args)

    def train(self):
        best_mse = float('inf')
        print('Start Training!')

        for n_epoch in range(self.n_epochs):
            starttime = time.time()
            for it, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin = sample['sin'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss, ortho_loss = self.model(orig_ts, samp_sin)
                mse_loss = mse_loss.mean() ; kl_loss = kl_loss.mean() ; ortho_loss = ortho_loss.mean()

                loss = mse_loss + kl_loss + (0.01 * ortho_loss)
                # loss = mse_loss + (0.01 * ortho_loss)
                loss.backward()
                self.optimizer.step()

                if not self.debug:
                    wandb.log({'train_loss': loss,
                               'train_kl_loss': kl_loss,
                               'train_mse_loss': mse_loss,
                               'train_ortho_loss': ortho_loss,
                               'epoch': n_epoch})

                print(f'[Train Loss]: {loss:.4f}     [Train MSE]: {mse_loss:.4f}      [Train Ortho]: {ortho_loss:.4f}')

            endtime = time.time()
            print(f'[Time]: {endtime - starttime}')

            eval_loss, eval_mse, eval_kl, eval_ortho_loss = self.evaluation()

            if not self.debug:
                wandb.log({'eval_loss': eval_loss,
                           'eval_mse': eval_mse,
                           'eval_kl': eval_kl,
                           'eval_ortho_loss': eval_ortho_loss,
                           'epoch': n_epoch})

            print(f'[Eval Loss]: {eval_loss:.4f}   [Eval MSE]: {eval_mse:.4f}  [Eval Ortho]: {eval_ortho_loss:.4f}')

            if best_mse > eval_loss:
                best_mse = eval_loss
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_mse}, self.file_path + '_best.pt')
                    print(f'Best model parameter saved at {n_epoch}')

            if n_epoch % 50 == 0:
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_mse}, self.file_path + f'_{n_epoch}.pt')


    def evaluation(self):
        self.model.eval()
        avg_eval_loss = 0.
        avg_eval_mse = 0.
        avg_kl = 0.
        avg_ortho_loss = 0.

        with torch.no_grad():
            for it, sample in enumerate(self.eval_dataloader):
                samp_sin = sample['sin'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss, ortho_loss = self.model(orig_ts, samp_sin)
                mse_loss = mse_loss.mean() ; kl_loss= kl_loss.mean() ; ortho_loss = ortho_loss.mean()
                loss = mse_loss + kl_loss + (0.01 * ortho_loss)
                # loss = mse_loss + (0.01 * ortho_loss)
                avg_eval_loss += (loss.item() * samp_sin.size(0))
                avg_eval_mse += (mse_loss.item() * samp_sin.size(0))
                avg_kl += (kl_loss.item() * samp_sin.size(0))
                avg_ortho_loss += (ortho_loss * samp_sin.size(0))

            avg_eval_loss /= self.eval_dataloader.dataset.__len__()
            avg_eval_mse /= self.eval_dataloader.dataset.__len__()
            avg_kl /= self.eval_dataloader.dataset.__len__()
            avg_ortho_loss /= self.eval_dataloader.dataset.__len__()
        return avg_eval_loss, avg_eval_mse, avg_kl, avg_ortho_loss




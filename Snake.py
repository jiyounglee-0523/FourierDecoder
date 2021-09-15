import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
import time
import os
import math
import wandb
from datetime import datetime
import argparse

from datasets.cond_dataset import get_dataloader
from utils.model_utils import count_parameters
from utils.trainer_utils import update_learning_rate, log

class Snake_fn(nn.Module):
    def __init__(self):
        super(Snake_fn, self).__init__()
        self.a = nn.Parameter()
        self.first = True

    def forward(self, x):
        if self.first:
            self.first = False
            a = torch.zeros_like(x[0]).normal_(mean=0,std=50).abs()
            self.a = nn.Parameter(a)
        return (x + (torch.sin(self.a * x) ** 2) / self.a)

class LearnableSnake(nn.Module):
    def __init__(self, args):
        super(LearnableSnake, self).__init__()

        self.linear1 = nn.Linear(1, 2*args.n_harmonics)
        self.linear2 = nn.Linear(2*args.n_harmonics, 1)
        #self.linear3 = nn.Linear(2*args.n_harmonics, 1)
        self.act1 = Snake_fn()
        #self.act2 = Snake_fn()

    def forward(self, x):
        # (1, S, 1)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        # x = self.act2(x)
        # x = self.linear3(x)
        return x

# ## for atmosphere (exact reproduction)
# class LearnableSnake(nn.Module):
#     def __init__(self, args):
#         super(LearnableSnake, self).__init__()
#
#         self.linear1 = nn.Linear(1, 100)
#         self.linear2 = nn.Linear(100, 100)
#         self.linear3 = nn.Linear(100, 1)
#         self.act1 = Snake_fn()
#         self.act2 = Snake_fn()
#
#     def forward(self, x):
#         # (1, S, 1)
#         x = self.linear1(x)
#         x = self.act1(x)
#         x = self.linear2(x)
#         x = self.act2(x)
#         x = self.linear3(x)
#         return x
class Fixed_Snake_fn(nn.Module):
    def __init__(self, a):
        super(Fixed_Snake_fn, self).__init__()
        self.a = a

    def forward(self, x):
        return x + ((torch.sin(self.a * x)) ** 2) / self.a


class FixedSnake(nn.Module):
    def __init__(self, args):
        super(FixedSnake, self).__init__()
        a = 1

        layers = []
        layers.append(nn.Linear(1, 2 * args.n_harmonics))
        layers.append(Fixed_Snake_fn(a=1))
        for i in range(args.n_layers - 2):
            layers.append(nn.Linear(2 * args.n_harmonics, 2 * args.n_harmonics))
            layers.append(Fixed_Snake_fn(a=1))
        layers.append(nn.Linear(2*args.n_harmonics, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # (1, S, 1)
        x = self.model(x)
        return x


class ModifiedFixedSnake(nn.Module):
    def __init__(self, args):
        super(ModifiedFixedSnake, self).__init__()
        self.n_harmonics = args.n_harmonics

        def snake(x):
            a = 1
            return x + ((torch.sin(a * x)) ** 2) / a

        self.linear1 = nn.Linear(1, 2*args.n_harmonics, bias=False)
        self.linear2 = nn.Linear(2*args.n_harmonics, 1, bias=False)
        self.act1 = snake

    def forward(self, x):
        # (1, S, 1)
        x = self.linear1(x)  # (1, S, 2H)
        x[:, :self.n_harmonics] = x[:, :self.n_harmonics] - (math.pi / 2)  # add bias -pi/2
        x = self.act1(x)
        x = self.linear2(x)
        return x



# # For marketindex (Exact Reproduction)
# class FixedSnake(nn.Module):
#     def __init__(self, args):
#         super(FixedSnake, self).__init__()
#
#         def snake(x):
#             a = 30
#             return x + ((torch.sin(a * x)) ** 2) / a
#
#         self.linear1 = nn.Linear(1, 64)
#         self.linear2 = nn.Linear(64, 64)
#         self.linear3 = nn.Linear(64, 64)
#         self.linear4 = nn.Linear(64, 1)
#         self.act1 = snake
#
#     def forward(self, x):
#         # (1, S, 1)
#         x = self.linear1(x)
#         x = self.act1(x)
#         x = self.linear2(x)
#         x = self.act1(x)
#         x = self.linear3(x)
#         x = self.act1(x)
#         x = self.linear4(x)
#         return x



class SnakeTrainer():
    def __init__(self, args):
        self.train_dataloader = get_dataloader(args, 'train')

        self.n_epochs = args.n_epochs

        self.debug = args.debug
        self.dataset_type = args.dataset_type
        self.n_harmonics = args.n_harmonics

        filename = f'{datetime.now().date()}_{args.model_type}_{args.n_layers}_{args.dataset_type}_{args.dataset_name}_{args.n_harmonics}_{args.notes}'

        args.filename = filename

        self.path = args.path + filename
        self.file_path = self.path + '/' + filename
        print(f'Model will be saved at {self.path}')

        if not self.debug:
            os.mkdir(self.path)
            self.logger = log(self.path + '/', file=filename+'.logs')

        if args.model_type == 'LearnableSnake':
            self.model = LearnableSnake(args=args).cuda()
        elif args.model_type == 'FixedSnake':
            self.model = FixedSnake(args=args).cuda()
        elif args.model_type == 'ModifiedFixedSnake':
            self.model = ModifiedFixedSnake(args=args).cuda()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        if not self.debug:
            self.logger.info(f'Number of parameters: {count_parameters(self.model)}')
            self.logger.info(f'Wandb Project Name: {args.dataset_type+args.dataset_name}')
            self.logger.info(f'Model will be saved at {self.path}')
            wandb.init(project=args.dataset_type+args.dataset_name, config=args, entity='fourierode')

        else:
            print(f'Number of parameters: {count_parameters(self.model)}')

    def train(self):
        best_mse = float('inf')

        for n_epoch in range(self.n_epochs):
            starttime = time.time()
            for it, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin = sample['sin'].cuda()  # (1, S, 1)
                orig_ts = sample['orig_ts'].cuda()

                predicted = self.model(orig_ts[0].unsqueeze(-1))    # (S, 1)
                mse_loss = nn.MSELoss()(predicted, samp_sin)

                mse_loss.backward()
                self.optimizer.step()

                if not self.debug:
                    wandb.log({'train_loss': mse_loss,
                               'lr': self.optimizer.param_groups[0]['lr'],
                               'epoch': n_epoch})

                    self.logger.info(f'[Train Loss]: {mse_loss:.4f}')
                else:
                    print(f'[Train Loss]: {mse_loss:.4f}')

            endtime = time.time()

            if not self.debug:
                self.logger.info(f'[Time]: {endtime-starttime}')
            else:
                print(f'[Time]: {endtime-starttime}')

            if best_mse > mse_loss:
                best_mse = mse_loss
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': best_mse}, self.file_path + '_best.pt')
                    self.logger.info(f'Best model parameter saved at {n_epoch}')

            if n_epoch % 50 == 0:
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': mse_loss}, self.file_path + f'_{n_epoch}.pt')

                    # if n_epoch != 0:
                    #     update_learning_rate(self.optimizer, decay_rate=0.99, lowest=1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['FixedSnake', 'ModifiedFixedSnake', 'LearnableSnake'], default='FixedSnake')
    parser.add_argument('--n_harmonics', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=2)

    # trainer
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true')

    # dataloader
    parser.add_argument('--path', type=str, default='./', help='parameter saving path')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--dataset_type', choices=['sin', 'ECG', 'NSynth', 'GP', 'atmosphere', 'marketindex', 'ECG_Onesample'])
    parser.add_argument('--notes', type=str, default='example')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    trainer = SnakeTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()





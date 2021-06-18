import torch

import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
import time

from datasets.cond_dataset import get_dataloader
from models.NeuralProcess import ConditionalFNP
from utils.model_utils import count_parameters

class ConditionalBaseTrainer():
    def __init__(self, args):
        self.train_dataloader = get_dataloader(args, 'train')
        self.eval_dataloader = get_dataloader(args, 'eval')
        self.n_epochs = args.n_epochs

        self.debug = args.debug
        self.dataset_type = args.dataset_type
        self.path = args.path + args.filename + '.pt'
        print(f'Model will be saved at {self.path}')

        if os.path.exists(self.path):
            print(self.path)
            raise OSError('saving directory already exists')


class ConditionalNPTrainer(ConditionalBaseTrainer):
    def __init__(self, args):
        super(ConditionalNPTrainer, self).__init__(args)

        self.model = ConditionalFNP(args).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        print(f'Number of parameters: {count_parameters(self.model)}')
        print(f'Description: {str(args.description)}')

        if not self.debug:
            wandb.init(project='conditionalODE')
            wandb.config.update(args)
            wandb.watch(self.model, log='all')

    def train(self):
        best_mse = float('inf')

        for n_epoch in range(self.n_epochs):
            starttime = time.time()
            for iter, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin = sample['sin'].cuda()
                freq = sample['freq'].cuda()
                amp = sample['amp'].cuda()
                label = sample['label'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss = self.model(orig_ts, samp_sin, label)
                loss = mse_loss + kl_loss
                loss.backward()
                self.optimizer.step()

                if best_mse > loss:
                    best_mse = loss
                    if not self.debug:
                        torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_mse}, self.path)
                        print(f'Model parameter saved at {n_epoch}')

                if not self.debug:
                    wandb.log({'train_loss': loss,
                               'train_kl_loss': kl_loss,
                               'train_mse_loss': mse_loss})

                    if self.dataset_type == 'ECG':
                        raise NotImplementedError
                    elif self.dataset_type == 'sin':
                        self.sin_result_plot(samp_sin[0], orig_ts[0], freq[0], amp[0], label[0])



import torch
import torch.nn as nn

import os
import matplotlib.pyplot as plt
import time
import wandb
import numpy as np
import random

from models.NeuralProcess import AttentiveNP, FNP
from utils.model_utils import count_parameters
from datasets.sinusoidal_dataloader import get_dataloader

class Trainer():
    def __init__(self, args):
        self.train_dataloader = get_dataloader(args)
        self.n_epochs = args.n_epochs
        self.dataset_type = args.dataset_type

        if args.model_type == 'NP':
            self.model = AttentiveNP(args.latent_dimension).cuda()
        elif args.model_type == 'FNP':
            self.model = FNP(args).cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.path = args.path + args.filename + '.pt'
        print('start training')
        print(f'number of params: {count_parameters(self.model)}')

        wandb.init(project='generativeode')
        wandb.config.update(args)

    def bucketing(self, x):
        cpu_x = x.cpu()[0]
        bins = np.linspace(-2, 2, 50)
        inds = np.digitize(cpu_x, bins)

        k = 20
        sample_idxs = []
        for bucket in bins:
            idxs = np.where(bins[inds] == bucket)[0]
            if len(idxs) < k:
                sample_idxs.extend(idxs)
            else:
                sample_idxs.extend(random.sample(list(idxs), k))

        return sorted(sample_idxs)


    def train(self):
        print(f'file name {self.path}')
        best_mse = float('inf')

        for n_epoch in range(self.n_epochs):
            starttime = time.time()
            for iter, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                if self.dataset_type == 'dataset9':
                    samp_sin, samp_ts = sample
                    sample_indxs = self.bucketing(samp_sin)
                    samp_ts = samp_ts[:, sample_indxs]
                    samp_sin = samp_sin[:, sample_indxs]
                    context_x = samp_ts.clone() ; target_x = samp_ts.clone()
                    context_y = samp_sin.clone() ;  target_y = samp_sin.clone()
                else:
                    samp_sin, samp_ts, amps = sample
                    samp_sin = samp_sin.squeeze(-1)
                    context_index = sorted(random.sample(range(0, samp_ts.size(1)), 150))
                    target_index = sorted(random.sample(range(0, samp_ts.size(1)), 150))
                    target_index.extend(context_index)
                    target_index = list(set(target_index))
                    context_x = samp_ts[:, context_index];
                    context_y = samp_sin[:, context_index]
                    target_x = samp_ts[:, target_index] ; target_y = samp_sin[:, target_index]

                context_x = context_x.cuda() ; target_x = target_x.cuda()
                context_y = context_y.cuda() ; target_y = target_y.cuda()


                #content_index = sorted(random.sample(range(0, ), 150))
                # content_index = sorted(random.sample(range(0, samp_sin.size(1)), 150))
                # target_index = sorted(list(set(list(range(0, len(sample_indxs)))) - set(content_index)))

                # content_x = samp_ts[:, content_index] ; content_y = samp_sin[:, content_index]
                # #target_x = samp_ts[:, target_index] ; target_y = samp_sin[:, target_index]

                mu, sigma, mse_loss, kl_loss, loss = self.model(context_x, context_y, target_x, target_y)
                loss.backward()
                self.optimizer.step()

                if best_mse > loss:
                    best_mse = loss
                    torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_mse}, self.path)
                    print(f'model parameter saved at {n_epoch}')

                wandb.log({'train_loss': loss,
                           'train_kl_loss': kl_loss,
                           'train_mse_loss': mse_loss})
                if self.dataset_type == 'dataset9':
                    self.ECG_result_plot(samp_sin, samp_ts)
                else:
                    self.sin_result_plot(samp_sin[0], samp_ts[0])
            endtime = time.time()
            print('time consuming: ', endtime-starttime)

    def ECG_result_plot(self, samp_sin, samp_ts):
        self.model.eval()
        cycle = 9
        test_ts = torch.Tensor(np.linspace(0, cycle, 360*cycle)).unsqueeze(0).to(samp_sin.device)
        # start = 0.
        # stop = 9. * np.pi
        # test_ts = torch.linspace(start, stop, 900).unsqueeze(0).cuda()
        mu, sigma, mse_loss, kl_loss, loss = self.model(samp_ts[0].unsqueeze(0), samp_sin[0].unsqueeze(0), test_ts)

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(samp_ts[0].squeeze().cpu().numpy(), samp_sin[0].squeeze().detach().cpu().numpy(), 'g', label='true trajectory')
        ax.plot(test_ts.squeeze().cpu().numpy(), mu.squeeze().detach().cpu().numpy(), 'r', label='learned trajectory')
        #ax.axvline(samp_ts[-1])
        wandb.log({'predict': wandb.Image(plt)})
        print('mu', mu[0][:10])

        plt.close('all')

    def sin_result_plot(self, samp_sin, samp_ts):
        samp_sin = samp_sin.unsqueeze(0).cuda()
        test_ts = torch.Tensor(np.linspace(0, 25, 2700)).unsqueeze(0).to(samp_sin.device)
        mu, sigma, mse_loss, kl_loss, loss = self.model(samp_ts.unsqueeze(0).cuda(), samp_sin, test_ts)

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(samp_ts.squeeze().cpu().numpy(), samp_sin.squeeze().detach().cpu().numpy(), 'g', label='true trajectory')
        ax.plot(test_ts.squeeze().cpu().numpy(), mu.squeeze().detach().cpu().numpy(), 'r', label='learned trajectory')
        wandb.log({'predict': wandb.Image(plt)})
        plt.close('all')



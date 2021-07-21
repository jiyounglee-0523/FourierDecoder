import os
import torch
import torch.nn as nn
import numpy as np

import wandb

from utils.model_utils import count_parameters, plot_grad_flow
from models.RNNODE import LatentNeuralDE
import matplotlib.pyplot as plt


class Trainer():
    def __init__(self, args, train_dataloader):
        self.train_dataloader = train_dataloader
        self.n_epochs = args.n_epochs

        self.model = LatentNeuralDE(input_dim=1, latent_dim=6, rnn_hidden_dim=32, n_harmonics=args.n_harmonics).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.path = args.path +  args.filename + '.pt'

        print('Number of params: {}'.format(count_parameters(self.model)))

        wandb.init(project='generativeode')
        wandb.config.update(args)

        # ckpt = torch.load(self.path)
        # self.model.load_state_dict(ckpt['model_state_dict'])

    def train(self):
        best_mse = float('inf')
        for n_epoch in range(self.n_epochs + 1):
            for iter, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin, samp_ts, _ = sample
                samp_sin = samp_sin.cuda() ; samp_ts = samp_ts.cuda()

                train_loss = self.model(samp_sin, samp_ts)
                train_loss.backward()

                plot_grad_flow(self.model.named_parameters())
                self.optimizer.step()

                print('epoch: {},  mse_loss: {}'.format(n_epoch, train_loss))

                if best_mse > train_loss:
                    best_mse = train_loss
                    #torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_mse}, self.path)
                    print('model parameter saved at epoch {}'.format(n_epoch))

                wandb.log({'train_loss': train_loss,
                           'best_mse': best_mse})

                self.result_plot(samp_sin[0], samp_ts[0])
                break
            break

    def result_plot(self, samp_sin, samp_ts):
        samp_sin = samp_sin.unsqueeze(0)
        test_ts = torch.Tensor(np.linspace(0., 25. * np.pi, 2700)).unsqueeze(0).to(samp_sin.device)
        output = self.model.predict(samp_sin, samp_ts.unsqueeze(0))

        test_tss = test_ts.squeeze()

        # plot output
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(samp_ts.cpu().numpy(), samp_sin.squeeze().detach().cpu().numpy(), 'g', label='original')
        ax.plot(samp_ts.cpu().numpy(), output.squeeze().detach().cpu().numpy(), 'r', label='learned trajectory')

        ax.axvline(samp_ts[-1])
        wandb.log({'predict': wandb.Image(plt)})
        plt.savefig('./example.png')
        plt.close('all')
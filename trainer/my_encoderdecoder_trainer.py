import os
import torch
import torch.nn as nn
import numpy as np

import wandb
import matplotlib.pyplot as plt
from utils.model_utils import count_parameters, plot_grad_flow
from models.my_encoder_decoder import LatentNeuralDE

class Trainer():
    def __init__(self, args, train_dataloader):
        self.train_dataloader = train_dataloader
        self.n_epochs = args.n_epochs

        self.model = LatentNeuralDE(args=args, input_dim=1, latent_dim=6, rnn_hidden_dim=32).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.path = args.path + args.filename + '.pt'
        print('number of params: {}'.format(count_parameters(self.model)))
        print('descriptoin: {}'.format(str(args.description)))

        wandb.init(project='generativeode')
        wandb.config.update(args)
        wandb.watch(self.model, log='all')

    def train(self):
        print('filename', self.path)
        best_mse = float('inf')

        for n_epoch in range(self.n_epochs + 1):
            for iter, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin, samp_ts, _ = sample
                samp_sin = samp_sin.cuda() ; samp_ts = samp_ts.cuda() #; latent_v = latent_v.cuda()

                train_loss = self.model(samp_ts, samp_sin)
                self.result_plot(samp_sin[0], samp_ts[0])

                train_loss.backward()
                plot_grad_flow(self.model.named_parameters())
                self.optimizer.step()

                if best_mse > train_loss:
                    best_mse = train_loss
                    torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_mse}, self.path)
                    print('model parameter saved at epoch {}'.format(n_epoch))

                wandb.log({'train_loss': train_loss,
                           'best_mse': best_mse})

                print('epoch: {},  mse_loss: {}'.format(n_epoch, train_loss))


    def result_plot(self, samp_sin, samp_ts):
        samp_sin = samp_sin.unsqueeze(0);
        #latent_v = latent_v.unsqueeze(0)
        test_ts = torch.Tensor(np.linspace(0., 25 * np.pi, 2700)).unsqueeze(0).to(samp_sin.device)

        output = self.model.predict(test_ts, samp_sin)
        test_tss = test_ts.squeeze()
        #real_output = amp * (-4 * torch.sin(test_tss) + torch.sin(2 * test_tss) - torch.cos(test_tss) + 0.5 * torch.cos(2 * test_tss))

        # plot output
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(samp_ts.squeeze().cpu().numpy(), samp_sin.squeeze().detach().cpu().numpy(), 'g', label='true trajectory')
        ax.plot(test_ts.squeeze().cpu().numpy(), output.squeeze().detach().cpu().numpy(), 'r', label='learned trajectory')
        ax.axvline(samp_ts[-1])

        wandb.log({"predict": wandb.Image(plt)})

        plt.close('all')

import os
import torch
import torch.nn as nn

import wandb

from utils.model_utils import count_parameters
from models.generative_ode import GalerkinDE

class Trainer():
    def __init__(self, args, train_dataloader):
        self.train_dataloader = train_dataloader
        self.n_epochs = args.n_epochs

        self.model = GalerkinDE(args).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.path = args.path + args.filename + '.pt'
        print('start training!')
        print('number of params: {}'.format(count_parameters(self.model)))

        print('dataset_type: {}'.format(str(args.dataset_type)))
        print('description: {}'.format(str(args.description)))

        wandb.init(project='generativeode')
        wandb.config.update(args)
        wandb.watch(self.model, log='all')

    def train(self):
        print('filename: {}'.format(self.path))

        best_mse = float('inf')
        if os.path.exists(self.path):
            ckpt = torch.load(self.path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            best_mse = ckpt['loss']
            print('loaded saved parameters')

        for n_epoch in range(self.n_epochs):
            for iter, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad()
                samp_sin, samp_ts, latent_v = sample
                samp_sin = samp_sin.cuda() ; samp_ts = samp_ts.cuda() ; latent_v = latent_v.cuda()

                train_loss = self.model(samp_ts, samp_sin, latent_v)

                train_loss.backward()
                self.optimizer.step()

                if best_mse > train_loss:
                    best_mse = train_loss
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'loss': best_mse}, self.path)
                    print('model parameter saved at epoch {}'.format(n_epoch))

                wandb.log({'train_loss': train_loss,
                           'best_mse': best_mse})

            print('epoch: {},  mse_loss: {}'.format(n_epoch, train_loss))







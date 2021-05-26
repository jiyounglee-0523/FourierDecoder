import torch

import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
import time

from models.wrapper import LatentNeuralDE
from datasets.sinusoidal_dataloader import get_dataloader
from utils.model_utils import count_parameters

class Trainer():
    def __init__(self, args):
        self.train_dataloader = get_dataloader(args)
        self.n_epochs = args.n_epochs

        self.model = LatentNeuralDE(args).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        self.path = args.path + args.filename + '.pt'

        if os.path.exists(self.path):
            print(self.path)
            raise RuntimeError('saving directory already exsits')

        print(f'number of parameter: {count_parameters(self.model)}')
        print(f'description: {str(args.description)}')

        wandb.init(project='generativeode')
        wandb.config.update(args)
        wandb.watch(self.model, log='all')

    def train(self):
        print(f'Model will be saved at {self.path}')
        best_mse = float('inf')

        for n_epoch in range(self.n_epochs):
            starttime = time.time()
            for iter, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin, samp_ts = sample
                samp_sin = samp_sin.cuda() ; samp_ts = samp_ts.cuda()

                train_loss = self.model(samp_ts, samp_sin)
                train_loss.backward()
                self.optimizer.step()

                if best_mse > train_loss:
                    best_mse = train_loss
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'loss': best_mse}, self.path)

                wandb.log({'train_loss': train_loss,
                           'best_mse': best_mse})
                print(f'epoch: {n_epoch}, mse_loss {train_loss}')

                self.result_plot(samp_sin[0], samp_ts[0])

            endtime = time.time()
            print(f'time consuming {endtime-starttime}')


    def result_plot(self, samp_sin, samp_ts):
        samp_sin = samp_sin.unsqueeze(0)
        cycle = 9
        test_ts = torch.Tensor(np.linspace(0, cycle, 360*cycle)).unsqueeze(0).to(samp_sin.device)

        output = self.model.predict(test_ts, samp_sin)
        test_tss = test_ts.squeeze()

        # plot output
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(samp_ts.squeeze().cpu().numpy(), samp_sin.squeeze().detach().cpu().numpy(), 'g', label='true trajectory')
        ax.plot(test_tss.cpu().numpy(), output.squeeze().detach().cpu().numpy(), 'r', label='learned trajectory')
        ax.axvline(samp_ts[-1])

        wandb.log({'predict': wandb.Image(plt)})
        plt.close('all')





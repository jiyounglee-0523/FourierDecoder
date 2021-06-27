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
        self.test_dataloder = get_dataloader(args, 'test')
        self.n_epochs = args.n_epochs

        self.debug = args.debug
        self.dataset_type = args.dataset_type
        self.path = args.path + args.dataset_type + '_' + args.filename + '.pt'
        print(f'Model will be saved at {self.path}')

        # if os.path.exists(self.path):
        #     print(self.path)
        #     raise OSError('saving directory already exists')

    def sin_result_plot(self, samp_sin, orig_ts, freq, amp, label):
        self.model.eval()

        # reconstruction
        samp_sin = samp_sin.unsqueeze(0)
        orig_ts = orig_ts.unsqueeze(0)
        test_tss = torch.Tensor(np.linspace(0, 5, 400)).to(samp_sin.device)   # (1, 1, S)
        with torch.no_grad():
            decoded_traj = self.model.predict(orig_ts, samp_sin, label.unsqueeze(0), test_tss.unsqueeze(0))

        test_ts = test_tss.cpu().numpy()
        orig_sin = amp[0] * np.sin(freq[0] * test_ts* 2 * np.pi) + amp[1] * np.sin(freq[1] * test_ts * 2 * np.pi) + amp[2] * np.sin(freq[2]*test_ts*2*np.pi) +\
            amp[3]*np.cos(freq[3]*test_ts*2*np.pi) + amp[4]*np.cos(freq[4]*test_ts*2*np.pi) + amp[5]*np.cos(freq[5]*test_ts*2*np.pi)

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(test_ts, orig_sin.cpu().numpy(), 'g', label='true trajectory')
        ax.scatter(orig_ts.cpu().numpy(), samp_sin[0].squeeze(-1).cpu().numpy(), s=5, label='sampled points')
        ax.plot(test_ts, decoded_traj.squeeze().detach().cpu().numpy(), 'r', label='learned trajectory')
        if not self.debug:
            wandb.log({'reconstruction': wandb.Image(plt)})
        plt.close('all')

        # inference - random sampling
        generated_traj = self.model.inference(test_tss.unsqueeze(0), label)
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(test_ts, generated_traj.squeeze().detach().cpu().numpy(), 'g', label='inference')
        plt.title(label)
        if not self.debug:
            wandb.log({'random sampling': wandb.Image(plt)})
        plt.close('all')



class ConditionalNPTrainer(ConditionalBaseTrainer):
    def __init__(self, args):
        super(ConditionalNPTrainer, self).__init__(args)

        self.model = ConditionalFNP(args).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        print(f'Number of parameters: {count_parameters(self.model)}')
        print(f'Description: {str(args.notes)}')

        # if os.path.exists(self.path):
        #     ckpt = torch.load(self.path)
        #     self.model.load_state_dict(ckpt['model_state_dict'])
        #     print(f'Loaded parameter from {self.path}')

        if not self.debug:
            wandb.init(project='conditionalODE')
            wandb.config.update(args)
            wandb.watch(self.model, log='all')

    def train(self):
        best_mse = float('inf')
        if os.path.exists(self.path):
            ckpt = torch.load(self.path)
            best_mse = ckpt['loss']

        for n_epoch in range(self.n_epochs):
            starttime = time.time()
            for iter, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin = sample['sin'].cuda()
                freq = sample['freq']
                amp = sample['amp']
                label = sample['label'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss = self.model(orig_ts, samp_sin, label, sampling=True)
                loss = mse_loss + kl_loss
                loss.backward()
                self.optimizer.step()

                if not self.debug:
                    wandb.log({'train_loss': loss,
                               'train_kl_loss': kl_loss,
                               'train_mse_loss': mse_loss})

            if self.dataset_type == 'ECG':
                raise NotImplementedError
            elif self.dataset_type == 'sin':
                self.sin_result_plot(samp_sin[0], orig_ts[0], freq[0], amp[0], label[0])

            endtime = time.time()
            print(f'[Time] : {endtime-starttime}')

            eval_loss, eval_mse, eval_kl = self.evaluation()
            if not self.debug:
                wandb.log({'eval_loss': eval_loss,
                           'eval_mse': eval_mse,
                           'eval_kl': eval_kl})

            if best_mse > eval_loss:
                best_mse = eval_loss
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_mse}, self.path)
                    print(f'Model parameter saved at {n_epoch}')

        self.test()


    def evaluation(self):
        self.model.eval()
        avg_eval_loss = 0.
        avg_eval_mse = 0.
        avg_kl = 0.
        with torch.no_grad():
            for iter, sample in enumerate(self.eval_dataloader):
                samp_sin = sample['sin'].cuda()
                label = sample['label'].cuda()
                orig_ts = sample['orig_ts'].cuda()

                mse_loss, kl_loss = self.model(orig_ts, samp_sin, label, sampling=False)
                loss = mse_loss + kl_loss
                avg_eval_loss += (loss.item() / len(self.eval_dataloader))
                avg_eval_mse += (mse_loss.item() / len(self.eval_dataloader))
                avg_kl += (kl_loss.item() / len(self.eval_dataloader))

        return avg_eval_loss, avg_eval_mse, avg_kl


    def test(self):
        self.model.eval()
        ckpt = torch.load(self.path)
        self.model.load_state_dict(ckpt['model_state_dict'])

        avg_test_loss = 0.
        avg_test_mse = 0.
        avg_kl = 0.
        with torch.no_grad():
            for iter, sample in enumerate(self.test_dataloder):
                samp_sin = sample['sin'].cuda()
                label = sample['label'].cuda()
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




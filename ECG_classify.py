import torch
import torch.nn as nn

import os
import numpy as np
import wandb
import argparse
import random

from datasets.cond_dataset import get_dataloader

class ConvClassify(nn.Module):
    def __init__(self, args):
        super(ConvClassify, self).__init__()

        self.conv = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, stride=3),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=3),
                                  nn.ReLU(),
                                  nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=3))
        self.glob_pool = nn.AdaptiveAvgPool1d(1)
        self.output_fc = nn.Linear(128, args.num_label)  # check the dimension

    def forward(self, x):
        # x shape of (B, S, 1),
        output = self.conv(x.permute(0, 2, 1))
        output = self.glob_pool(output).squeeze(-1)
        output = self.output_fc(output)
        return output


class ECGTrainer():
    def __init__(self, args):
        self.train_dataloader = get_dataloader(args, 'train')
        self.eval_dataloader = get_dataloader(args, 'eval')
        self.n_epochs = args.n_epochs
        self.debug = args.debug

        self.dataset_type = args.dataset_type
        self.path = args.path + args.dataset_type + '_' + args.filename + '.pt'
        print(f'Model will be saved at {self.path}')

        self.model = ConvClassify(args).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        if not args.debug:
            wandb.init(project='conditionalODE', config=args)

    def train(self):
        best_mse = float('inf')
        for n_epoch in range(self.n_epochs):
            for iter, sample in enumerate(self.train_dataloader):
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                samp_sin = sample['sin'].cuda()
                label = sample['label'].cuda()

                output = self.model(samp_sin)  # (B, C)
                loss = nn.CrossEntropyLoss()(output, label.squeeze(-1))
                loss.backward()
                self.optimizer.step()

                if not self.debug:
                    wandb.log({'train_loss': loss})

            eval_loss = self.evaluation()
            if not self.debug:
                wandb.log({'eval_loss': loss})

            if best_mse > eval_loss:
                best_mse = loss
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(), 'loss': best_mse}, self.path)
                    print(f'Model parameter saved at {n_epoch}')


    def evaluation(self):
        self.model.eval()
        avg_loss = 0.
        with torch.no_grad():
            for iter, sample in enumerate(self.eval_dataloader):
                samp_sin = sample['sin'].cuda()
                label = sample['label'].cuda()

                output = self.model(samp_sin)
                loss = nn.CrossEntropyLoss()(output, label.squeeze(-1))
                avg_loss = (loss.item() / len(self.eval_dataloader))

        return avg_loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--path', type=str, default='/home/jylee/data/generativeODE/output/ECG/', help='parameter saving path')
    parser.add_argument('--dataset_path', type=str, default='/home/jylee/data/generativeODE/input/not_duplicatedECG/')
    parser.add_argument('--filename', type=str, default='test')
    parser.add_argument('--dataset_type', choices=['sin', 'ECG', 'NSynth'], default='ECG')
    parser.add_argument('--ECG_type', choices=['V1', 'V6'], default='V6')
    parser.add_argument('--notes', type=str, default='example')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device_num', type=str, default='0')
    args = parser.parse_args()

    if args.dataset_type == 'ECG':
        args.num_label = 3

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    trainer = ECGTrainer(args)
    trainer.train()



if __name__ == '__main__':
    main()

import torch

import numpy as np
import argparse
from datetime import datetime
import os
import wandb
import random

from models.latentmodel import AEAttnFNP
from utils.model_utils import count_parameters
from utils.trainer_utils import update_learning_rate

class TempTrainer():
    def __init__(self, args):
        self.n_epochs = args.n_epochs
        self.run_continue = args.run_continue

        self.debug = args.debug
        self.dataset_type = args.dataset_type
        self.n_harmonics = args.n_harmonics
        if args.attn and not args.query:
            if args.period:
                filename = f'{datetime.now().date()}_{args.dataset_name}_attn_period_{args.encoder}_{args.stride}_{args.encoder_blocks}layer_decoder{args.decoder_layers}_{args.decoder_hidden_dim}_harmonics{args.n_harmonics}'
            elif not args.period:
                filename = f'{datetime.now().date()}_{args.dataset_name}_attn_{args.encoder}_{args.stride}_{args.encoder_blocks}layer_decoder{args.decoder_layers}_{args.decoder_hidden_dim}_harmonics{args.n_harmonics}'
        elif args.query and not args.attn:
            filename = f'{datetime.now().date()}_{args.dataset_name}_query_{args.encoder}_{args.stride}_{args.encoder_blocks}layer_decoder{args.decoder_layers}_{args.decoder_hidden_dim}'
        args.filename = filename

        self.path = args.path + filename
        self.file_path = self.path + '/' + filename
        print(f'Model will be saved at {self.path}')
        if not self.debug:
            os.mkdir(self.path)

        self.model = AEAttnFNP(args).cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        print(f'Number of parameters: {count_parameters(self.model)}')

        if not self.debug:
            wandb.init(project=args.dataset_type+args.dataset_name, config=args, entity='fourierode')

    def train(self):
        best_mse = float('inf')
        L = 6
        atmos_data = np.load('/home/edlab/jylee/generativeODE/input/atmosphere/minamitorishima.npy')
        atmos_data = torch.FloatTensor(atmos_data)[:52*L]
        atmos_data = atmos_data.unsqueeze(0).unsqueeze(-1).cuda()  # (1, S, 1)
        time_span = torch.linspace(0, 10, 520)[:52*L]
        time_span = time_span.unsqueeze(0).cuda()   # (1, S)


        self.model.train()
        for n_epoch in range(self.n_epochs):
            self.optimizer.zero_grad(set_to_none=True)

            mse_loss, ortho_loss = self.model(time_span, atmos_data)
            loss = mse_loss + (0.01*ortho_loss)
            loss.backward()
            self.optimizer.step()

            if not self.debug:
                wandb.log({'train_loss': loss,
                           'train_mse_loss': mse_loss,
                           'train_ortho_loss': ortho_loss,
                           'lr': self.optimizer.param_groups[0]['lr'],
                           'epoch': n_epoch})

            print(f'[Train Loss]: {loss:.4f}     [Train MSE]: {mse_loss:.4f}      [Train Ortho]: {ortho_loss:.4f}')

            if best_mse > loss:
                best_mse = loss
                if not self.debug:
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                               'loss': best_mse}, self.file_path + '_best.pt')
                    print(f'Best model parameter saved at {n_epoch}')

            if n_epoch % 50 == 0:
                if n_epoch != 0:
                    update_learning_rate(self.optimizer, decay_rate=0.999, lowest=1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['FNODEs', 'FNP', 'NP', 'NODEs'], default='FNP')

    # Encoder
    parser.add_argument('--encoder', choices=['Transformer', 'Conv', 'TransConv'])
    parser.add_argument('--encoder_embedding_dim', type=int, default=128)
    parser.add_argument('--encoder_hidden_dim', type=int, default=256)
    parser.add_argument('--encoder_attnheads', type=int, default=1, help='for transformer encoder')
    parser.add_argument('--encoder_blocks', type=int, default=3, help='number of layers in Conv')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--maxpool_kernelsize', type=int, default=2)

    # Decoder
    parser.add_argument('--decoder_layers', type=int, default=2)
    parser.add_argument('--decoder_hidden_dim', type=int, default=256)
    parser.add_argument('--in_features', type=int, default=1)
    parser.add_argument('--out_features', type=int, default=1)
    parser.add_argument('--latent_dimension', type=int, default=256)
    parser.add_argument('--expfunc', type=str, default='fourier')
    parser.add_argument('--n_harmonics', type=int)
    parser.add_argument('--n_eig', type=int, default=2)
    parser.add_argument('--lower_bound', type=float, default=1)
    parser.add_argument('--upper_bound', type=float)
    parser.add_argument('--skip_step', type=int)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--path', type=str, default='./', help='parameter saving path')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dataset_name', type=str, default='default')
    # parser.add_argument('--filename', type=str, default='test')
    parser.add_argument('--dataset_type', choices=['sin', 'ECG', 'NSynth', 'atmosphere'])
    parser.add_argument('--notes', type=str, default='example')
    parser.add_argument('--device_num', type=str, default='0')
    parser.add_argument('--query', action='store_true')
    parser.add_argument('--attn', action='store_true')
    parser.add_argument('--period', action='store_true')
    parser.add_argument('--run_continue', action='store_true')
    args = parser.parse_args()

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    trainer = TempTrainer(args)
    trainer.train()



if __name__ == '__main__':
    main()
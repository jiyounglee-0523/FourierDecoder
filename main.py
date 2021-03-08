import torch
import torch.nn as nn

import argparse
import random
import numpy as np

from datasets.sinusoidal_dataloader import get_dataloader
#from trainer.base_trainer import Trainer
#from trainer.dilation_param_trainer import Trainer
from trainer.fine_grain_trainer import Trainer
#from trainer.encoderdecoder_trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_features', type=int, default=1)
    parser.add_argument('--out_features', type=int, default=1)
    parser.add_argument('--latent_dimension', type=int, default=3)
    parser.add_argument('--expfunc', type=str, default='fourier')
    parser.add_argument('--n_harmonics', type=int, default=1)
    parser.add_argument('--n_eig', type=int, default=2)
    parser.add_argument('--zero_out', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--filename', type=str, default='test')
    parser.add_argument('--dataset_type', type=str, default='dataset2')
    parser.add_argument('--description', type=str, default='example')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=512)

    args = parser.parse_args()
    # parameters will be saved in 'path + filename + '.pt'
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    train_dataloader = get_dataloader(args)

    trainer = Trainer(args, train_dataloader)
    trainer.train()

if __name__ == '__main__':
    main()
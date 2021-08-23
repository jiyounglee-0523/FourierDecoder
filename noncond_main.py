import torch

import argparse
import random
import numpy as np
import os

from trainer.UnconditionalTrianer import UnconditionalAETrainer as Trainer


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
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--orthonormal_loss', action='store_true')


    parser.add_argument('--path', type=str, default='./', help='parameter saving path')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--dataset_name', type=str, default='default')
    # parser.add_argument('--filename', type=str, default='test')
    parser.add_argument('--dataset_type', choices=['sin', 'sin_onesample', 'ECG', 'NSynth', 'GP', 'atmosphere', 'marketindex'])
    parser.add_argument('--notes', type=str, default='example')
    parser.add_argument('--device_num', type=str, default='0')
    parser.add_argument('--query', action='store_true')
    parser.add_argument('--attn', action='store_true')
    parser.add_argument('--period', action='store_true')
    parser.add_argument('--NP_model', action='store_true')
    parser.add_argument('--run_continue', action='store_true')
    args = parser.parse_args()

    # if args.dataset_type == 'sin':
    #     args.num_label = 4
    # elif args.dataset_type == 'NSynth':
    #     args.num_label = 3
    # elif args.dataset_type == 'ECG':
    #     args.num_label = 3

    assert ((args.query and args.attn) is False) and ((args.query or args.attn) is True), "The model should be either query or attention"

    if args.debug:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # check n_harmonics corresponds to lower and upper bound
    assert ((args.upper_bound - args.lower_bound + 1) == args.n_harmonics), "the number of harmonics and lower and upper bound should match"

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
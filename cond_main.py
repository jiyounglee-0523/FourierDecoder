import torch

import argparse
import random
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_model', choices=['NODE', 'NP'], default='NODE', help='NP = transformer for both encoder and decoder')
    parser.add_argument('--model_type', choices=['FNODEs', 'FNP', 'NP', 'NODEs'], default='FNODEs')
    parser.add_argument('--encoder', choices=['RNNODE', 'Transformer', 'BiRNN'], default=None)

    # Encoder
    parser.add_argument('--encoder_embedding_dim', type=int, default=1, help='1 for RNN 32 for Transformer')
    parser.add_argument('--encoder_hidden_dim', type=int, default=32)
    parser.add_argument('--encoder_attnheads', type=int, default=1, help='for transformer encoder')
    parser.add_argument('--encoder_blocks', type=int, default=2, help='for transformer encoder')
    parser.add_argument('--data_length', type=int, default=500)

    # Decoder
    parser.add_argument('--in_features', type=int, default=1)
    parser.add_argument('--out_features', type=int, default=1)
    parser.add_argument('--latent_dimension', type=int, default=3, help='dimension for NP')
    parser.add_argument('--expfunc', type=str, default='fourier')
    parser.add_argument('--n_harmonics', type=int, default=1)
    parser.add_argument('--n_eig', type=int, default=2)
    parser.add_argument('--lower_bound', type=float, default=1)
    parser.add_argument('--upper_bound', type=float)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=int, default=0.1)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--path', type=str, default='./', help='parameter saving path')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--filename', type=str, default='test')
    parser.add_argument('--dataset_type', choices=['sin', 'ECG', 'NSynth'])
    parser.add_argument('--description', type=str, default='example')
    parser.add_argument('--device_num', type=str, default='0')
    args = parser.parse_args()

    if args.dataset_type == 'sin':
        args.num_label = 4
    else:
        raise NotImplementedError

    if args.model_type == 'FNP':
        from trainer.ConditionalTrainer import ConditionalNPTrainer as Trainer
    else:
        raise NotImplementedError

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

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
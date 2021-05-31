import torch
import torch.nn as nn

import argparse
import random
import numpy as np

from datasets.sinusoidal_dataloader import get_dataloader
#from trainer.base_trainer import Trainer
#from trainer.dilation_param_trainer import Trainer
#from trainer.fine_grain_trainer import Trainer
from trainer.fine_grain_recon_trainer import Trainer
#from trainer.my_encoderdecoder_trainer import Trainer
#from trainer.dilation_test_trainer import Trainer
#from trainer.enc_dec_trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_model', choices=['NODE', 'NP'], default='NODE', help='NP = transformer for both encoder and decoder')
    parser.add_argument('--model_type', choices=['FNODEs', 'NODEs', 'ANODEs', 'SONODEs'], default='FNODEs')
    parser.add_argument('--encoder', choices=['RNNODE', 'Transformer', 'BiRNN'], default=None)

    # Encoder
    parser.add_argument('--encoder_embedding_dim', type=int, default=1, help='1 for RNN 32 for Transformer')
    parser.add_argument('--encoder_hidden_dim', type=int, default=32)
    parser.add_argument('--encoder_output_dim', type=int, default=3)
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
    parser.add_argument('--lower_bound', type=float)
    parser.add_argument('--upper_bound', type=float)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=int, default=0.1)

    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--filename', type=str, default='test')
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--description', type=str, default='example')
    args = parser.parse_args()
    # parameters will be saved in 'path + filename + '.pt'

    # if args.encoder is None:
    #     from trainer.fine_grain_recon_trainer import Trainer
    # else:
    #     from trainer.my_encoderdecoder_trainer import Trainer
    if args.test_model == 'NODE':
        from trainer.base_trainer import Trainer
    elif args.test_model == 'NP':
        from trainer.NP_trainer import Trainer

    #assert args.encoder_output_dim == args.latent_dimension, 'output of encoder should have the same dimension as latent dimension'

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
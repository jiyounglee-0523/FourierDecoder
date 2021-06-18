import torch
import torch.nn as nn

import numpy as np
import random

from models.encoder import *
from models.FNODEs import FNODEs



class LatentNeuralDE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dataset_type = args.dataset_type
        self.encoder = args.encoder
        self.latent_dim = args.latent_dimension

        if args.encoder == 'RNNODE':
            self.encoder = RNNODEEncoder(input_dim=args.encoder_embedding_dim, output_dim=args.encoder_output_dim, rnn_hidden_dim=args.encoder_hidden_dim)
        elif args.encoder == 'Transformer':
            self.encoder = TransformerEncoder(args)
        elif args.encoder == 'BiRNN':
            raise NotImplementedError

        if args.model_type == 'FNODEs':
            self.decoder = FNODEs(args)

    def forward(self, t, x):
        t = torch.squeeze(t[0]).cuda()

        # bucketing ECG dataset
        if self.dataset_type == 'dataset9':
            sample_idxs = self.bucketing(x)
            print(f'Number of sampled time-stamp {len(sample_idxs)}')
            t = t[:, sample_idxs]
            x = x[:, sample_idxs]

        else:
            sample_idxs = self.time_sampling(t)
            t = t[sample_idxs]
            x = x[:, sample_idxs]

        if self.encoder is not None:
            z = self.encoder(x, span=t)
            # z sampling

            x = x.squeeze(-1)
        else:
            z = torch.ones(x.size(0), self.latent_dim, device=x.device)

        decoded_traj = self.decoder(t, x, z)
        mse_loss = nn.MSELoss()(decoded_traj, x)
        return mse_loss

    def predict(self, t, x):
        with torch.no_grad():
            t = torch.squeeze(t[0])

            if self.encoder is not None:
                z = self.encoder(x, span=t)
                x = x.squeeze(-1)
            else:
                z = torch.ones(x.size(0), self.latent_dim, device=x.device)

            decoded_traj = self.decoder(t, x, z)
        return decoded_traj


    def bucketing(self, x):
        cpu_x = x.cpu()[0]
        bins = np.linspace(-2, 2, 50)
        inds = np.digitize(cpu_x, bins)

        k = 20
        sample_idxs = []
        for bucket in bins:
            idxs = np.where(bins[inds] == bucket)[0]
            if len(idxs) < k:
                sample_idxs.extend(idxs)
            else:
                sample_idxs.extend(random.sample(list(idxs), k))

        return sorted(sample_idxs)

    def time_sampling(self, t):
        index = torch.sort(torch.LongTensor(np.random.choice(t.size(0), 250, replace=False)))[0]
        return index
        # index = torch.sort(torch.LongTensor(np.random.choice(t.size(0), 400, replace=False)))[0]
        # t = t[index]
        # x = x[:, index]




class ConditionLatentDE(nn.Module):
    def __init__(self, args):
        super(ConditionLatentDE, self).__init__()
        self.dataset_type = args.dataset_type

        if args.encoder == 'RNNODE':
            self.encoder = RNNODEEncoder(input_dim=args.encoder_embedding_dim, output_dim=args.encoder_output_dim, rnn_hidden_dim=args.encoder_hidden_dim)
        elif args.encoder == 'Transformer':
            raise NotImplementedError

        if args.model_type == 'FNODEs':
            self.decoder = FNODEs(args)

        assert args.dataset_type == 'sin', "number of label in ECG and NSynth is not yet defined"
        self.label_num = {'sin': 4,
                        'ECG': 5,
                        'NSynth': 20}

    def forward(self, t, x, label):
        t = torch.squeeze(t[0]).cuda()
        B = x.size(0)

        if self.dataset_type == 'ECG':
            sample_idxs = self.bucketing(x)
            print(f'Number of sampled time-stamp {len(sample_idxs)}')
            t = t[:, sample_idxs]
            x = x[:, sample_idxs]
        elif self.dataset_type == 'sin':
            sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(0), 300, replace=False)))[0]
            t = t[sample_idxs]
            x = x[:, sample_idxs]

        z = self.encoder(x, span=t)   # sampled z
        # concat label information
        label_embed = torch.zeros(B, self.label_num[self.dataset_type])
        label_embed[range(B), label] = 1

        z = torch.cat((z, label_embed), dim=0)
        x = x.squeeze(-1)

        decoded_traj = self.decoder(t, x, z)
        mse_loss = nn.MSELoss()(decoded_traj, x)
        return mse_loss

    def predict(self, t, x):
        with torch.no_grad():
            t = torch.squeeze(t[0])

            z = self.encoder(x, span=t)
            x = x.squeeze(-1)

            decoded_traj = self.decoder(t, x, z)
        return decoded_traj


    def bucketing(self, x):
        cpu_x = x.cpu()[0]
        bins = np.linspace(-2, 2, 50)
        inds = np.digitize(cpu_x, bins)

        k=20
        sample_idxs = []
        for bucket in bins:
            idxs = np.where(bins[inds] == bucket)[0]
            if len(idxs) < k:
                sample_idxs.extend(idxs)
            else:
                sample_idxs.extend(random.sample(list(idxs), k))

        return sorted(sample_idxs)
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

from models.encoder import *
from models.FNODEs import FNODEs
from models.NeuralProcess import *
from utils.loss import normal_kl



class LatentNeuralDE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dataset_type = args.dataset_type
        self.encoder = args.encoder
        self.latent_dim = args.latent_dimension

        if args.encoder == 'RNNODE':
            self.encoder = RNNODEEncoder(input_dim=args.encoder_embedding_dim, output_dim=args.latent_dimension, rnn_hidden_dim=args.encoder_hidden_dim)
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
            self.encoder = RNNODEEncoder(input_dim=args.encoder_embedding_dim, output_dim=args.latent_dimension, rnn_hidden_dim=args.encoder_hidden_dim)
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



class ConditionalShiftFNP(nn.Module):
    def __init__(self, args):
        super(ConditionalShiftFNP, self).__init__()
        self.dataset_type = args.dataset_type
        self.num_label = args.num_label
        self.latent_dim = args.latent_dimension

        if args.encoder == 'Transformer':
            self.encoder = TransformerEncoder(args=args)
        elif args.encoder == 'Conv':
            self.encoder = ConvEncoder(args=args)

        self.decoder = FNPShiftDecoder(args)

    def forward(self, t, x, label, sampling):
        # t (B, S)  x (B, S, 1)  label (B)
        B = x.size(0)

        # label information
        label_embed = torch.zeros(B, self.num_label).cuda()
        label_embed[range(B), label] = 1

        memory, z, qz0_mean, qz0_logvar = self.encoder(x, label_embed, span=t[0])

        kl_loss = normal_kl(qz0_mean, qz0_logvar, torch.zeros(z.size()).cuda(), torch.zeros(z.size()).cuda()).sum(-1).mean(0)

        # concat label information
        z = torch.cat((z, label_embed), dim=-1)
        x = x.squeeze(-1)

        decoded_traj = self.decoder(t.unsqueeze(-1), z, memory)
        mse_loss = nn.MSELoss()(decoded_traj, x)
        return mse_loss, kl_loss




class ConditionalQueryFNP(nn.Module):
    def __init__(self, args):
        super(ConditionalQueryFNP, self).__init__()
        self.dataset_type = args.dataset_type
        self.num_label = args.num_label
        self.latent_dim = args.latent_dimension
        self.n_harmonics = args.n_harmonics

        if args.encoder == 'RNNODE':
            raise NotImplementedError("change the input argument")
        elif args.encoder == 'Transformer':
            self.encoder = TransformerEncoder(args=args)
        elif args.encoder == 'Conv':
            self.encoder = ConvEncoder(args=args)

        self.decoder = FNP_QueryDecoder(args=args)

    def sampling(self, t, x):
        if self.dataset_type == 'sin':
            sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(-1), 150, replace=False)))[0]
            t = t[:, sample_idxs]  # (150)
            x = x[:, sample_idxs]
        # elif self.dataset_type == 'NSynth':
        #     sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(-1), 8000, replace=False)))[0]   # sampling..
        #     t = t[:, sample_idxs]
        #     x = x[:, sample_idxs]
        # not sampling for ECG for now
        return t, x

    def forward(self, t, x, label, sampling):
        # t (B, 300)  x (B, S, 1)  label(B)
        B = x.size(0)

        # label information
        label_embed = torch.zeros(B, self.num_label).cuda()
        label_embed[range(B), label] = 1

        if sampling:
            sampled_t, sampled_x = self.sampling(t, x)
            memory, z, qz0_mean, qz0_logvar = self.encoder(sampled_x, label_embed, span=sampled_t[0])
        else:
            memory, z, qz0_mean, qz0_logvar = self.encoder(x, label_embed, span=t[0])

        kl_loss = normal_kl(qz0_mean, qz0_logvar, torch.zeros(z.size()).cuda(), torch.zeros(z.size()).cuda()).sum(-1).mean(0)

        # concat label information
        z = torch.cat((z, label_embed), dim=-1)
        x = x.squeeze(-1)

        decoded_traj = self.decoder(t.unsqueeze(-1), z)
        mse_loss = nn.MSELoss()(decoded_traj, x)

        # orthogonal matrix
        harmonic_embedding = self.decoder.coeff_generator.harmonic_embedding.weight   # (H, 2E)
        harmonic_embedding = F.normalize(harmonic_embedding, dim=1, p=2)   # normalize
        weight_mat = torch.matmul(harmonic_embedding, harmonic_embedding.T)   # (H, H)
        weight_mat = (weight_mat - torch.eye(self.n_harmonics, self.n_harmonics).cuda())
        orthogonal_loss = torch.norm(weight_mat, p='fro')

        return mse_loss, kl_loss, orthogonal_loss

    def predict(self, t, x, label, test_t):
        with torch.no_grad():
            B = x.size(0)
            label_embed = torch.zeros(B, self.num_label).cuda()
            label_embed[range(B), label] = 1

            memory, z, qz0_mean, qz0_logvar = self.encoder(x, label_embed, span=t[0])
            z = torch.cat((z, label_embed), dim=-1)
            decoded_traj = self.decoder(test_t.unsqueeze(-1), z)

        return decoded_traj

    def inference(self, t, label):
        with torch.no_grad():
            z = torch.randn(1, self.latent_dim).cuda()
            label_embed = torch.zeros(1, self.num_label).cuda()
            label_embed[0, label] = 1
            z = torch.cat((z, label_embed), dim=-1)

            decoded_traj = self.decoder(t.unsqueeze(-1), z)
        return decoded_traj


class ConditionalQueryShiftFNP(nn.Module):
    def __init__(self, args):
        super(ConditionalQueryShiftFNP, self).__init__()
        self.dataset_type = args.dataset_type
        self.num_label = args.num_label
        self.latent_dim = args.latent_dimension

        if args.encoder == 'RNNODE':
            raise NotImplementedError("change the input argument")
        elif args.encoder == 'Transformer':
            self.encoder = TransformerEncoder(args=args)
        elif args.encoder == 'Conv':
            self.encoder = ConvEncoder(args=args)

        self.decoder = FNP_QueryShiftDecoder(args=args)

    def sampling(self, t, x):
        if self.dataset_type == 'sin':
            sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(-1), 150, replace=False)))[0]
            t = t[:, sample_idxs]  # (150)
            x = x[:, sample_idxs]
        # elif self.dataset_type == 'NSynth':
        #     sample_idxs = torch.sort(torch.LongTensor(np.random.choice(t.size(-1), 8000, replace=False)))[0]   # sampling..
        #     t = t[:, sample_idxs]
        #     x = x[:, sample_idxs]
        # not sampling for ECG for now
        return t, x

    def forward(self, t, x, label, sampling):
        # t (B, 300)  x (B, S, 1)  label(B)
        assert not sampling, 'We do not care sampling for now'
        B = x.size(0)

        # label information
        label_embed = torch.zeros(B, self.num_label).cuda()
        label_embed[range(B), label] = 1

        if sampling:
            sampled_t, sampled_x = self.sampling(t, x)
            memory, z, qz0_mean, qz0_logvar = self.encoder(sampled_x, label_embed, span=sampled_t[0])
        else:
            memory, z, qz0_mean, qz0_logvar = self.encoder(x, label_embed, span=t[0])

        kl_loss = normal_kl(qz0_mean, qz0_logvar, torch.zeros(z.size()).cuda(), torch.zeros(z.size()).cuda()).sum(-1).mean(0)

        # concat label information
        z = torch.cat((z, label_embed), dim=-1)
        x = x.squeeze(-1)

        decoded_traj = self.decoder(t.unsqueeze(-1), z, memory)
        mse_loss = nn.MSELoss()(decoded_traj, x)

        return mse_loss, kl_loss

    def predict(self, t, x, label, test_t):
        with torch.no_grad():
            B = x.size(0)
            label_embed = torch.zeros(B, self.num_label).cuda()
            label_embed[range(B), label] = 1

            memory, z, qz0_mean, qz0_logvar = self.encoder(x, label_embed, span=t[0])
            z = torch.cat((z, label_embed), dim=-1)
            decoded_traj = self.decoder(test_t.unsqueeze(-1), z, memory)

        return decoded_traj

    def inference(self, t, label):
        with torch.no_grad():
            z = torch.randn(1, self.latent_dim).cuda()
            label_embed = torch.zeros(1, self.num_label).cuda()
            label_embed[0, label] = 1
            z = torch.cat((z, label_embed), dim=-1)

            decoded_traj = self.decoder(t.unsqueeze(-1), z)
        return decoded_traj


## Continual Learning
class ConditionalQueryContinualFNP(nn.Module):
    def __init__(self, args):
        super(ConditionalQueryContinualFNP, self).__init__()
        self.dataset_type = args.dataset_type
        self.num_label = args.num_label
        self.latent_dim = args.latent_dimension

        if args.encoder == 'Transformer':
            self.encoder = TransformerEncoder(args=args)
        elif args.encoder == 'Conv':
            self.encoder = ConvEncoder(args=args)

        self.decoder = FNP_QueryContinualDecoder(args=args)

    def forward(self, t, x, label, sampling, coeff_num):
        B = x.size(0)

        # label information
        label_embed = torch.zeros(B, self.num_label).cuda()
        label_embed[range(B), label] = 1

        memory, z, qz0_mean, qz0_logvar = self.encoder(x, label_embed, span=t[0])
        kl_loss = normal_kl(qz0_mean, qz0_logvar, torch.zeros(z.size()).cuda(), torch.zeros(z.size()).cuda()).sum(-1).mean(0)

        # concat label information
        z = torch.cat((z, label_embed), dim=-1)
        x = x.squeeze(-1)

        decoded_traj = self.decoder(t.unsqueeze(-1), z, coeff_num)
        mse_loss = nn.MSELoss()(decoded_traj, x)

        return mse_loss, kl_loss

    def predict(self, t, x, label, test_t, coeff_num):
        with torch.no_grad():
            B = x.size(0)
            label_embed = torch.zeros(B, self.num_label).cuda()
            label_embed[range(B), label] = 1

            memory, z, qz0_mean, qz0_logvar = self.encoder(x, label_embed, span=t[0])

            z = torch.cat((z, label_embed), dim=-1)
            decoded_traj = self.decoder(test_t.unsqueeze(-1), z, coeff_num)
            return decoded_traj



class NonConditionalQueryFNP(nn.Module):
    def __init__(self, args):
        super(NonConditionalQueryFNP, self).__init__()
        self.dataset_type = args.dataset_type
        self.n_harmonics = args.n_harmonics

        if args.encoder == 'Transformer':
            self.encoder = UnconditionalTransformerEncoder(args=args)
        elif args.encoder == 'Conv':
            self.encoder = UnconditionConvEncoder(args=args)
        elif args.encoder == 'TransConv':
            self.encoder = UnconditionTransConvEncoder(args=args)

        self.decoder = FNP_UnconditionQueryDecoder(args=args)

    def forward(self, t, x):
        # No sampling / label for now
        # t (B, S)  x (B, S, 1)

        memory, z0, qz0_mean, qz0_logvar = self.encoder(x, span=t[0])

        kl_loss = normal_kl(qz0_mean, qz0_logvar, torch.zeros(z0.size()).cuda(), torch.zeros(z0.size()).cuda()).sum(-1).mean(0)

        x = x.squeeze(-1)
        decoded_traj = self.decoder(t.unsqueeze(-1), z0)
        mse_loss = nn.MSELoss()(decoded_traj, x)

        # orthogonal loss if necessary
        harmonic_embedding = self.decoder.coeff_generator.harmonic_embedding.weight   # (H, E)
        harmonic_embedding = F.normalize(harmonic_embedding, dim=1, p=2)   # normalize
        weight_mat = torch.matmul(harmonic_embedding, harmonic_embedding.T)  # (H, H)
        weight_mat = (weight_mat - torch.eye(self.n_harmonics, self.n_harmonics).cuda())
        orthonormal_loss = torch.norm(weight_mat, p='fro')

        return mse_loss, kl_loss, orthonormal_loss


class AEQueryFNP(nn.Module):
    def __init__(self, args):
        super(AEQueryFNP, self).__init__()
        self.dataset_type = args.dataset_type
        self.n_harmonics = args.n_harmonics

        if args.encoder == 'Transformer':
            self.encoder = UnconditionalTransformerEncoder(args=args)
        elif args.encoder == 'Conv':
            self.encoder = UnconditionConvEncoder(args=args)
        elif args.encoder == 'TransConv':
            self.encoder = UnconditionTransConvEncoder(args=args)

        self.decoder = FNP_UnconditionQueryDecoder(args=args)

    def forward(self, t, x):
        # No sampling / label for now
        # t (B, S)  x (B, S, 1)

        memory = self.encoder(x, span=t[0])

        # kl_loss = normal_kl(qz0_mean, qz0_logvar, torch.zeros(z0.size()).cuda(), torch.zeros(z0.size()).cuda()).sum(-1).mean(0)

        x = x.squeeze(-1)
        decoded_traj = self.decoder(t.unsqueeze(-1), memory)
        mse_loss = nn.MSELoss()(decoded_traj, x)

        # orthogonal loss if necessary
        harmonic_embedding = self.decoder.coeff_generator.harmonic_embedding.weight   # (H, E)
        harmonic_embedding = F.normalize(harmonic_embedding, dim=1, p=2)   # normalize
        weight_mat = torch.matmul(harmonic_embedding, harmonic_embedding.T)  # (H, H)
        weight_mat = (weight_mat - torch.eye(self.n_harmonics, self.n_harmonics).cuda())
        orthonormal_loss = torch.norm(weight_mat, p='fro')

        return mse_loss, orthonormal_loss


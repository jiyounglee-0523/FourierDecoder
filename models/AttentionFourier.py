import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from models.encoder import UnconditionalTransformerEncoder, UnconditionConvEncoder
from utils.loss import normal_kl

# calculate attention between cos/sin and original signal and use it as amplitude
class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.n_harmonics = args.n_harmonics
        self.lower_bound, self.upper_bound = args.lower_bound, args.upper_bound

        # harmonic embedding
        self.key_embedding = nn.Embedding(args.n_harmonics, args.latent_dimension)
        self.query = nn.Linear(args.latent_dimension, args.latent_dimension)

    def forward(self, query):
        # query (B, E)
        query = self.query(query)  # (B, E)
        key = torch.linspace(self.lower_bound, self.upper_bound, self.n_harmonics, requires_grad=False, dtype=torch.long).cuda()
        key = self.key_embedding(key - 1)    # (H, E)
        key = F.normalize(key, dim=1, p=2)
        attn_weight = torch.matmul(query, key.T)  # (B, H)
        return attn_weight


# share the basis embedding -> showed the same result as separate embedding
class OneAttention(nn.Module):
    def __init__(self, args):
        super(OneAttention, self).__init__()
        self.n_harmonics = args.n_harmonics
        self.lower_bound, self.upper_bound = args.lower_bound, args.upper_bound

        # harmonic
        self.harmonic_embedding = nn.Embedding(args.n_harmonics, args.latent_dimension)
        self.sin_weight = nn.Linear(args.latent_dimension, args.latent_dimension)
        self.cos_weight = nn.Linear(args.latent_dimension, args.latent_dimension)
        self.query = nn.Linear(args.latent_dimension, args.latent_dimension)

    def forward(self, query):
        # query (B, E)
        query = self.query(query)   # (B, E)
        key = torch.linspace(self.lower_bound, self.upper_bound, self.n_harmonics, requires_grad=False, dtype=torch.long).cuda()
        key = self.harmonic_embedding(key - 1)  # (H, E)
        sin_key = self.sin_weight(key)  # (H, E)
        cos_key = self.cos_weight(key)  # (H, E)

        sin_attn_weight = torch.matmul(query, sin_key.T)
        cos_attn_weight = torch.matmul(query, cos_key.T)
        return sin_attn_weight, cos_attn_weight


# TODO: remae the module name...
class NonperiodicDecoder(nn.Module):
    def __init__(self, args):
        super(NonperiodicDecoder, self).__init__()

        layers = []
        layers.append(nn.Linear(args.latent_dimension + 1, args.decoder_hidden_dim))
        layers.append(nn.SiLU())

        for i in range(args.decoder_layers):
            layers.append(nn.Linear(args.decoder_hidden_dim, args.decoder_hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(args.decoder_hidden_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, r, target_x):
        # query (B, E), target_x = (B, S, 1)
        S = target_x.size()[1]
        r = r.unsqueeze(1).repeat(1, S, 1)   # (B, S, E)

        # concat query and timestamp
        input_pairs = torch.cat((target_x, r), dim=-1)   # (B, S, E+1)

        output = self.model(input_pairs)   # (B, S, 1)
        return output


class UnconditionalAttnDecoder(nn.Module):
    def __init__(self, args):
        super(UnconditionalAttnDecoder, self).__init__()
        self.lower_bound, self.upper_bound, self.n_harmonics = args.lower_bound, args.upper_bound, args.n_harmonics
        self.skip_step = args.skip_step

        self.sin_attn = Attention(args)
        self.cos_attn = Attention(args)
        self.nonperiodic_decoder = NonperiodicDecoder(args)

    def forward(self, target_x, r):
        # target_x = (B, S, 1),   r (B, E)
        nonperiodic_signal = self.nonperiodic_decoder(r, target_x).squeeze(-1)   # (B, S)

        sin_coeffs = self.sin_attn(r)  # (B, H)
        cos_coeffs = self.cos_attn(r)  # (B, H)
        # sin_coeffs, cos_coeffs = self.attn(r)
        self.sin_coeffs = sin_coeffs ; self.cos_coeffs = cos_coeffs

        # make cos / sin matrix
        cos_x = torch.cos(target_x * self.lower_bound * 2 * math.pi)
        sin_x = torch.sin(target_x * self.lower_bound * 2 * math.pi)
        for i in range(int(self.lower_bound + self.skip_step), int(self.upper_bound + self.skip_step), int(self.skip_step)):
            cos_x = torch.cat((cos_x, torch.cos(target_x * 2 * i * math.pi)), dim=-1)   # (B, S, H)
            sin_x = torch.cat((sin_x, torch.sin(target_x * 2 * i * math.pi)), dim=-1)   # (B, S, H)

        cos_x = torch.mul(cos_x, cos_coeffs.unsqueeze(1))
        sin_x = torch.mul(sin_x, sin_coeffs.unsqueeze(1))

        cos_x = cos_x.sum(-1) ; sin_x = sin_x.sum(-1)
        periodic_signal = (cos_x + sin_x)  # (B, S)
        return nonperiodic_signal + periodic_signal




"""
class UnconditionalAttnFNP(nn.Module):
    def __init__(self, args):
        super(UnconditionalAttnFNP, self).__init__()
        self.dataset_type = args.dataset_type
        self.n_harmonics = args.n_harmonics

        if args.encoder == 'Transformer':
            self.encoder = UnconditionalTransformerEncoder(args=args)
        elif args.encoder == 'Conv':
            self.encoder = UnconditionConvEncoder(args=args)

        self.decoder = UnconditionalAttnDecoder(args=args)

    def forward(self, t, x):
        # No sampling nor label for now
        # t (B, S)  x (B, S, 1)

        memory, z0, qz0_mean, qz0_logvar = self.encoder(x, span=t[0])

        # kl_loss = normal_kl(qz0_mean, qz0_logvar, torch.zeros(z0.size()).cuda(), torch.zeros(z0.size()).cuda()).sum(-1).mean(0)
        kl_loss = torch.zeros(1)

        x = x.squeeze(-1)
        decoded_traj = self.decoder(t.unsqueeze(-1), z0)
        # decoded_traj = self.decoder(t.unsqueeze(-1), memory)    # no VAE
        mse_loss = nn.MSELoss()(decoded_traj, x)

        # orthonormal loss
        # sin
        sin_harmonic_embedding = self.decoder.sin_attn.key_embedding.weight     # (H, E)
        sin_harmonic_embedding = F.normalize(sin_harmonic_embedding, dim=1, p=2)
        sin_weight_mat = torch.matmul(sin_harmonic_embedding, sin_harmonic_embedding.T)  # (H, H)
        sin_weight_mat = (sin_weight_mat - torch.eye(self.n_harmonics, self.n_harmonics).cuda())
        sin_orthonormal_loss = torch.norm(sin_weight_mat, p='fro')

        # cos
        cos_harmonic_embedding = self.decoder.cos_attn.key_embedding.weight  # (H, E)
        cos_harmonic_embedding = F.normalize(cos_harmonic_embedding, dim=1, p=2)
        cos_weight_mat = torch.matmul(cos_harmonic_embedding, cos_harmonic_embedding.T)  # (H, H)
        cos_weight_mat = (cos_weight_mat - torch.eye(self.n_harmonics, self.n_harmonics).cuda())
        cos_orthonormal_loss = torch.norm(cos_weight_mat, p='fro')

        orthonormal_loss = sin_orthonormal_loss + cos_orthonormal_loss

        # harmonic_embedding = self.decoder.attn.harmonic_embedding.weight # (H, E)
        # harmonic_embedding = F.normalize(harmonic_embedding, dim=1, p=2)
        # weight_mat = torch.matmul(harmonic_embedding, harmonic_embedding.T)  # (H, H)
        # weight_mat = (weight_mat - torch.eye(self.n_harmonics, self.n_harmonics).cuda())
        # orthonormal_loss = torch.norm(weight_mat, p='fro')

        return mse_loss, kl_loss, orthonormal_loss
"""
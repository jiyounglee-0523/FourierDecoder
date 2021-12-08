import torch
import torch.nn as nn

import math


class QueryGenerator(nn.Module):
    def __init__(self, args):
        super(QueryGenerator, self).__init__()
        self.n_harmonics = int(args.n_harmonics)
        self.lower_bound, self.upper_bound = args.lower_bound, args.upper_bound

        layers = []
        layers.append(nn.Linear(args.latent_dimension + args.num_label, 2*args.latent_dimension))
        layers.append(nn.SiLU())

        for i in range(args.decoder_layers):
            layers.append(nn.Linear(2*args.latent_dimension, 2*args.latent_dimension))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(2*args.latent_dimension, 2))
        self.model = nn.Sequential(*layers)

        # harmonic embedding
        self.harmonic_embedding = nn.Embedding(args.n_harmonics, args.latent_dimension + args.num_label)
        self.harmonics = torch.linspace(0, self.n_harmonics-1, self.n_harmonics, requires_grad=False, dtype=torch.long).cuda()

    def forward(self, x):
        # (B, E + label)
        B, E = x.size()

        # broadcast to the number of harmonics
        x = torch.broadcast_to(x.unsqueeze(1), (B, self.n_harmonics, E))  # (B, H, E)

        harmonics = self.harmonic_embedding(self.harmonics)  # (B, H, E)
        harmonics = torch.broadcast_to(harmonics.unsqueeze(0), (B, self.n_harmonics, E))
        x = x + harmonics
        x = self.model(x)
        return x

class ConditionalFNP(nn.Module):
    def __init__(self, args):
        super(ConditionalFNP, self).__init__()
        self.lower_bound, self.upper_bound, self.n_harmonics = args.lower_bound, args.upper_bound, args.n_harmonics
        self.skip_step = args.skip_step
        self.NP = args.NP

        # harmonic embedding
        self.coeff_generator = QueryGenerator(args)

    def forward(self, target_x, z, x):
        # target_x (B, S, 1)  r (B, E)

        coeffs = self.coeff_generator(z)
        self.coeffs = coeffs
        sin_coeffs = coeffs[:, :, 0]
        cos_coeffs = coeffs[:, :, 1]

        # make cos / sin matrix
        cos_x = torch.cos(target_x * self.lower_bound * 2 * math.pi)
        sin_x = torch.sin(target_x * self.lower_bound * 2 * math.pi)
        for i in range(int(self.lower_bound + self.skip_step), int(self.upper_bound + self.skip_step),
                       int(self.skip_step)):
            cos_x = torch.cat((cos_x, torch.cos(target_x * 2 * i * math.pi)), dim=-1)  # (B, S, H)
            sin_x = torch.cat((sin_x, torch.sin(target_x * 2 * i * math.pi)), dim=-1)  # (B, S, H)

        cos_x = torch.mul(cos_x, cos_coeffs.unsqueeze(1))
        sin_x = torch.mul(sin_x, sin_coeffs.unsqueeze(1))

        cos_x = cos_x.sum(-1)
        sin_x = sin_x.sum(-1)
        periodic_signal = (cos_x + sin_x)  # (B, S)
        return periodic_signal